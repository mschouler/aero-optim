import dill as pickle
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from typing import Any

from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)
from aero_optim.mf_sm.mf_infill import maximize_ED, minimize_LCB, maximize_MPI_BO, compute_pareto
from aero_optim.mf_sm.mf_models import MfDNN, MultiObjectiveModel, MfSMT
from aero_optim.optim.evolution import PymooEvolution
from aero_optim.optim.optimizer import WolfOptimizer
from aero_optim.optim.pymoo_optimizer import PymooWolfOptimizer
from aero_optim.simulator.simulator import Simulator
from aero_optim.utils import check_dir, check_file, cp_filelist, replace_in_file

logger = logging.getLogger()


class CustomSimulator(Simulator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.set_model(config["simulator"]["model_file"])

    def process_config(self):
        logger.info("processing config..")
        if "model_file" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <model_file> entry in {self.config['simulator']}")

    def set_solver_name(self):
        self.solver_name = self.config["optim"]["model_name"]

    def set_model(self, model_file: str):
        check_file(model_file)
        with open(model_file, "rb") as handle:
            self.model = pickle.load(handle)

    def execute_sim(self, candidates: list[float] | np.ndarray, gid: int = 0):
        logger.info(f"execute simulations g{gid} with {self.solver_name}")
        QoI: list[np.ndarray] = self.model.evaluate(np.array(candidates))
        self.df_dict[gid] = {
            cid: pd.DataFrame({"ResTot": 1., "loss_ADP": QoI[0][cid], "loss_OP": QoI[1][cid]})
            for cid in range(len(candidates))
        }


class CustomOptimizer(PymooWolfOptimizer):
    def __init__(self, config: dict):
        """
         **Inner**

        - feasible_cid (dict[int, list[int]]): dictionary containing feasible cid of each gid.
        """
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=2, n_ieq_constr=4, xl=self.bound[0], xu=self.bound[1]
        )
        self.feasible_cid: dict[int, list[int]] = {}

    def set_inner(self):
        """
        **Sets** some inner attributes:

        - bsl_w_ADP (float)
        - bsl_w_OP (float)
        - bsl_camber_th (tuple[np.ndarray, float, float, np.ndarray])
        - bsl_area (float)
        - bsl_c (float)
        - bsl_c_ax (float)
        - bsl_cog (np.ndarray)
        - bsl_cog_x (float)
        - infill_nb (int): adaptive infill number.
        - infill_lf_size (int): number of new low fidelity candidates for each adaptive infill.
        - infill_hf_size (int): number of new high fidelity candidates for each adaptive infill.
        - infill_nb_gen (int): number of generations for each infill adaptation algorithm.
        - bayesian_infill (bool): infill strategy: bayesian (True) or not (False).
        - lf_config (str): path to the hf simulation config template.
        - hf_config (str): path to the hf simulation config template.
        - infill_ctr (int): infill counter.
        """
        # constraints
        self.bsl_w_ADP = self.config["optim"]["bsl_hf_w_ADP"]
        self.bsl_w_OP = self.config["optim"]["bsl_hf_w_OP"]
        bsl_pts = self.ffd.pts
        self.bsl_c, self.bsl_c_ax = get_chords(bsl_pts)
        logger.info(f"baseline chord = {self.bsl_c} m, baseline axial chord = {self.bsl_c_ax}")
        bsl_upper, bsl_lower = split_profile(bsl_pts)
        self.bsl_camber_th = get_camber_th(bsl_upper, bsl_lower, interpolate=True)
        self.bsl_th_over_c = self.bsl_camber_th[1] / self.bsl_c
        self.bsl_Xth_over_cax = self.bsl_camber_th[2] / self.bsl_c_ax
        logger.info(f"baseline th_max = {self.bsl_camber_th[1]} m, "
                    f"Xth_max {self.bsl_camber_th[2]} m, "
                    f"th_max / c = {self.bsl_th_over_c}, "
                    f"Xth_max / c_ax = {self.bsl_Xth_over_cax}")
        self.bsl_area = get_area(bsl_pts)
        self.bsl_area_over_c2 = self.bsl_area / self.bsl_c**2
        logger.info(f"baseline area = {self.bsl_area} m2, "
                    f"baseline area / (c * c) = {self.bsl_area_over_c2}")
        self.bsl_cog = get_cog(bsl_pts)
        self.bsl_Xcg_over_cax = self.bsl_cog[0] / self.bsl_c_ax
        logger.info(f"baseline X_cg over c_ax = {self.bsl_Xcg_over_cax}")
        # infill
        self.infill_nb = self.config["optim"]["infill_nb"]
        self.infill_lf_size = self.config["optim"]["infill_lf_size"]
        self.infill_hf_size = self.config["optim"]["infill_hf_size"]
        self.infill_nb_gen = self.config["optim"]["infill_nb_gen"]
        self.bayesian_infill = self.config["optim"]["bayesian_infill"]
        self.lf_config = self.config["optim"]["lf_config"]
        self.hf_config = self.config["optim"]["hf_config"]
        self.infill_ctr = 0

    def set_gmsh_mesh_class(self):
        self.MeshClass = None

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Computes** the objective function and constraints for each candidate in the generation.

        Note:
            for this use-case, constraints can be computed before simulations.
            Unfeasible candidates are not simulated.
        """
        gid = self.gen_ctr
        self.feasible_cid[gid] = []

        # compute candidates constraints
        out["G"] = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        for cid in range(len(X)):
            if cid in self.feasible_cid[gid]:
                loss_ADP = self.simulator.df_dict[gid][cid]["loss_ADP"].iloc[-1]
                loss_OP = self.simulator.df_dict[gid][cid]["loss_OP"].iloc[-1]
                logger.info(f"g{gid}, c{cid}: "
                            f"w_ADP = {loss_ADP}, w_OP = {loss_OP}")
                self.J.append([loss_ADP, loss_OP])
            else:
                self.J.append([float("nan"), float("nan")])
        out["F"] = np.row_stack(self.J[-self.doe_size:])

        self._observe(out["F"])
        self.gen_ctr += 1

    def execute_constrained_candidates(self, candidates: np.ndarray, gid: int) -> np.ndarray:
        """
        **Executes** feasible candidates only and **waits** for them to finish.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        constraint = []
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            ffd_file, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)
            logger.info(f"candidate g{gid}, c{cid} constraint computation..")
            constraint.append(self.apply_candidate_constraints(ffd_profile, gid, cid))
            # only mesh and execute feasible candidates
            if len([v for v in constraint[cid] if v > 0.]) == 0:
                self.feasible_cid[gid].append(cid)
            else:
                logger.info(f"unfeasible candidate g{gid}, c{cid}")

        self.execute_candidates(candidates, gid)
        return np.row_stack(constraint)

    def apply_candidate_constraints(
            self, profile: np.ndarray, gid: int | str, cid: int
    ) -> list[float]:
        """
        **Computes** various relative and absolute constraints of a given candidate
        and **returns** their values as a list of floats.

        Note:
            when some constraint is violated, a graph is also generated.
        """
        # relative constraints
        # thmax / c:        +/- 30%
        # Xthmax / c_ax:    +/- 20%
        upper, lower = split_profile(profile)
        c, c_ax = get_chords(profile)
        camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
        th_over_c = thmax / c
        Xth_over_cax = Xthmax / c_ax
        logger.debug(f"th_max = {thmax} m, Xth_max {Xthmax} m")
        logger.debug(f"th_max / c = {th_over_c}, Xth_max / c_ax = {Xth_over_cax}")
        th_cond = abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3
        logger.debug(f"th_max / c: {'violated' if th_cond > 0 else 'not violated'} ({th_cond})")
        Xth_cond = abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2
        logger.debug(f"Xth_max / c_ax: {'violated' if Xth_cond > 0 else 'not violated'} "
                     f"({Xth_cond})")
        # area / (c * c):   +/- 20%
        area = get_area(profile)
        area_over_c2 = area / c**2
        area_cond = abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2
        logger.debug(f"area / (c * c): {'violated' if area_cond > 0 else 'not violated'} "
                     f"({area_cond})")
        # X_cg / c_ax:      +/- 20%
        cog = get_cog(profile)
        Xcg_over_cax = cog[0] / c_ax
        cog_cond = abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2
        logger.debug(f"X_cg / c_ax: {'violated' if cog_cond > 0 else 'not violated'} ({cog_cond})")
        # absolute constraints
        O_le, O_te = get_circle_centers(upper[:, :2], lower[:, :2])
        le_circle = get_circle(O_le, 0.005 * c)
        te_circle = get_circle(O_te, 0.005 * c)
        le_radius_cond = get_radius_violation(profile, O_le, 0.005 * c)
        logger.debug(f"le radius: {'violated' if le_radius_cond > 0 else 'not violated'} "
                     f"({le_radius_cond})")
        te_radius_cond = get_radius_violation(profile, O_te, 0.005 * c)
        logger.debug(f"te radius: {'violated' if te_radius_cond > 0 else 'not violated'} "
                     f"({te_radius_cond})")
        if cog_cond > 0:
            fig_name = os.path.join(self.figdir, f"profile_g{gid}_c{cid}.png")
            plot_profile(profile, cog, fig_name)
        if th_cond > 0 or Xth_cond > 0 or area_cond > 0:
            fig_name = os.path.join(self.figdir, f"sides_g{gid}_c{cid}.png")
            plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
        return [th_cond, Xth_cond, area_cond, cog_cond]

    def execute_candidates(self, candidates, gid: int):
        logger.info(f"evaluating candidates of generation {gid}..")
        self.simulator.execute_sim(candidates, gid)

    def _observe(self, pop_fitness: np.ndarray):
        """
        **Plots** some results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the candidates fitnesses,</br>
        > the baseline and deformed profiles.
        """
        gid = self.gen_ctr

        # plot settings
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]
        cmap = mpl.colormaps[self.cmap].resampled(self.doe_size)
        colors = cmap(np.linspace(0, 1, self.doe_size))
        # subplot construction
        fig = plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(2, 1, 1)  # profiles
        ax2 = plt.subplot(2, 1, 2)  # fitness (loss_ADP vs loss_OP)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        # loop over candidates through the last generated profiles
        logger.debug(f"pop fitness: {pop_fitness} and shape {pop_fitness.shape}")
        for cid in self.feasible_cid[gid]:
            ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[cid], label=f"c{cid}")
            ax2.scatter(pop_fitness[cid, 0], pop_fitness[cid, 1],
                        color=colors[cid], label=f"c{cid}")
        ax2.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="baseline")
        # legend and title
        fig.suptitle(
            f"Generation {gid} results", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom
        ax2.set_title(f"{self.QoI} ADP vs {self.QoI} OP", weight="bold")
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_xlabel('$w_\\text{ADP}$')
        ax2.set_ylabel('$w_\\text{OP}$')
        # save figure as png
        fig_name = f"pymoo_g{gid}.png"
        logger.info(f"saving {fig_name} to {self.figdir}")
        plt.savefig(os.path.join(self.figdir, fig_name), bbox_inches='tight')
        plt.close()

    def final_observe(self, best_candidates: np.ndarray):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        gen_fitness = np.row_stack(self.J)

        # plotting data: last gen. pareto, y_hf infill, hf baseline
        nsga = compute_pareto(gen_fitness[-self.doe_size:, 0], gen_fitness[-self.doe_size:, 1])
        ax.plot(nsga[:, 0], nsga[:, 1],
                color="forestgreen", label=f"g{len(gen_fitness) // self.doe_size - 1}")
        w_ADP_hf, w_OP_hf = get_hf_DOE(self.simulator.model)
        ax.scatter(w_ADP_hf, w_OP_hf,
                   marker="s", color="k", facecolors="none", label="hf DOE & infill")
        ax.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="hf baseline")
        ax.plot()
        ax.set_axisbelow(True)
        plt.grid(True, color="grey", linestyle="dashed")

        # legend and title
        ax.set_title(f"Optimization evolution ({self.gen_ctr} g. x {self.doe_size} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('$w_\\text{ADP}$')
        ax.set_ylabel('$w_\\text{OP}$')

        # save figure as png
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()


def get_hf_DOE(model: MfDNN | MultiObjectiveModel) -> tuple[np.ndarray, np.ndarray]:
    """
    **Returns** the two objectives high-fidelity DOEs
    """
    if isinstance(model, MultiObjectiveModel):
        assert model.models[0].y_hf_DOE is not None
        assert model.models[1].y_hf_DOE is not None
        w_ADP_hf = model.models[0].y_hf_DOE
        w_OP_hf = model.models[1].y_hf_DOE
    elif isinstance(model, MfDNN):
        assert model.y_hf_DOE is not None
        w_ADP_hf = model.y_hf_DOE[:, 0]
        w_OP_hf = model.y_hf_DOE[:, 1]
    else:
        raise Exception(f"{type(model)} is currently not supported")
    return w_ADP_hf, w_OP_hf


def compute_bayesian_infill(
        model: MfDNN | MultiObjectiveModel,
        infill_lf_size: int,
        infill_nb_gen: int,
        n_design: int,
        bound: tuple[Any],
        seed: int
) -> np.ndarray:
    """
    **Computes** the low fidelity Bayesian infill candidates.
    """
    assert isinstance(model, MultiObjectiveModel)
    # Probability of Improvement
    infill_lf = maximize_MPI_BO(model, n_design, bound, seed, infill_nb_gen)
    # Lower Confidence Bound /objective 1
    assert isinstance(model.models[0], MfSMT)
    infill_lf_LCB_1 = minimize_LCB(model.models[0], n_design, bound, seed, infill_nb_gen)
    infill_lf = np.vstack((infill_lf, infill_lf_LCB_1))
    # Lower Confidence Bound /objective 2
    assert isinstance(model.models[1], MfSMT)
    infill_lf_LCB_2 = minimize_LCB(model.models[1], n_design, bound, seed, infill_nb_gen)
    infill_lf = np.vstack((infill_lf, infill_lf_LCB_2))
    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    current_DOE = np.vstack((current_DOE, infill_lf))
    for _ in range(infill_lf_size - 3):
        infill_lf_ED = maximize_ED(current_DOE, n_design, bound, seed, infill_nb_gen)
        infill_lf = np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
    return infill_lf


def compute_non_bayesian_infill(
    model: MfDNN | MultiObjectiveModel,
    pareto_cand: np.ndarray,
    pareto_fit: np.ndarray,
    infill_lf_size: int,
    infill_nb_gen: int,
    n_design: int,
    bound: tuple[Any],
    seed: int
) -> np.ndarray:
    """
    **Computes** the low fidelity non-Bayesian infill candidates.
    """
    if len(pareto_fit) == 1:
        infill_lf = pareto_cand[0]
        n_p = 1
    elif len(pareto_fit) == 2:
        infill_lf = pareto_cand[:2]
        n_p = 2
    else:
        # matrix made of the distance between each point in the ordered pareto front
        d_matrix = np.linalg.norm(pareto_fit[:, np.newaxis] - pareto_fit[np.newaxis, :], axis=-1)
        # 1d array with the distance between two consecutive points along the ordered pareto front
        s = np.array([d_matrix[i + 1, i] for i in range(len(d_matrix) - 1)])
        s_length = np.sum(s)
        # index of the closest point to the pareto set center
        idx = np.argmin([abs(np.sum(s[:i]) - s_length / 2.) for i in range(len(s))])
        infill_lf = pareto_cand[idx]
        # best candidate wrt 1st objective
        infill_lf = np.vstack((infill_lf, pareto_cand[0]))
        # best candidate wrt 2nd objective
        infill_lf = np.vstack((infill_lf, pareto_cand[-1]))
        n_p = 3
    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    for _ in range(infill_lf_size - n_p):
        infill_lf_ED = maximize_ED(current_DOE, n_design, bound, seed, infill_nb_gen)
        infill_lf = np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
    return infill_lf


def execute_infill(
        X: np.ndarray, config: str, n_design: int, outdir: str, ite: int, fidelity: str
) -> list[np.ndarray]:
    """
    **Executes** infill candidates and returns their associated fitnesses.
    """
    name = f"{fidelity}_infill_{ite}"
    df_dict = execute_single_gen(
        X=X,
        config=config,
        outdir=os.path.join(outdir, name),
        name=name,
        n_design=n_design
    )
    loss_ADP = np.array(
        [df_dict[0][cid]["ADP"]["LossCoef"].iloc[-1] for cid in range(len(df_dict[0]))]
    )
    loss_OP = np.array(
        [0.5 * (df_dict[0][cid]["OP1"]["LossCoef"].iloc[-1]
                + df_dict[0][cid]["OP2"]["LossCoef"].iloc[-1])
            for cid in range(len(df_dict[0]))]
    )
    assert len(loss_ADP) == len(np.atleast_2d(X))
    return [loss_ADP, loss_OP]


def execute_single_gen(
        X: np.ndarray, config: str, outdir: str, name: str, n_design: int = 0
) -> dict[int, dict[int, pd.DataFrame]]:
    """
    **Executes** a single generation of candidates.
    """
    check_file(config)
    check_dir(outdir)
    cp_filelist([config], [outdir])
    config_path = os.path.join(outdir, config)
    custom_doe = os.path.join(outdir, f"{name}.txt")
    np.savetxt(custom_doe, np.atleast_2d(X))
    # updates @outdir, @n_design, @doe_size, @custom_doe
    # Note: @n_design is the number of FFD control points even when using POD
    config_args = {
        "@outdir": outdir,
        "@n_design": f"{n_design if n_design else np.atleast_2d(X).shape[1]}",
        "@doe_size": f"{np.atleast_2d(X).shape[0]}",
        "@custom_doe": f"{custom_doe}"
    }
    replace_in_file(config_path, config_args)
    # execute single generation
    exec_cmd = ["optim", "-c", f"{config_path}", "-v", "3", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    # load results
    with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
        df_dict = pickle.load(handle)
    return df_dict


class CustomEvolution(PymooEvolution):
    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = NSGA2(
            pop_size=self.optimizer.doe_size,
            sampling=self.optimizer.generator._pymoo_generator(),
            **self.optimizer.ea_kwargs
        )

    def evolve(self):
        logger.info("EXECUTE CUSTOM EVOLVE")
        res = minimize(problem=self.optimizer,
                       algorithm=self.ea,
                       termination=get_termination("n_gen", self.optimizer.max_generations),
                       seed=self.optimizer.seed,
                       verbose=True)

        self.optimizer.final_observe(res.F)

        # output results
        best_QoI, best_cand = res.F, res.X
        np.set_printoptions(linewidth=np.nan)
        logger.info(f"optimal QoIs:\n{best_QoI}")
        logger.info(f"optimal candidates:\n{best_cand}")
