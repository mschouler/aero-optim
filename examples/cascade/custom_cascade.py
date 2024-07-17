import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import signal
import time

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)
from aero_optim.optim.evolution import PymooEvolution
from aero_optim.optim.optimizer import WolfOptimizer
from aero_optim.optim.pymoo_optimizer import PymooWolfOptimizer
from aero_optim.simulator.simulator import WolfSimulator


logger = logging.getLogger()


class CustomSimulator(WolfSimulator):
    def __init__(self, config: dict):
        super().__init__(config)

    def post_process(self, dict_id: dict, sim_out_dir: str) -> dict[str, pd.DataFrame]:
        """
        **Post-processes** the results of a terminated triple simulation.</br>
        **Returns** the extracted results in a dictionary of DataFrames.

        Note:
            there are two QoIs: loss_ADP and loss_OP = 1/2(loss_OP1 + loss_OP2)
            to be extracted from: sim_out_dir/ADP, sim_out_dir/OP1  and sim_out_dir/OP2
        """
        df_sub_dict: dict[str, pd.DataFrame] = {}
        for fname in ["ADP", "OP1", "OP2"]:
            logger.info(f"post_process g{dict_id['gid']}, c{dict_id['cid']} {fname}..")
            df_sub_dict[fname] = super().post_process(dict_id, os.path.join(sim_out_dir, fname))
        return df_sub_dict


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
        **Sets** some baseline quantities required to compute the relative constraints:

        - bsl_camber_th (tuple[np.ndarray, float, float, np.ndarray])
        - bsl_area (float)
        - bsl_c (float)
        - bsl_c_ax (float)
        - bsl_cog (np.ndarray)
        - bsl_cog_x (float)
        """
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

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Computes** the objective function and constraints for each candidate in the generation.

        Note:
            For this use-case, constraints can be computed before simulations.
            Unfeasible candidates are not simulated.
        """
        gid = self.gen_ctr
        self.feasible_cid[gid] = []

        # compute candidates constraints and execute feasible candidates only
        out["G"] = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        for cid in range(len(X)):
            if np.where(out["G"][cid] > 0.)[0].size == 0:
                self.feasible_cid[gid].append(cid)
                loss_ADP = self.simulator.df_dict[gid][cid]["ADP"][self.QoI].iloc[-1]
                loss_OP1 = self.simulator.df_dict[gid][cid]["OP1"][self.QoI].iloc[-1]
                loss_OP2 = self.simulator.df_dict[gid][cid]["OP2"][self.QoI].iloc[-1]
                logger.info(f"g{gid}, c{cid}: "
                            f"w_ADP = {loss_ADP}, w_OP = {0.5 * (loss_OP1 + loss_OP2)}")
                self.J.append([loss_ADP, 0.5 * (loss_OP1 + loss_OP2)])
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
            logger.info(f"candidate g{gid}, c{cid} constraint computation")
            constraint.append(self.apply_candidate_constraints(ffd_profile, gid, cid))
            # only mesh and execute feasible candidates
            if len([v for v in constraint[cid] if v > 0.]) == 0:
                self.feasible_cid[gid].append(cid)
                # meshing with proper sigint management
                # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/842
                ORIGINAL_SIGINT_HANDLER = signal.signal(signal.SIGINT, signal.SIG_DFL)
                mesh_file = self.mesh(ffd_file)
                signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
                while self.simulator.monitor_sim_progress() * self.nproc_per_sim >= self.budget:
                    time.sleep(1)
                self.simulator.execute_sim(meshfile=mesh_file, gid=gid, cid=cid)
            else:
                logger.info(f"unfeasible candidate g{gid}, c{cid} not simulated")

        # wait for last candidates to finish
        while self.simulator.monitor_sim_progress() > 0:
            time.sleep(0.1)
        return np.row_stack(constraint)

    def apply_candidate_constraints(self, profile: np.ndarray, gid: int, cid: int) -> list[float]:
        """
        **Computes** various relative and absolute constraints of a given candidate
        and **returns** their values as a list of floats.

        Note:
            When some constraint is violated, a graph is also generated.
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
            fig_name = os.path.join(self.outdir, f"profile_g{gid}_c{cid}.png")
            plot_profile(profile, cog, fig_name)
        if th_cond > 0 or Xth_cond > 0 or area_cond > 0:
            fig_name = os.path.join(self.outdir, f"sides_g{gid}_c{cid}.png")
            plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
        return [th_cond, Xth_cond, area_cond, cog_cond]

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
        res_dict = self.simulator.df_dict[gid]
        df_key = res_dict[0]["ADP"].columns  # ResTot  LossCoef   x   y  Cp  Mis
        cmap = mpl.colormaps[self.cmap].resampled(self.doe_size)
        colors = cmap(np.linspace(0, 1, self.doe_size))
        # subplot construction
        fig = plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(2, 1, 1)  # profiles
        ax2 = plt.subplot(2, 2, 3)  # ResTot
        ax3 = plt.subplot(2, 2, 4)  # fitness (loss_ADP vs loss_OP)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        # loop over candidates through the last generated profiles
        for cid in self.feasible_cid[gid]:
            # only plot half profiles and residuals
            ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[cid], label=f"c{cid}")
            res_dict[cid]["ADP"][df_key[0]].plot(ax=ax2, color=colors[cid], label=f"c{cid}")
            # plot all fitnesses
            ax3.scatter(pop_fitness[cid, 0], pop_fitness[cid, 1],
                        color=colors[cid], label=f"c{cid}")
        # legend and title
        fig.suptitle(
            f"Generation {gid} results", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom left
        ax2.set_title(f"ADP {df_key[0]}", weight="bold")
        ax2.set_yscale("log")
        ax2.set_xlabel('it. #')
        ax2.set_ylabel('residual')
        # bottom right
        ax3.set_title(f"{self.QoI} ADP vs {self.QoI} OP", weight="bold")
        ax3.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax3.set_xlabel('$w_\\text{ADP}$')
        ax3.set_ylabel('$w_\\text{OP}$')
        # save figure as png
        fig_name = f"pymoo_g{gid}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()

    def final_observe(self):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        gen_fitness = np.row_stack(self.J)

        # plotting data
        cmap = mpl.colormaps[self.cmap].resampled(self.max_generations)
        colors = cmap(np.linspace(0, 1, self.max_generations))
        for gid in range(self.max_generations):
            ax.scatter(gen_fitness[gid * self.doe_size: (gid + 1) * self.doe_size][:, 0],
                       gen_fitness[gid * self.doe_size: (gid + 1) * self.doe_size][:, 1],
                       color=colors[gid], label=f"g{gid}")
        plt.grid(True)

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

        self.optimizer.final_observe()

        # output results
        best = res.F
        logger.info(f"optimal J_pymoo: {best}")
