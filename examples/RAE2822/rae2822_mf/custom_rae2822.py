import dill as pickle
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess

from matplotlib.ticker import MaxNLocator
from pymoo.core.problem import Problem

from aero_optim.mf_sm.mf_infill import maximize_ED, maximize_EI, minimize_LCB
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
        cl_over_cd = self.model.evaluate(np.array(candidates))
        self.df_dict[gid] = {
            cid: pd.DataFrame({"ResTot": 1., "CL/CD": cl_over_cd[cid]})
            for cid in range(len(candidates))
        }


class CustomOptimizer(PymooWolfOptimizer):
    def __init__(self, config: dict):
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=1, xl=self.bound[0], xu=self.bound[1]
        )

    def set_inner(self):
        """
        **Sets** some inner attributes:

        - bsl_CL_over_CD (float): baseline high fidelity Cl / Cd.
        - infill_freq (int): adaptive infill frequency.
        - infill_nb (int): adaptive infill number.
        - infill_lf_size (int): number of new low fidelity candidates for each adaptive infill.
        - infill_hf_size (int): number of new high fidelity candidates for each adaptive infill.
        - infill_nb_gen (int): number of generations for each infill adaptation algorithm.
        - bayesian_infill (bool): infill strategy: bayesian (True) or not (False).
        - lf_config (str): path to the hf simulation config template.
        - hf_config (str): path to the hf simulation config template.
        - infill_ctr (int): infill counter.
        """
        self.bsl_CL_over_CD = self.config["optim"]["baseline_CL_over_CD"]
        self.infill_freq = self.config["optim"]["infill_freq"]
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
        **Computes** the objective function for each candidate in the generation.

        Note:
            for this use-case there are no constraints
        """
        gid = self.gen_ctr
        self.execute_candidates(X, gid)

        # update candidates fitness
        self.J.extend([
            self.simulator.df_dict[gid][cid][self.QoI].iloc[-1] for cid in range(len(X))
        ])

        out["F"] = np.array(self.J[-self.doe_size:])
        self._observe(out["F"])
        self.gen_ctr += 1

    def execute_candidates(self, candidates, gid):
        logger.info(f"execute candidates of generation {gid}..")
        if gid > 0 and (gid + 1) % self.infill_freq == 0 and self.infill_ctr < self.infill_nb:
            logger.info(f"infill computation ({self.infill_ctr + 1})")
            infill_lf = self.compute_lf_infill()
            logger.info(f"lf infill candidates of generation {gid}:\n {infill_lf}")
            y_lf = self.execute_infill(gid, self.lf_config, infill_lf, "lf")
            logger.info(f"lf infill fitnesses of generation {gid}:\n {y_lf}")
            infill_hf = infill_lf[np.argmin(y_lf)]
            logger.info(f"hf infill candidates of generation {gid}:\n {infill_hf}")
            y_hf = self.execute_infill(gid, self.hf_config, infill_hf, "hf")
            logger.info(f"hf infill fitnesses of generation {gid}:\n {y_hf}")
            self.simulator.model.set_DOE(x_lf=infill_lf, y_lf=y_lf, x_hf=infill_hf, y_hf=y_hf)
            logger.info(f"model infill_hf prediction before update: "
                        f"{self.simulator.model.evaluate(np.atleast_2d(infill_hf))}")
            self.simulator.model.train()
            logger.info(f"model infill_hf prediction after update: "
                        f"{self.simulator.model.evaluate(np.atleast_2d(infill_hf))}")
            self.infill_ctr += 1

        logger.info(f"evaluating candidates of generation {gid}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            _, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)

        self.simulator.execute_sim(candidates, gid)

    def compute_lf_infill(self) -> np.ndarray:
        """
        **Computes** the low fidelity infill candidates.
        """
        if self.bayesian_infill:
            # Expected Improvement
            infill_lf = maximize_EI(
                model=self.simulator.model,
                n_var=self.n_design,
                bound=self.bound,
                seed=self.seed,
                n_gen=self.infill_nb_gen
            )
            # Lower Confidence Bound
            infill_lf_LCB = minimize_LCB(
                model=self.simulator.model,
                n_var=self.n_design,
                bound=self.bound,
                seed=self.seed,
                n_gen=self.infill_nb_gen
            )
            infill_lf = np.vstack((infill_lf, infill_lf_LCB))
            # max-min Euclidean Distance
            current_DOE = self.simulator.model.get_DOE()
            current_DOE = np.vstack((current_DOE, infill_lf))
            for _ in range(self.infill_lf_size - 2):
                infill_lf_ED = maximize_ED(
                    model=self.simulator.model,
                    DOE=current_DOE,
                    n_var=self.n_design,
                    bound=self.bound,
                    seed=self.seed,
                    n_gen=self.infill_nb_gen
                )
                infill_lf = np.vstack((infill_lf, infill_lf_ED))
                current_DOE = np.vstack((current_DOE, infill_lf_ED))
        else:
            # max-min Euclidean Distance
            current_DOE = self.simulator.model.get_DOE()
            for _ in range(self.infill_lf_size):
                infill_lf_ED = maximize_ED(
                    model=self.simulator.model,
                    DOE=current_DOE,
                    n_var=self.n_design,
                    bound=self.bound,
                    seed=self.seed,
                    n_gen=self.infill_nb_gen
                )
                infill_lf = np.vstack((infill_lf, infill_lf_ED))
                current_DOE = np.vstack((current_DOE, infill_lf_ED))
        return infill_lf

    def execute_infill(
            self, gid: int, config: str, X: np.ndarray, fidelity: str
    ) -> np.ndarray:
        """
        **Executes** infill candidates and returns their associated fitnesses.
        """
        name = f"{fidelity}_infill_{gid}"
        df_dict = execute_single_gen(
            outdir=os.path.join(self.outdir, name),
            config=config,
            X=X,
            name=name,
            n_design=self.config["ffd"].get("ffd_ncontrol", self.n_design)
        )
        Cl = np.array([df_dict[0][cid]["CL"].iloc[-1] for cid in range(len(df_dict[0]))])
        Cd = np.array([df_dict[0][cid]["CD"].iloc[-1] for cid in range(len(df_dict[0]))])
        assert len(Cl) == len(np.atleast_2d(X))
        return -Cl / Cd

    def _observe(self, pop_fitness: np.ndarray):
        """
        **Plots** some results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the candidates fitnesses,</br>
        > the baseline and deformed profiles.
        """
        gid = self.gen_ctr

        # compute population statistics
        self.compute_statistics(pop_fitness)
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

        # plot settings
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]
        res_dict = self.simulator.df_dict[gid]
        df_key = res_dict[0].columns  # ResTot, CL/CD
        cmap = mpl.colormaps[self.cmap].resampled(self.n_plt)
        colors = cmap(np.linspace(0, 1, self.n_plt))
        # subplot construction
        fig = plt.figure(figsize=(16, 5))
        ax1 = plt.subplot2grid((1, 4), (0, 0), colspan=3)  # profiles
        ax2 = plt.subplot2grid((1, 4), (0, 3), colspan=1)  # fitness (CL / CD)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        # loop over candidates through the last generated profiles
        for col_id, cid in enumerate(np.argsort(pop_fitness, kind="stable")[:self.n_plt]):
            ax1.plot(
                profiles[cid][:, 0], profiles[cid][:, 1], color=colors[col_id], label=f"c{cid}"
            )
            ax2.scatter(
                cid, res_dict[cid][df_key[1]], color=colors[col_id], label=f"c{cid}"
            )
            xmin, xmax = ax2.get_xlim()
        ax2.hlines(
            y=-self.bsl_CL_over_CD,
            xmin=xmin, xmax=xmax, linestyle="dashed", color="k", label="baseline"
        )
        # legend and title
        fig.suptitle(
            f"Generation {gid} results", size="x-large", weight="bold", y=1
        )
        # top left
        ax1.set_title("FFD profiles", weight="bold")
        ax1.set_xlabel('$x$ $[m]$')
        ax1.set_ylabel('$y$ $[m]$')
        # top right
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_title(f"{df_key[1]}", weight="bold")
        ax2.set_xlabel('candidate $[\\cdot]$')
        ax2.set_ylabel('$-C_l/C_d$ $[\\cdot]$')
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # save figure as png
        fig_name = f"pymoo_g{gid}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.figdir, fig_name), bbox_inches='tight')
        plt.close()

    def final_observe(self, *args, **kwargs):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations
        """
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name, baseline_value=-self.bsl_CL_over_CD)


def execute_single_gen(
        outdir: str, config: str, X: np.ndarray, name: str, n_design: int = 0
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
        "@output": outdir,
        "@n_design": f"{n_design if n_design else np.atleast_2d(X).shape[1]}",
        "@doe_size": f"{np.atleast_2d(X).shape[0]}",
        "@custom_doe": f"{custom_doe}"
    }
    replace_in_file(config_path, config_args)
    print(f"{name} computation..")
    # execute single generation
    exec_cmd = ["optim", "-c", f"{config_path}", "-v", "3", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"{name} computation finished")
    # load results
    with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
        df_dict = pickle.load(handle)
    return df_dict
