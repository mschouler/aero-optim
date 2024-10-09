import dill as pickle
import logging
import numpy as np
import os
import pandas as pd
import subprocess

from pymoo.core.problem import Problem

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
        """
        self.bsl_CL_over_CD = self.config["optim"]["baseline_CL_over_CD"]

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
        logger.info(f"evaluating candidates of generation {gid}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            _, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)

        self.simulator.execute_sim(candidates, gid)

    def _observe(self, pop_fitness: np.ndarray):
        """
        **Computes** population fitness statistics.
        """
        gid = self.gen_ctr

        # compute population statistics
        self.compute_statistics(pop_fitness)
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

    def final_observe(self, *args, **kwargs):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations
        """
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name, baseline_value=-self.bsl_CL_over_CD)


def execute_infill(X: np.ndarray, config: str, n_design: int, outdir: str, ite: int, fidelity: str):
    # n_design is the number of design points in the FFD sense
    name = f"{fidelity}_infill_{ite}"
    df_dict = execute_single_gen(
        X=X,
        config=config,
        outdir=os.path.join(outdir, name),
        n_design=n_design,
        name=name
    )
    Cl = np.array([df_dict[0][cid]["CL"].iloc[-1] for cid in range(len(df_dict[0]))])
    Cd = np.array([df_dict[0][cid]["CD"].iloc[-1] for cid in range(len(df_dict[0]))])
    assert len(Cl) == len(np.atleast_2d(X))
    return -Cl / Cd


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
