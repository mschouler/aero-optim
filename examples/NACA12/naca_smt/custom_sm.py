import dill as pickle
import logging
import numpy as np
import pandas as pd

from src.optim.pymoo_optimizer import PymooWolfOptimizer
from src.simulator.simulator import Simulator
from src.utils import check_file

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
        self.solver_name = "smt_model"

    def set_model(self, model_file: str):
        check_file(model_file)
        with open(model_file, "rb") as handle:
            self.model = pickle.load(handle)

    def execute_sim(self, candidates: list[float] | np.ndarray, gid: int = 0):
        logger.info(f"execute simulations g{gid} with {self.solver_name}")
        cd, cl = self.model.predict(np.array(candidates))
        self.df_dict[gid] = {
            cid: pd.DataFrame({"ResTot": 1., "CD": cd[cid], "CL": cl[cid]})
            for cid in range(len(candidates))
        }


class CustomOptimizer(PymooWolfOptimizer):
    def set_gmsh_mesh_class(self):
        self.MeshClass = None

    def execute_candidates(self, candidates, gid):
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])

        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            _, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)

        self.simulator.execute_sim(candidates, gid)
