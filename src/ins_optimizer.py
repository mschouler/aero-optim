import logging
import numpy as np
import os

from abc import ABC, abstractmethod
from inspyred.ec import Individual
from random import Random
import signal
import time

from .ffd import FFD_2D
from .ins_generator import Generator
from .naca_base_mesh import NACABaseMesh
from .naca_block_mesh import NACABlockMesh
from .simulator import WolfSimulator
from .utils import check_dir

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    """
    This class implements a basic Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the Optimizer object.

        Input
            >> config: the config file dictionary.

        Inner
            >> n_design: the number of design variables (dimensions of the problem).
            >> doe_size: the size of the initial and subsequent generations.
            >> max_generations: the number of generations before termination.
            >> dat_file: path to input_geometry.dat (baseline geometry).
            >> outdir: highest level optimization output directory.

            Note: the result folder tree is structured as follows:
            outdir
            |__ FFD (contains <geom>_gXX_cYY.dat)
            |__ MESH (contains <geom>_gXX_cYY.mesh, .log, .geo_unrolled)
            |__ WOLF
                |__ wolf_gXX_cYY (contains the results of each simulation)

            >> study_type: use-case/meshing routine.
            >> budget: maximum number of concurrent proc in use.
            >> nproc_per_sim: number of proc per simulation.
            >> bound: design variables boundaries.
            >> sampler_name: name of the sampling algorithm used to generate the initial generation.
            >> seed: seed number of the random processes involved in the optimization.
            >> prng: pseudo-random generator passed to inspyred generator.
            >> gen_ctr: generation counter.
            >> generator: Generator object for the initial generation sampling.
            >> ffd: FFD_2D object to generate deformed geometries.
            >> gmsh_mesh: Mesh object to generate deformed geometries meshes.
        """
        self.config = config
        self.process_config()
        # required entries
        self.n_design: int = config["optim"]["n_design"]
        self.doe_size: int = config["optim"]["doe_size"]
        self.max_generations: int = config["optim"]["max_generations"]
        self.dat_file: str = config["study"]["file"]
        self.outdir: str = config["study"]["outdir"]
        self.study_type: str = config["study"]["study_type"]
        # optional entries
        self.budget: int = config["optim"].get("budget", 4)
        self.nproc_per_sim: int = config["optim"].get("nproc_per_sim", 1)
        self.bound: tuple[float, float] = config["optim"].get("bound", (-5, 5))
        self.sampler_name: str = config["optim"].get("sampler_name", "halton")
        # reproducibility variables
        self.seed: int = config["optim"].get("seed", 123)
        self.prng: Random = Random()
        self.prng.seed(self.seed)
        # generation counter
        self.gen_ctr: int = 0
        # optimization objects
        self.generator: Generator = Generator(
            self.seed, self.n_design, self.doe_size, self.sampler_name, self.bound
        )
        self.ffd: FFD_2D = FFD_2D(self.dat_file, self.n_design // 2)
        if self.study_type == "base":
            self.gmsh_mesh = NACABaseMesh
        elif self.study_type == "block":
            self.gmsh_mesh = NACABlockMesh

    def process_config(self):
        """
        Makes sure the config file contains the required information.
        """
        logger.info("process config..")
        if "n_design" not in self.config["optim"]:
            raise Exception(f"ERROR -- no <n_design> entry in {self.config['optim']}")
        if "doe_size" not in self.config["optim"]:
            raise Exception(f"ERROR -- no <doe_size> entry in {self.config['optim']}")
        if "max_generations" not in self.config["optim"]:
            raise Exception(f"ERROR -- no <max_generations> entry in {self.config['optim']}")
        if "file" not in self.config["study"]:
            raise Exception(f"ERROR -- no <file> entry in {self.config['study']}")
        if "budget" not in self.config["optim"]:
            logger.warning(f"no <budget> entry in {self.config['optim']}")
        if "nproc_per_sim" not in self.config["optim"]:
            logger.warning(f"no <nproc_per_sim> entry in {self.config['optim']}")
        if "bound" not in self.config["optim"]:
            logger.warning(f"no <bound> entry in {self.config['optim']}")
        if "sampler_name" not in self.config["optim"]:
            logger.warning(f"no <sampler_name> entry in {self.config['optim']}")
        if "seed" not in self.config["optim"]:
            logger.warning(f"no <seed> entry in {self.config['optim']}")
        #  alter config for optimization purposes
        if "outfile" in self.config["study"]:
            logger.warning(f"<outfile> entry in {self.config['study']} will be ignored")
            del self.config["study"]["outfile"]
        if "GUI" in self.config["gmsh"]["view"]:
            logger.warning(
                f"<GUI> entry in {self.config['gmsh']['view']} forced to False"
            )
            self.config["gmsh"]["view"]["GUI"] = False

    def deform(self, Delta: np.ndarray, gid: int, cid: int) -> str:
        """
        Performs FFD on a given candidate and returns its resulting file.
        """
        ffd_dir = os.path.join(self.outdir, "FFD")
        check_dir(ffd_dir)
        logger.info(f"generate profile with deformation {Delta}")
        profile: np.ndarray = self.ffd.apply_ffd(Delta)
        return self.ffd.write_ffd(profile, Delta, ffd_dir, gid=gid, cid=cid)

    def mesh(self, ffdfile: str) -> str:
        """
        Builds mesh for a given candidate and returns its resulting file.
        """
        mesh_dir = os.path.join(self.outdir, "MESH")
        check_dir(mesh_dir)
        gmsh_mesh = self.gmsh_mesh(self.config, ffdfile)
        gmsh_mesh.build_mesh()
        return gmsh_mesh.write_mesh(mesh_dir)

    @abstractmethod
    def evaluate(self, candidates: Individual, args: dict) -> list[float]:
        """
        Computes all candidates outputs and return the optimizer list of QoIs.
        """


class WolfOptimizer(Optimizer):
    """
    This class implements a Wolf based Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfOptimizer object.

        Inner
            >> simulator: WolfSimulator object to perform Wolf simulations.
        """
        super().__init__(config)
        self.simulator: WolfSimulator = WolfSimulator(self.config)

    def evaluate(self, candidates: Individual, args: dict) -> list[float]:
        """
        Executes Wolf simulations, extracts results and returns the list of candidates QoIs.
        """
        logger.info("Enters evaluate")
        J: list[float] = []
        QoI: str = args.get("QoI", "CD")
        gid = self.gen_ctr
        # execute all candidates
        for cid, cand in enumerate(candidates):
            ffd_file = self.deform(cand, gid, cid)
            # meshing with proper sigint management
            # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/842
            ORIGINAL_SIGINT_HANDLER = signal.signal(signal.SIGINT, signal.SIG_DFL)
            mesh_file = self.mesh(ffd_file)
            signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
            while self.simulator.monitor_sim_progress() * self.nproc_per_sim >= self.budget:
                time.sleep(1)
            self.simulator.execute_sim(meshfile=mesh_file, gid=gid, cid=cid)
        # wait for last candidates
        while self.simulator.monitor_sim_progress() > 0:
            time.sleep(1)
        J.extend([df[QoI].iloc[-1] for df in self.simulator.df_list[-self.doe_size:]])
        self.gen_ctr += 1
        return J
