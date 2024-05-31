import logging
import numpy as np
import os

from abc import ABC, abstractmethod
from random import Random
from typing import Any, Type

from src.ffd.ffd import FFD_2D
from src.optim.generator import Generator
from src.mesh.naca_base_mesh import NACABaseMesh
from src.mesh.naca_block_mesh import NACABlockMesh
from src.mesh.cascade_mesh import CascadeMesh
from src.utils import check_dir

logger = logging.getLogger(__name__)


def shoe_lace(xy: np.ndarray) -> float:
    """
    **Returns** the geometry area computed with the shoelace formula.</br>
    see https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python
    """
    return 0.5 * np.abs(
        np.sum([xy[i - 1, 0] * xy[i, 1] - xy[i, 0] * xy[i - 1, 1] for i in range(len(xy))])
    )


class Optimizer(ABC):
    """
    This class implements an abstract optimizer.
    """
    def __init__(self, config: dict, debug: bool = False):
        """
        Instantiates the Optimizer object.

        **Input**

        - config (dict): the config file dictionary.
        - debug (bool): skip FFD and Mesh objects instantation for debugging purposes.

        **Inner**

        - n_design (int): the number of design variables (dimensions of the problem).
        - doe_size (int): the size of the initial and subsequent generations.
        - max_generations (int): the number of generations before termination.
        - dat_file (str): path to input_geometry.dat (baseline geometry).
        - outdir (str): highest level optimization output directory.

            Note: the result folder tree is structured as follows:
            ```
            outdir
            |__ FFD (contains <geom>_gXX_cYY.dat)
            |__ MESH (contains <geom>_gXX_cYY.mesh, .log, .geo_unrolled)
            |__ SOLVER
                |__ solver_gXX_cYY (contains the results of each simulation)
            ```

        - study_type (str): use-case/meshing routine.
        - strategy (str): the optimization algorithm amongst inspyred's [ES, GA, SA, PSO]</br>
            see https://pythonhosted.org/inspyred/examples.html#standard-algorithms

        - maximize (bool): whether to maximize or minimize the objective QoIs.
        - budget (int): maximum number of concurrent proc in use.
        - nproc_per_sim (int): number of proc per simulation.
        - bound tuple[float]: design variables boundaries.
        - sampler_name (str): name of the sampling algorithm used to generate samples.
          the initial generation.
        - seed (int): seed number of the random processes involved in the optimization.
        - prng (random.Random): pseudo-random generator passed to inspyred generator.
        - ea_kwargs (dict): additional arguments to be passed to the evolution algorithm.
        - gen_ctr (int): generation counter.
        - generator (Generator): Generator object for the initial generation sampling.
        - ffd (FFD_2D): FFD_2D object to generate deformed geometries.
        - gmsh_mesh (Mesh): Mesh object to generate deformed geometries meshes.
        - mean (list[float]): list of populations mean fitness.
        - median (list[float]): list of populations median fitness.
        - max (list[float]): list of populations max fitness.
        - min (list[float]): list of populations min fitness.
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
        self.strategy: str = config["optim"].get("strategy", "ES")
        self.maximize: bool = config["optim"].get("maximize", False)
        self.budget: int = config["optim"].get("budget", 4)
        self.nproc_per_sim: int = config["optim"].get("nproc_per_sim", 1)
        self.bound: tuple[Any, ...] = tuple(config["optim"].get("bound", [-1, 1]))
        self.sampler_name: str = config["optim"].get("sampler_name", "lhs")
        self.ea_kwargs: dict = config["optim"].get("ea_kwargs", {})
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
        if not debug:
            self.ffd: FFD_2D = FFD_2D(self.dat_file, self.n_design // 2)
            self.gmsh_mesh: Type[NACABaseMesh] | Type[NACABlockMesh] | Type[CascadeMesh]
            if self.study_type == "base":
                self.gmsh_mesh = NACABaseMesh
            elif self.study_type == "block":
                self.gmsh_mesh = NACABlockMesh
            elif self.study_type == "cascade":
                self.gmsh_mesh = CascadeMesh
        # population statistics
        self.mean: list[float] = []
        self.median: list[float] = []
        self.max: list[float] = []
        self.min: list[float] = []

    def process_config(self):
        """
        **Makes sure** the config file contains the required information.
        """
        logger.info("processing config..")
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
        if "view" in self.config["gmsh"] and "GUI" in self.config["gmsh"]["view"]:
            logger.warning(
                f"<GUI> entry in {self.config['gmsh']['view']} forced to False"
            )
            self.config["gmsh"]["view"]["GUI"] = False

    def deform(self, Delta: np.ndarray, gid: int, cid: int) -> tuple[str, np.ndarray]:
        """
        **Applies** FFD on a given candidate and returns its resulting file.
        """
        ffd_dir = os.path.join(self.outdir, "FFD")
        check_dir(ffd_dir)
        logger.info(f"g{gid}, c{cid} generate profile with deformation {Delta}")
        profile: np.ndarray = self.ffd.apply_ffd(Delta)
        return self.ffd.write_ffd(profile, Delta, ffd_dir, gid=gid, cid=cid), profile

    def mesh(self, ffdfile: str) -> str:
        """
        **Builds** mesh for a given candidate and returns its resulting file.
        """
        mesh_dir = os.path.join(self.outdir, "MESH")
        check_dir(mesh_dir)
        gmsh_mesh = self.gmsh_mesh(self.config, ffdfile)
        gmsh_mesh.build_mesh()
        return gmsh_mesh.write_mesh(mesh_dir)

    @abstractmethod
    def _evaluate(self, *args, **kwargs) -> list[float] | None:
        """
        Computes all candidates outputs and return the optimizer list of QoIs.
        """


class ABCCustomEvolution(ABC):
    """
    This class implements an abstract custom evolution.
    """
    @abstractmethod
    def set_ea(self, *args, **kwargs):
        """
        Sets the evolutionary computation algorithm.
        """
        return

    @abstractmethod
    def custom_evolve(self, *args, **kwargs):
        """
        Defines how to execute the optimization.
        """
        return
