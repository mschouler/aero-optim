import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from abc import ABC, abstractmethod
from inspyred.ec import Individual
from random import Random
from typing import Any
import signal
import time

from .ffd import FFD_2D
from .ins_generator import Generator
from .naca_base_mesh import NACABaseMesh
from .naca_block_mesh import NACABlockMesh
from .simulator import WolfSimulator
from .utils import check_dir

plt.set_loglevel(level='warning')
logger = logging.getLogger(__name__)


def shoe_lace(xy: np.ndarray) -> float:
    """
    Returns the geometry area computed with the shoelace formula.
    see https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python
    """
    return 0.5 * np.abs(
        np.sum([xy[i - 1, 0] * xy[i, 1] - xy[i, 0] * xy[i - 1, 1] for i in range(len(xy))])
    )


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
        self.bound: tuple[Any, ...] = tuple(config["optim"].get("bound", [-1., 1.]))
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
        if "GUI" in self.config["gmsh"]["view"]:
            logger.warning(
                f"<GUI> entry in {self.config['gmsh']['view']} forced to False"
            )
            self.config["gmsh"]["view"]["GUI"] = False

    def deform(self, Delta: np.ndarray, gid: int, cid: int) -> tuple[str, np.ndarray]:
        """
        Performs FFD on a given candidate and returns its resulting file.
        """
        ffd_dir = os.path.join(self.outdir, "FFD")
        check_dir(ffd_dir)
        logger.info(f"generate profile with deformation {Delta}")
        profile: np.ndarray = self.ffd.apply_ffd(Delta)
        return self.ffd.write_ffd(profile, Delta, ffd_dir, gid=gid, cid=cid), profile

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
            >> ffd_profiles: all deformed geometries {gid: {cid: ffd_profile}}.
            >> QoI: the quantity of intereset to minimize/maximize.
            >> n_plt: the number of best candidates results to display after each evaluation.
            >> baseline_CD: the drag coefficient of the baseline geometry.
            >> baseline_CL: the lift coefficient of the baseline geometry.
            >> baseline_area: the baseline area that is used as a structural constraint.
            >> penalty_arg: a (key, value) constraint not to be worsen by the optimization.
            >> cmap: the colormaps used for the observer plot
               (see https://matplotlib.org/stable/users/explain/colors/colormaps.html).
        """
        super().__init__(config)
        self.simulator: WolfSimulator = WolfSimulator(self.config)
        self.J: list[float] = []
        self.ffd_profiles: list[list[np.ndarray]] = []
        self.QoI: str = config["optim"].get("QoI", "CD")
        self.n_plt: int = config["optim"].get("n_plt", 5)
        self.baseline_CD: float = config["optim"].get("baseline_CD", 0.150484)
        self.baseline_CL: float = config["optim"].get("baseline_CL", 0.36236)
        self.baseline_area: float = shoe_lace(self.ffd.pts)
        self.penalty: list = config["optim"].get("penalty", ["CL", self.baseline_CL])
        self.cmap: str = config["optim"].get("cmap", "viridis")

    def constraint(self, gid: int, cid: int, ffd_profile: np.ndarray, pen_value: float) -> float:
        """
        Returns a penalty value based on some specific constraints.
        see https://inspyred.readthedocs.io/en/latest/recipes.html#constraint-selection
        """
        area_cond: bool = (shoe_lace(ffd_profile) > 1.4 * self.baseline_area
                           or shoe_lace(ffd_profile) < 0.6 * self.baseline_area)
        penalty_cond: bool = pen_value < self.penalty[-1]
        if area_cond or penalty_cond:
            logger.info(f"penalized candidate g{gid} c{cid} "
                        f"with area {shoe_lace(ffd_profile)} and CL {pen_value}")
            return 1.
        return 0.

    def observe(
            self,
            population: Individual,
            num_generations: int,
            num_evaluations: int,
            args: dict
    ):
        """
        Displays the n_plt best results each time a generation has been evaluated:
            - the simulations residuals,
            - the simulations CD & CL,
            - the candidates fitness,
            - the baseline and deformed profiles.
        """
        fitness: np.ndarray = np.array(self.J[-self.doe_size:])
        sorted_idx = np.argsort(fitness)[:self.n_plt]
        baseline: np.ndarray = self.ffd.pts
        logger.info(f"extracting {self.n_plt} best profiles in g{num_generations}: {sorted_idx}..")
        profiles: list[np.ndarray] = self.ffd_profiles[num_generations][-self.doe_size:]
        # plot settings
        cmap = mpl.colormaps[self.cmap].resampled(self.n_plt)
        colors = cmap(np.linspace(0, 1, self.n_plt))
        # subplot construction
        plt.figure(figsize=(16, 12))
        ax1 = plt.subplot(2, 1, 1)  # profiles
        ax2 = plt.subplot(2, 3, 4)  # ResTot
        ax3 = plt.subplot(2, 3, 5)  # CD & CL
        ax4 = plt.subplot(2, 3, 6)  # fitness (CD)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        ax3.axhline(y=self.baseline_CD, color='k', label="baseline")
        ax3.axhline(y=self.baseline_CL, color='k', linestyle="--", label="baseline")
        ax4.axhline(y=self.baseline_CD, color='k', linestyle="--", label="baseline")
        # loop over candidates through the last generated profiles
        for color, cid in enumerate(sorted_idx):
            ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[color], label=f"c{cid}")
            res_list = self.simulator.df_list[-self.doe_size:]
            df_key = res_list[0].columns  # "ResTot", "CD", "CL", "ResCD", "ResCL", "x", "y", "Cp"
            res_list[cid][df_key[0]].plot(ax=ax2, color=colors[color], label=f"c{cid}")  # ResTot
            res_list[cid][df_key[1]].plot(ax=ax3, color=colors[color], label=f"{df_key[1]} c{cid}")
            res_list[cid][df_key[2]].plot(
                ax=ax3, color=colors[color], ls="--", label=f"{df_key[2]} c{cid}"
            )
            ax4.scatter(cid, fitness[cid], color=colors[color], label=f"c{cid} fitness")  # fitness
        # legend and title
        # top
        ax1.set_title("FFD profiles")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom left
        ax2.set_title(f"{df_key[0]}")
        ax2.set_yscale("log")
        ax2.set_xlabel('it. #')
        ax2.set_ylabel('residual')
        # bottom center
        ax3.set_title(f"{df_key[1]} & {df_key[2]}")
        ax3.set_xlabel('it. #')
        ax3.set_ylabel('aerodynamic coeff.')
        # bottom right
        ax4.set_title(f"fitness: penalized {self.QoI}")
        ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax4.set_xlabel('candidate #')
        ax4.set_ylabel("fitness")
        # save figure as png
        fig_name = f"res_g{num_generations}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name))

    def evaluate(self, candidates: Individual, args: dict) -> list[float]:
        """
        Executes Wolf simulations, extracts results and returns the list of candidates QoIs.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        gid = self.gen_ctr
        self.ffd_profiles.append([])
        # execute all candidates
        for cid, cand in enumerate(candidates):
            ffd_file, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)
            # meshing with proper sigint management
            # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/842
            ORIGINAL_SIGINT_HANDLER = signal.signal(signal.SIGINT, signal.SIG_DFL)
            mesh_file = self.mesh(ffd_file)
            signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
            while self.simulator.monitor_sim_progress() * self.nproc_per_sim >= self.budget:
                time.sleep(1)
            self.simulator.execute_sim(meshfile=mesh_file, gid=gid, cid=cid)
        # wait for last candidates to finish
        while self.simulator.monitor_sim_progress() > 0:
            time.sleep(1)
        # add penalty to the candidates fitness
        for cid, _ in enumerate(candidates):
            self.J.append(
                self.constraint(
                    gid, cid,
                    self.ffd_profiles[gid][cid],
                    self.simulator.df_list[-self.doe_size + cid][self.penalty[0]].iloc[-1]
                )
            )
            self.J[-1] += self.simulator.df_list[-self.doe_size + cid][self.QoI].iloc[-1]
        self.gen_ctr += 1
        return self.J[-self.doe_size:]

    def final_observe(self):
        """
        Displays convergence progress by plotting the fitness values
        obtained with the successive generations.
        """
        logger.info(f"plotting {self.gen_ctr} generations results..")
        # plot settings
        cmap = mpl.colormaps[self.cmap].resampled(self.gen_ctr)
        colors = cmap(np.linspace(0, 1, self.gen_ctr))
        # subplot construction
        _, ax = plt.subplots(figsize=(8, 8))
        ax.axhline(y=self.baseline_CD, color='k', label="baseline")
        # loop over generations
        for gid in range(self.gen_ctr):
            for cid in range(self.doe_size):
                ax.scatter(cid, self.J[gid * self.doe_size + cid],
                           color=colors[gid], label=f"g{gid}")
        # legend and title
        # top
        ax.set_title("Convergence of the optimization")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('cid')
        ax.set_ylabel('penalized fitness')
        # save figure as png
        fig_name = f"optim_g{self.gen_ctr}_c{self.doe_size}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name))
