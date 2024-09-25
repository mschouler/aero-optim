import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import signal
import time

from abc import ABC, abstractmethod
from inspyred.ec import Individual
from matplotlib.ticker import MaxNLocator
from random import Random
from typing import Any

from aero_optim.geom import get_area
from aero_optim.ffd.ffd import FFD_2D
from aero_optim.mesh.naca_base_mesh import NACABaseMesh
from aero_optim.mesh.naca_block_mesh import NACABlockMesh
from aero_optim.mesh.cascade_mesh import CascadeMesh
from aero_optim.optim.generator import Generator
from aero_optim.simulator.simulator import DebugSimulator, WolfSimulator
from aero_optim.utils import check_dir, get_custom_class, STUDY_TYPE

# set pillow and matplotlib loggers to WARNING mode
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.set_loglevel(level='warning')

# get framework logger
logger = logging.getLogger(__name__)


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

        Note:
            the result folder tree is structured as follows:
            ```
            outdir
            |__ FFD (contains <geom>_gXX_cYY.dat)
            |__ Figs (contains the figures generated during the optimization)
            |__ MESH (contains <geom>_gXX_cYY.mesh, .log, .geo_unrolled)
            |__ SOLVER
                |__ solver_gXX_cYY (contains the results of each simulation)
            ```

        - study_type (str): use-case/meshing routine.
        - strategy (str): the optimization algorithm amongst inspyred's [ES, PSO]
            and pymoo's [GA, PSO]</br>
            see https://pythonhosted.org/inspyred/examples.html#standard-algorithms</br>
            and https://pymoo.org/algorithms/list.html#nb-algorithms-list

        - maximize (bool): whether to maximize or minimize the objective QoIs.
        - budget (int): maximum number of concurrent proc in use.
        - nproc_per_sim (int): number of proc per simulation.
        - bound (tuple[float]): design variables boundaries.
        - custom_doe (str): path to a custom doe.
        - sampler_name (str): name of the sampling algorithm used to generate samples.
          the initial generation.
        - seed (int): seed number of the random processes involved in the optimization.
        - prng (random.Random): pseudo-random generator passed to inspyred generator.
        - ea_kwargs (dict): additional arguments to be passed to the evolution algorithm.
        - gen_ctr (int): generation counter.
        - generator (Generator): Generator object for the initial generation sampling.
        - ffd (FFD_2D): FFD_2D object to generate deformed geometries.
        - gmsh_mesh (Mesh): Mesh class to generate deformed geometries meshes.
        - simulator (Simulator): Simulator object to perform simulations.
        - mean (list[float]): list of populations mean fitness.
        - median (list[float]): list of populations median fitness.
        - max (list[float]): list of populations max fitness.
        - min (list[float]): list of populations min fitness.
        - J (list[float | list[float]]): the list of all generated candidates fitnesses.
        - inputs (list[list[np.ndarray]]): all input candidates.
        - ffd_profiles (list[list[np.ndarray]]): all deformed geometries {gid: {cid: ffd_profile}}.
        - QoI (str): the quantity of intereset to minimize/maximize.
        - n_plt (int): the number of best candidates results to display after each evaluation.
        - cmap (str): the colormaps used for the observer plot</br>
            see https://matplotlib.org/stable/users/explain/colors/colormaps.html.
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
        self.custom_file: str = config["study"].get("custom_file", "")
        self.strategy: str = config["optim"].get("strategy", "PSO")
        self.maximize: bool = config["optim"].get("maximize", False)
        self.budget: int = config["optim"].get("budget", 4)
        self.nproc_per_sim: int = config["optim"].get("nproc_per_sim", 1)
        self.bound: tuple[Any, ...] = tuple(config["optim"].get("bound", [-1, 1]))
        self.custom_doe: str = config["optim"].get("custom_doe", "")
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
            self.seed, self.n_design, self.doe_size, self.sampler_name, self.bound, self.custom_doe
        )
        if not debug:
            self.ffd: FFD_2D = FFD_2D(self.dat_file, self.n_design // 2)
            self.set_gmsh_mesh_class()
        self.set_simulator_class()
        self.simulator = self.SimulatorClass(self.config)
        # population statistics
        self.mean: list[float] = []
        self.median: list[float] = []
        self.max: list[float] = []
        self.min: list[float] = []
        # set other inner optimization variables
        self.J: list[float | list[float]] = []
        self.inputs: list[list[np.ndarray]] = []
        self.ffd_profiles: list[list[np.ndarray]] = []
        self.QoI: str = self.config["optim"].get("QoI", "CD")
        self.n_plt: int = self.config["optim"].get("n_plt", 5)
        self.cmap: str = self.config["optim"].get("cmap", "viridis")
        self.set_inner()
        # figure directory
        self.figdir: str = os.path.join(self.outdir, "Figs")
        check_dir(self.figdir)

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

    def set_gmsh_mesh_class(self):
        """
        **Instantiates** the mesher class as custom if found,
        as one of the default meshers otherwise.
        """
        self.MeshClass = (
            get_custom_class(self.custom_file, "CustomMesh") if self.custom_file else None
        )
        if not self.MeshClass:
            if self.study_type == STUDY_TYPE[0]:
                self.MeshClass = NACABaseMesh
            elif self.study_type == STUDY_TYPE[1]:
                self.MeshClass = NACABlockMesh
            elif self.study_type == STUDY_TYPE[2]:
                self.MeshClass = CascadeMesh
            else:
                raise Exception(f"ERROR -- incorrect study_type <{self.study_type}>")

    def set_inner(self):
        """
        **Sets** some use-case specific inner variables:
        """
        logger.info("set_inner not implemented")

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

        Note:
            if a mesh file matching the pattern name already exists, it is not rebuilt.
        """
        mesh_dir = os.path.join(self.outdir, "MESH")
        check_dir(mesh_dir)
        gmsh_mesh = self.MeshClass(self.config, ffdfile)
        if os.path.isfile(gmsh_mesh.get_meshfile(mesh_dir)):
            return gmsh_mesh.get_meshfile(mesh_dir)
        gmsh_mesh.build_mesh()
        return gmsh_mesh.write_mesh(mesh_dir)

    def execute_candidates(self, candidates: list[Individual] | np.ndarray, gid: int):
        """
        **Executes** all candidates and **waits** for them to finish.

        Note:
            this method is meant to be called in _evaluate.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
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
            time.sleep(0.1)

    def compute_statistics(self, gen_fitness: np.ndarray):
        """
        **Computes** generation statistics.

        Note:
            this method is meant to be called in `_observe`.
        """
        self.mean.append(np.mean(gen_fitness))
        self.median.append(np.median(gen_fitness))
        self.min.append(min(gen_fitness))
        self.max.append(max(gen_fitness))

    def _observe(self, *args, **kwargs):
        """
        **Plots** generation data after each evaluation.
        """
        logger.info("_observe not implemented")

    def plot_generation(
            self,
            gid: int,
            sorted_idx: np.ndarray,
            gen_fitness: np.ndarray,
            fig_name: str
    ):
        """
        **Plots** the results of the last evaluated generation.
        **Saves** the graph in the output directory.

        Note:
            this method is meant to be called in `_observe`.
        """
        logger.info("plot_generation not implemented")

    def plot_progress(self, gen_nbr: int, fig_name: str, baseline_value: float | None = None):
        """
        **Plots** and **saves** the overall progress of the optimization.

        Note:
            this method is meant to be called in `final_observe`.
        """
        logger.info(f"plotting populations statistics after {gen_nbr} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        psize = self.doe_size
        if baseline_value:
            ax.axhline(y=baseline_value, color='k', ls="--", label="baseline")

        # plotting data
        best = self.max if self.maximize else self.min
        worst = self.min if self.maximize else self.max
        data = [self.mean, self.median, best, worst]
        colors = ["grey", "blue", "green", "red"]
        labels = ["mean", "median", "best", "worst"]
        for val, col, lab in zip(data, colors, labels):
            ax.plot(range(self.gen_ctr), val, color=col, label=lab)
        plt.fill_between(range(self.gen_ctr), data[2], data[3], color='#e6f2e6')
        plt.grid(True)
        ymin = min([min(d) for d in data])
        ymax = max([max(d) for d in data])
        yrange = ymax - ymin
        plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # legend and title
        ax.set_title(f"Optimization evolution ({gen_nbr} g. x {psize} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('generation $[\\cdot]$')
        ax.set_ylabel('fitness')

        # save figure as png
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()

    def save_results(self):
        """
        **Saves** candidates and fitnesses to file.
        """
        logger.info(f"optimization results saved to {self.outdir}")
        np.savetxt(
            os.path.join(self.outdir, "candidates.txt"),
            np.reshape(self.inputs, (-1, self.n_design))
        )
        np.savetxt(os.path.join(self.outdir, "fitnesses.txt"), self.J)

    @abstractmethod
    def set_simulator_class(self):
        """
        Instantiates the simulator class with CustomSimulator if found.
        """
        self.SimulatorClass = (
            get_custom_class(self.custom_file, "CustomSimulator") if self.custom_file else None
        )

    @abstractmethod
    def _evaluate(self, *args, **kwargs) -> list[float | list[float]] | None:
        """
        Computes all candidates outputs and return the optimizer list of QoIs.
        """


class WolfOptimizer(Optimizer, ABC):
    """
    This class implements a Wolf based Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfOptimizer object.

        **Input**

        - config (dict): the config file dictionary.
        """
        super().__init__(config)

    def set_simulator_class(self):
        """
        **Sets** the simulator class as custom if found, as WolfSimulator otherwise.
        """
        super().set_simulator_class()
        if not self.SimulatorClass:
            self.SimulatorClass = WolfSimulator

    def set_inner(self):
        """
        **Sets** some use-case specific inner variables:

        - baseline_CD (float): the drag coefficient of the baseline geometry.
        - baseline_CL (float): the lift coefficient of the baseline geometry.
        - baseline_area (float): the baseline area that is used as a structural constraint.
        - area_margin (float): area tolerance margin given as a percentage wrt baseline_area</br>
            i.e. a candidate with an area greater/smaller than +/- area_margin % of the
            baseline_area will be penalized.
        - penalty (list): a [key, value] constraint not to be worsen by the optimization.
        - constraint (bool): constraints are applied (True) or not (False)
        """
        self.baseline_CD: float = self.config["optim"].get("baseline_CD", 0.15)
        self.baseline_CL: float = self.config["optim"].get("baseline_CL", 0.36)
        self.baseline_area: float = abs(get_area(self.ffd.pts))
        self.area_margin: float = self.config["optim"].get("area_margin", 40.) / 100.
        self.penalty: list = self.config["optim"].get("penalty", ["CL", self.baseline_CL])
        self.constraint: bool = self.config["optim"].get("constraint", True)

    def plot_generation(
            self,
            gid: int,
            sorted_idx: np.ndarray,
            gen_fitness: np.ndarray,
            fig_name: str
    ):
        """
        **Plots** the results of the last evaluated generation.
        **Saves** the graph in the output directory.
        """
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]
        res_dict = self.simulator.df_dict[gid]
        df_key = res_dict[0].columns  # "ResTot", "CD", "CL", "ResCD", "ResCL", "x", "y", "Cp"

        cmap = mpl.colormaps[self.cmap].resampled(self.n_plt)
        colors = cmap(np.linspace(0, 1, self.n_plt))
        # subplot construction
        fig = plt.figure(figsize=(16, 12))
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
            res_dict[cid][df_key[0]].plot(ax=ax2, color=colors[color], label=f"c{cid}")  # ResTot
            res_dict[cid][df_key[1]].plot(ax=ax3, color=colors[color], label=f"{df_key[1]} c{cid}")
            res_dict[cid][df_key[2]].plot(
                ax=ax3, color=colors[color], ls="--", label=f"{df_key[2]} c{cid}"
            )
            ax4.scatter(cid, gen_fitness[cid], color=colors[color], label=f"c{cid}")
        # legend and title
        fig.suptitle(
            f"Generation {gid} - {self.n_plt} top candidates", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('$x$ $[m]$')
        ax1.set_ylabel('$y$ $[m]$')
        # bottom left
        ax2.set_title(f"{df_key[0]}", weight="bold")
        ax2.set_yscale("log")
        ax2.set_xlabel('iteration $[\\cdot]$')
        ax2.set_ylabel('residual $[\\cdot]$')
        # bottom center
        ax3.set_title(f"{df_key[1]} & {df_key[2]}", weight="bold")
        ax3.set_xlabel('iteration $[\\cdot]$')
        ax3.set_ylabel('aerodynamic coefficients $[\\cdot]$')
        # bottom right
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.set_title(f"fitness: {self.QoI}", weight="bold")
        ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax4.set_xlabel('candidate $[\\cdot]$')
        ax4.set_ylabel("fitness")
        # save figure as png
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()

    def save_results(self):
        super().save_results()
        with open(os.path.join(self.outdir, "df_dict.pkl"), "wb") as handle:
            pickle.dump(self.simulator.df_dict, handle)
        logger.info(f"results dictionary saved to {self.outdir}")

    @abstractmethod
    def apply_constraints(self, *args, **kwargs):
        """
        Looks for constraints violations.
        """

    @abstractmethod
    def final_observe(self, *args, **kwargs):
        """
        Plots convergence progress by plotting the fitness values
        obtained with the successive generations.
        """


class DebugOptimizer(Optimizer, ABC):
    def __init__(self, config: dict):
        """
        Dummy init.
        """
        super().__init__(config, debug=True)

    def set_simulator_class(self):
        """
        **Sets** the simulator class as custom if found, as DebugSimulator otherwise.
        """
        super().set_simulator_class()
        if not self.SimulatorClass:
            self.SimulatorClass = DebugSimulator

    def set_inner(self):
        return

    def execute_candidates(self, candidates: list[Individual] | np.ndarray, gid: int):
        """
        **Executes** all candidates and **waits** for them to finish.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.inputs.append([])
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            logger.debug(f"g{gid}, c{cid} cand {cand}")
            self.simulator.execute_sim(cand, gid, cid)
            logger.debug(f"g{gid}, c{cid} cand {cand}, "
                         f"fitness {self.simulator.df_dict[gid][cid]['result'].iloc[-1]}")
