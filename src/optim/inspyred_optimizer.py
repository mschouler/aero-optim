import inspyred
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import signal
import time

from inspyred.ec import Individual
from random import Random
from src.optim.optimizer import Optimizer, shoe_lace
from src.simulator.simulator import WolfSimulator, DEBUGSimulator

plt.set_loglevel(level='warning')
logger = logging.getLogger(__name__)


def select_strategy(strategy_name: str, prng: Random) -> inspyred.ec.EvolutionaryComputation:
    """
    Returns the evolution algorithm object if the strategy is well defined,
    an exception otherwise.
    """
    if strategy_name == "ES":
        ea = inspyred.ec.ES(prng)
    elif strategy_name == "PSO":
        ea = inspyred.swarm.PSO(prng)
    else:
        raise Exception(f"ERROR -- unsupported strategy {strategy_name}")
    logger.info(f"optimization selected strategy: {strategy_name}")
    return ea


class WolfOptimizer(Optimizer):
    """
    This class implements a Wolf based Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfOptimizer object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - simulator (WolfSimulator): WolfSimulator object to perform Wolf simulations.
        - J (list[float]): the list of all generated candidates fitnesses.
        - ffd_profiles (list[list[np.ndarray]]): all deformed geometries {gid: {cid: ffd_profile}}.
        - QoI (str): the quantity of intereset to minimize/maximize.
        - n_plt (int): the number of best candidates results to display after each evaluation.
        - baseline_CD (float): the drag coefficient of the baseline geometry.
        - baseline_CL (float): the lift coefficient of the baseline geometry.
        - baseline_area (float): the baseline area that is used as a structural constraint.
        - area_margin (float): area tolerance margin given as a percentage wrt baseline_area</br>
            i.e. a candidate with an area greater/smaller than +/- area_margin % of the
            baseline_area will be penalized.
        - penalty (list): a [key, value] constraint not to be worsen by the optimization.
        - cmap (str): the colormaps used for the observer plot</br>
            see https://matplotlib.org/stable/users/explain/colors/colormaps.html.
        """
        super().__init__(config)
        self.simulator: WolfSimulator = WolfSimulator(self.config)
        self.J: list[float] = []
        self.ffd_profiles: list[list[np.ndarray]] = []
        self.QoI: str = config["optim"].get("QoI", "CD")
        self.n_plt: int = config["optim"].get("n_plt", 5)
        self.baseline_CD: float = config["optim"].get("baseline_CD", 0.15)
        self.baseline_CL: float = config["optim"].get("baseline_CL", 0.36)
        self.baseline_area: float = shoe_lace(self.ffd.pts)
        self.area_margin: float = config["optim"].get("area_margin", 40.) / 100.
        self.penalty: list = config["optim"].get("penalty", ["CL", self.baseline_CL])
        self.cmap: str = config["optim"].get("cmap", "viridis")

    def _evaluate(self, candidates: list[Individual], args: dict) -> list[float]:
        """
        **Executes** Wolf simulations, **extracts** results
        and **returns** the list of candidates QoIs.

        Note:
            __candidates__ and __args__ are inspyred mandatory arguments</br>
            see https://pythonhosted.org/inspyred/tutorial.html#the-evaluator
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
                    self.simulator.df_dict[gid][cid][self.penalty[0]].iloc[-1]
                )
            )
            self.J[-1] += self.simulator.df_dict[gid][cid][self.QoI].iloc[-1]

        self.gen_ctr += 1
        return self.J[-self.doe_size:]

    def constraint(self, gid: int, cid: int, ffd_profile: np.ndarray, pen_value: float) -> float:
        """
        **Returns** a penalty value based on some specific constraints</br>
        see https://inspyred.readthedocs.io/en/latest/recipes.html#constraint-selection
        """
        area_cond: bool = (shoe_lace(ffd_profile) > (1. + self.area_margin) * self.baseline_area
                           or shoe_lace(ffd_profile) < (1. - self.area_margin) * self.baseline_area)
        penalty_cond: bool = pen_value < self.penalty[-1]
        if area_cond or penalty_cond:
            logger.info(f"penalized candidate g{gid}, c{cid} "
                        f"with area {shoe_lace(ffd_profile)} and CL {pen_value}")
            return 1.
        return 0.

    def _observe(
            self,
            population: list[Individual],
            num_generations: int,
            num_evaluations: int,
            args: dict
    ):
        """
        **Plots** the n_plt best results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the simulations CD & CL,</br>
        > the candidates fitness,</br>
        > the baseline and deformed profiles.

        Note:
            __num_generations__, __num_evaluations__ and __args__
            are inspyred mandatory arguments</br>
            see https://pythonhosted.org/inspyred/examples.html#custom-observer
        """
        gid = num_generations
        res_dict = self.simulator.df_dict[gid]
        df_key = res_dict[0].columns  # "ResTot", "CD", "CL", "ResCD", "ResCL", "x", "y", "Cp"

        # extract generation best profiles
        fitness: np.ndarray = np.array(self.J[-self.doe_size:])
        sorted_idx = np.argsort(fitness, kind="stable")[:self.n_plt]
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]

        # compute population statistics
        self.mean.append(np.mean([ind.fitness for ind in population]))
        self.median.append(np.median([ind.fitness for ind in population]))
        self.min.append(min([ind.fitness for ind in population]))
        self.max.append(max([ind.fitness for ind in population]))

        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {fitness}")
        logger.debug(f"g{gid} P-fitness (population) {[ind.fitness for ind in population]}")

        # plot settings
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
            ax4.scatter(cid, fitness[cid], color=colors[color], label=f"c{cid}")
        # legend and title
        fig.suptitle(
            f"Generation {gid} - {self.n_plt} top candidates", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom left
        ax2.set_title(f"{df_key[0]}", weight="bold")
        ax2.set_yscale("log")
        ax2.set_xlabel('it. #')
        ax2.set_ylabel('residual')
        # bottom center
        ax3.set_title(f"{df_key[1]} & {df_key[2]}", weight="bold")
        ax3.set_xlabel('it. #')
        ax3.set_ylabel('aerodynamic coeff.')
        # bottom right
        ax4.set_title(f"fitness: penalized {self.QoI}", weight="bold")
        ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax4.set_xlabel('candidate #')
        ax4.set_ylabel("fitness")
        # save figure as png
        fig_name = f"res_g{num_generations}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()

    def final_observe(self):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations</br>
        see https://pythonhosted.org/inspyred/reference.html#inspyred.ec.analysis.generation_plot
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr - 1} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        psize = self.doe_size
        ax.axhline(y=self.baseline_CD, color='k', ls="--", label="baseline")

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

        # legend and title
        ax.set_title(f"Populations evolution ({self.gen_ctr - 1} g. x {psize} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('generation #')
        ax.set_ylabel('penalized fitness')

        # save figure as png
        fig_name = f"optim_g{self.gen_ctr - 1}_c{psize}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()


class DEBUGOptimizer(Optimizer):
    def __init__(self, config: dict):
        """
        Dummy init.
        """
        super().__init__(config, debug=True)
        self.simulator: DEBUGSimulator = DEBUGSimulator(config)
        self.J: list[float] = []
        self.n_plt: int = 5

    def _evaluate(self, candidates: list[Individual], args: dict) -> list[float]:
        """
        Executes dummy simulations, extracts results and returns the list of candidates QoIs.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        gid = self.gen_ctr
        self.simulator.df_dict[gid] = {}
        logger.debug(f"g{gid} evaluation..")

        # execute all candidates
        for cid, cand in enumerate(candidates):
            logger.debug(f"g{gid}, c{cid} cand {cand}")
            self.simulator.execute_sim(cand, gid, cid)
            logger.debug(f"g{gid}, c{cid} cand {cand}, "
                         f"fitness {self.simulator.df_dict[gid][cid]['result'].iloc[-1]}")

        for cid, _ in enumerate(candidates):
            self.J.append(self.simulator.df_dict[gid][cid]["result"].iloc[-1])

        self.gen_ctr += 1
        return self.J[-self.doe_size:]

    def _observe(
            self,
            population: list[Individual],
            num_generations: int,
            num_evaluations: int,
            args: dict
    ):
        """
        Dummy observe function.
        """
        # extract best profiles
        gid = num_generations
        fitness: np.ndarray = np.array(self.J[-self.doe_size:])
        sorted_idx = np.argsort(fitness, kind="stable")[:self.n_plt]
        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {fitness}")
        logger.debug(f"g{gid} P-fitness (population) {[ind.fitness for ind in population]}")

        # compute population statistics
        self.mean.append(np.mean([ind.fitness for ind in population]))
        self.median.append(np.median([ind.fitness for ind in population]))
        self.min.append(min([ind.fitness for ind in population]))
        self.max.append(max([ind.fitness for ind in population]))

    def final_observe(self):
        """
        Dummy final oberve function.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr - 1} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        psize = self.doe_size

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

        # legend and title
        ax.set_title(f"Optimization evolution ({self.gen_ctr - 1} g. x {psize} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('generation #')
        ax.set_ylabel('penalized fitness')

        # save figure as png
        fig_name = f"optim_g{self.gen_ctr - 1}_c{psize}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()
