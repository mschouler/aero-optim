import inspyred
import logging
import matplotlib.pyplot as plt
import numpy as np

from inspyred.ec import Individual
from random import Random
from src.optim.optimizer import Optimizer, shoe_lace
from src.simulator.simulator import WolfSimulator, DebugSimulator

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
        """
        super().__init__(config)

    def set_simulator(self):
        """
        **Sets** the simulator object as custom if found, as WolfSimulator otherwise.
        """
        super().set_simulator()
        if not self.SimulatorClass:
            self.SimulatorClass = WolfSimulator

    def _evaluate(self, candidates: list[Individual], args: dict) -> list[float]:
        """
        **Executes** Wolf simulations, **extracts** results
        and **returns** the list of candidates QoIs.

        Note:
            __candidates__ and __args__ are inspyred mandatory arguments</br>
            see https://pythonhosted.org/inspyred/tutorial.html#the-evaluator
        """
        gid = self.gen_ctr

        # execute all candidates
        self.execute_candidates(candidates, gid)

        # add penalty to the candidates fitness
        for cid, _ in enumerate(candidates):
            self.J.append(
                self.apply_constraints(
                    gid, cid,
                    self.ffd_profiles[gid][cid],
                    self.simulator.df_dict[gid][cid][self.penalty[0]].iloc[-1]
                )
            )
            self.J[-1] += self.simulator.df_dict[gid][cid][self.QoI].iloc[-1]

        self.gen_ctr += 1
        return self.J[-self.doe_size:]

    def apply_constraints(
            self, gid: int, cid: int, ffd_profile: np.ndarray, pen_value: float
    ) -> float:
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

        # extract generation best profiles
        fitness: np.ndarray = np.array(self.J[-self.doe_size:])
        sorted_idx = np.argsort(fitness, kind="stable")[:self.n_plt]

        # compute population statistics
        self.compute_statistics(np.array([ind.fitness for ind in population]))

        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {fitness}")
        logger.debug(f"g{gid} P-fitness (population) {[ind.fitness for ind in population]}")

        # plot settings
        fig_name = f"inspyred_g{num_generations}.png"
        self.plot_generation(gid, sorted_idx, fitness, fig_name)

    def final_observe(self):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations</br>
        see https://pythonhosted.org/inspyred/reference.html#inspyred.ec.analysis.generation_plot
        """
        fig_name = f"inspyred_optim_g{self.gen_ctr - 1}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr - 1, fig_name, baseline_value=self.baseline_CD)


class DebugOptimizer(Optimizer):
    def __init__(self, config: dict):
        """
        Dummy init.
        """
        super().__init__(config, debug=True)

    def set_simulator(self):
        """
        **Sets** the simulator object as custom if found, as DebugSimulator otherwise.
        """
        super().set_simulator()
        if not self.SimulatorClass:
            self.SimulatorClass = DebugSimulator

    def set_inner(self):
        return

    def _evaluate(self, candidates: list[Individual], args: dict) -> list[float]:
        """
        **Executes** dummy simulations, **extracts** results
        and **returns** the list of candidates QoIs.
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
        self.compute_statistics(np.array([ind.fitness for ind in population]))

    def final_observe(self):
        """
        Dummy final oberve function.
        """
        fig_name = f"inspyred_optim_g{self.gen_ctr - 1}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr - 1, fig_name)
