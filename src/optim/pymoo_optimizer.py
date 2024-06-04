import logging
import matplotlib.pyplot as plt
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from src.optim.optimizer import Optimizer, shoe_lace
from src.simulator.simulator import DebugSimulator, WolfSimulator

plt.set_loglevel(level='warning')
logger = logging.getLogger(__name__)


def select_strategy(strategy_name: str, doe_size: int, X0: np.ndarray, options: dict) -> GA | PSO:
    """
    Returns the evolution algorithm object if the strategy is well defined,
    an exception otherwise.
    """
    if strategy_name == "GA":
        ea = GA(pop_size=doe_size, sampling=X0, **options)
    elif strategy_name == "PSO":
        ea = PSO(pop_size=doe_size, sampling=X0, **options)
    else:
        raise Exception(f"ERROR -- unsupported strategy {strategy_name}")
    logger.info(f"optimization selected strategy: {strategy_name}")
    return ea


class WolfOptimizer(Optimizer, Problem):
    """
    This class implements a Wolf based Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfOptimizer object.

        **Input**

        - config (dict): the config file dictionary.
        """
        Optimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=1, n_ieq_constr=2, xl=self.bound[0], xu=self.bound[1]
        )

    def set_simulator(self):
        """
        **Sets** the simulator object as custom if found, as WolfSimulator otherwise.
        """
        super().set_simulator()
        if not self.SimulatorClass:
            self.SimulatorClass = WolfSimulator

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Executes** Wolf simulations, **extracts** results
        and **returns** arrays of candidates QoIs and constraints.
        """
        gid = self.gen_ctr

        # execute all candidates
        self.execute_candidates(X, gid)

        # add penalty to the candidates fitness
        self.J.extend([
            self.simulator.df_dict[gid][cid][self.QoI].iloc[-1] for cid in range(len(X))
        ])

        self.gen_ctr += 1
        out["F"] = np.array(self.J[-self.doe_size:])
        out["G"] = self.apply_inequality_constraints(gid)
        self._observe(out["F"])

    def apply_inequality_constraints(self, gid: int) -> np.ndarray:
        """
        **Returns** a constraint array ensuring negative inequality</br>
        see https://pymoo.org/constraints/index.html
        """
        out = []
        for cid, pro in enumerate(self.ffd_profiles[gid]):
            ieq_1 = abs(shoe_lace(pro) - self.baseline_area) / self.baseline_area - self.area_margin
            ieq_2 = self.penalty[-1] - self.simulator.df_dict[gid][cid][self.penalty[0]].iloc[-1]
            if ieq_1 > 0 or ieq_2 > 0:
                logger.info(f"penalized candidate g{gid}, c{cid} "
                            f"with area {shoe_lace(pro)} "
                            f"and CL {self.simulator.df_dict[gid][cid][self.penalty[0]].iloc[-1]}")
            out.append([ieq_1, ieq_2])
        return np.row_stack(out)

    def _observe(self, pop_fitness: np.ndarray):
        """
        **Plots** the n_plt best results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the simulations CD & CL,</br>
        > the candidates fitness,</br>
        > the baseline and deformed profiles.
        """
        gid = self.gen_ctr - 1

        # extract generation best profiles
        sorted_idx = np.argsort(pop_fitness, kind="stable")[:self.n_plt]

        # compute population statistics
        self.compute_statistics(pop_fitness)

        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

        # plot settings
        fig_name = f"pymoo_g{gid}.png"
        self.plot_generation(gid, sorted_idx, pop_fitness, fig_name)

    def final_observe(self):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations
        """
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name, baseline_value=self.baseline_CD)


class DebugOptimizer(Optimizer, Problem):
    def __init__(self, config: dict):
        """
        Dummy init.
        """
        Optimizer.__init__(self, config, debug=True)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=1, n_ieq_constr=0, xl=self.bound[0], xu=self.bound[1]
        )

    def set_simulator(self):
        """
        **Sets** the simulator object as custom if found, as DebugSimulator otherwise.
        """
        super().set_simulator()
        if not self.SimulatorClass:
            self.SimulatorClass = DebugSimulator

    def set_inner(self):
        return

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Executes** dummy simulations, **extracts** results
        and **returns** the list of candidates QoIs.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        gid = self.gen_ctr
        self.simulator.df_dict[gid] = {}
        logger.debug(f"g{gid} evaluation..")

        # execute all candidates
        for cid, cand in enumerate(X):
            logger.debug(f"g{gid}, c{cid} cand {cand}")
            self.simulator.execute_sim(cand, gid, cid)
            logger.debug(f"g{gid}, c{cid} cand {cand}, "
                         f"fitness {self.simulator.df_dict[gid][cid]['result'].iloc[-1]}")

        for cid, _ in enumerate(X):
            self.J.append(self.simulator.df_dict[gid][cid]["result"].iloc[-1])

        self.gen_ctr += 1
        out["F"] = np.array(self.J[-self.doe_size:])
        self._observe(out["F"])

    def _observe(self, pop_fitness: np.ndarray):
        """
        Dummy observe function.
        """
        # extract best profiles
        gid = self.gen_ctr - 1
        sorted_idx = np.argsort(pop_fitness, kind="stable")[:self.n_plt]
        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

        # compute population statistics
        self.compute_statistics(pop_fitness)

    def final_observe(self):
        """
        Dummy final oberve function.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name)
