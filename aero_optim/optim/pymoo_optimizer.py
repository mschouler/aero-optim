import logging
import numpy as np

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem

from aero_optim.geom import get_area
from aero_optim.optim.optimizer import DebugOptimizer, WolfOptimizer

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


class PymooWolfOptimizer(WolfOptimizer, Problem):
    """
    This class implements a Wolf based Optimizer.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfOptimizer object.

        **Input**

        - config (dict): the config file dictionary.
        """
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=1, n_ieq_constr=2, xl=self.bound[0], xu=self.bound[1]
        )

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Executes** Wolf simulations, **extracts** results
        and **returns** arrays of candidates QoIs and constraints.
        """
        gid = self.gen_ctr

        # execute all candidates
        self.execute_candidates(X, gid)

        # update candidates fitness
        self.J.extend([
            self.simulator.df_dict[gid][cid][self.QoI].iloc[-1] for cid in range(len(X))
        ])

        out["F"] = np.array(self.J[-self.doe_size:])
        out["G"] = self.apply_constraints(gid)
        self._observe(out["F"])
        self.gen_ctr += 1

    def apply_constraints(self, gid: int) -> np.ndarray:
        """
        **Returns** a constraint array ensuring negative inequality</br>
        see https://pymoo.org/constraints/index.html
        """
        out = []
        for cid, pro in enumerate(self.ffd_profiles[gid]):
            ieq_1 = (
                abs(abs(get_area(pro)) - self.baseline_area) / self.baseline_area - self.area_margin
            )
            ieq_2 = self.penalty[-1] - self.simulator.df_dict[gid][cid][self.penalty[0]].iloc[-1]
            if ieq_1 > 0 or ieq_2 > 0:
                logger.info(f"penalized candidate g{gid}, c{cid} "
                            f"with area {abs(get_area(pro))} "
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
        gid = self.gen_ctr

        # extract generation best profiles
        sorted_idx = np.argsort(pop_fitness, kind="stable")[:self.n_plt]

        # compute population statistics
        self.compute_statistics(pop_fitness)

        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

        # plot settings
        fig_name = f"pymoo_g{gid}.png"
        self.plot_generation(gid, sorted_idx, pop_fitness, fig_name)

    def final_observe(self, *args, **kwargs):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations
        """
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name, baseline_value=self.baseline_CD)


class PymooDebugOptimizer(DebugOptimizer, Problem):
    def __init__(self, config: dict):
        """
        Dummy init.
        """
        DebugOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=1, n_ieq_constr=0, xl=self.bound[0], xu=self.bound[1]
        )

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Executes** dummy simulations, **extracts** results
        and **returns** the list of candidates QoIs.
        """
        gid = self.gen_ctr

        # execute all candidates
        self.execute_candidates(X, gid)

        for cid, _ in enumerate(X):
            self.J.append(self.simulator.df_dict[gid][cid]["result"].iloc[-1])

        out["F"] = np.array(self.J[-self.doe_size:])
        self._observe(out["F"])
        self.gen_ctr += 1

    def _observe(self, pop_fitness: np.ndarray):
        """
        Dummy _observe function.
        """
        # extract best profiles
        gid = self.gen_ctr
        sorted_idx = np.argsort(pop_fitness, kind="stable")[:self.n_plt]
        logger.info(f"extracting {self.n_plt} best profiles in g{gid}: {sorted_idx}..")
        logger.debug(f"g{gid} J-fitnesses (candidates): {pop_fitness}")

        # compute population statistics
        self.compute_statistics(pop_fitness)

    def final_observe(self):
        """
        Dummy final_observe function.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        self.plot_progress(self.gen_ctr, fig_name)
