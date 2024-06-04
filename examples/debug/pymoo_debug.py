import logging

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.optim.evolution import PymooEvolution
from src.optim.pymoo_optimizer import DebugOptimizer

logger = logging.getLogger()


class CustomOptimizer(DebugOptimizer):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM OPTIMIZER")


class CustomEvolution(PymooEvolution):
    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = PSO(
            pop_size=self.optimizer.doe_size,
            sampling=self.optimizer.generator._pymoo_generator(),
            **self.optimizer.ea_kwargs
        )

    def evolve(self):
        res = minimize(problem=self.optimizer,
                       algorithm=self.ea,
                       termination=get_termination("n_gen", self.optimizer.max_generations),
                       seed=self.optimizer.seed,
                       verbose=True)
        self.optimizer.final_observe()

        # output results
        best = res.F
        index, opt_J = min(enumerate(self.optimizer.J), key=lambda x: abs(best - x[1]))
        gid, cid = (index // self.optimizer.doe_size, index % self.optimizer.doe_size)
        logger.info(f"optimal(J): {opt_J} ({best}), "
                    f"D: {' '.join([str(d) for d in res.X])} "
                    f"[g{gid}, c{cid}]")
