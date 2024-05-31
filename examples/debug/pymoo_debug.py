import logging

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.optim.optimizer import ABCCustomEvolution
from src.optim.pymoo_optimizer import DEBUGOptimizer

logger = logging.getLogger()


class CustomOptimizer(DEBUGOptimizer):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM OPTIMIZER")


class CustomEvolution(ABCCustomEvolution):
    def __init__(self, config: dict):
        self.opt = CustomOptimizer(config)
        self.set_ea()

    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = PSO(
            pop_size=self.opt.doe_size,
            sampling=self.opt.generator._pymoo_generator(),
            **self.opt.ea_kwargs
        )

    def custom_evolve(self):
        res = minimize(problem=self.opt,
                       algorithm=self.ea,
                       termination=get_termination("n_gen", self.opt.max_generations),
                       seed=self.opt.seed,
                       verbose=True)
        self.opt.final_observe()

        # output results
        best = res.F
        index, opt_J = min(enumerate(self.opt.J), key=lambda x: abs(best - x[1]))
        gid, cid = (index // self.opt.doe_size, index % self.opt.doe_size)
        logger.info(f"optimal(J): {opt_J} ({best}), "
                    f"D: {' '.join([str(d) for d in res.X])} "
                    f"[g{gid}, c{cid}]")
