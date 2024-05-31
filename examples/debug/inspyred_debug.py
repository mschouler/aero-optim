import inspyred
import logging
import operator

from src.optim.inspyred_optimizer import DEBUGOptimizer
from src.optim.optimizer import ABCCustomEvolution

logger = logging.getLogger()


class CustomOptimizer(DEBUGOptimizer):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM OPTIMIZER")


class CustomEvolution(ABCCustomEvolution):
    def __init__(self, config: dict):
        self.opt = CustomOptimizer(config)
        self.set_ea()
        self.set_observe()
        self.set_terminator()

    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = inspyred.swarm.PSO(self.opt.prng)

    def set_observe(self):
        logger.info("SET CUSTOM OBSERVER")
        self.ea.observer = self.opt._observe

    def set_terminator(self):
        logger.info("SET CUSTOM TERMINATOR")
        self.ea.terminator = inspyred.ec.terminators.generation_termination

    def custom_evolve(self):
        final_pop = self.ea.evolve(generator=self.opt.generator._ins_generator,
                                   evaluator=self.opt._evaluate,
                                   pop_size=self.opt.doe_size,
                                   max_generations=self.opt.max_generations,
                                   bounder=inspyred.ec.Bounder(*self.opt.bound),
                                   maximize=self.opt.maximize,
                                   **self.opt.ea_kwargs)
        self.opt.final_observe()

        # output results
        best = max(final_pop)
        index, opt_J = (
            max(enumerate(self.opt.J), key=operator.itemgetter(1)) if self.opt.maximize else
            min(enumerate(self.opt.J), key=operator.itemgetter(1))
        )
        gid, cid = (index // self.opt.doe_size, index % self.opt.doe_size)
        logger.info(f"optimal(J): {opt_J}, "
                    f"D: {' '.join([str(d) for d in best.candidate[:self.opt.n_design]])} "
                    f"[g{gid}, c{cid}]")
