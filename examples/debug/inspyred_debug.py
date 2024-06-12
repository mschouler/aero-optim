import inspyred
import logging
import operator

from aero_optim.optim.inspyred_optimizer import InspyredDebugOptimizer
from aero_optim.optim.evolution import InspyredEvolution
from aero_optim.simulator.simulator import DebugSimulator

logger = logging.getLogger()


class CustomSimulator(DebugSimulator):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM SIMULATOR")


class CustomOptimizer(InspyredDebugOptimizer):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM OPTIMIZER")


class CustomEvolution(InspyredEvolution):
    def __init__(self, config: dict, debug: bool):
        super().__init__(config, debug)
        self.set_observe()
        self.set_terminator()

    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.algorithm = inspyred.swarm.PSO(self.optimizer.prng)

    def set_observe(self):
        logger.info("SET CUSTOM OBSERVER")
        self.algorithm.observer = self.optimizer._observe

    def set_terminator(self):
        logger.info("SET CUSTOM TERMINATOR")
        self.algorithm.terminator = inspyred.ec.terminators.generation_termination

    def evolve(self):
        logger.info("EXECUTE CUSTOM EVOLVE")
        final_pop = self.algorithm.evolve(generator=self.optimizer.generator._ins_generator,
                                          evaluator=self.optimizer._evaluate,
                                          pop_size=self.optimizer.doe_size,
                                          max_generations=self.optimizer.max_generations,
                                          bounder=inspyred.ec.Bounder(*self.optimizer.bound),
                                          maximize=self.optimizer.maximize,
                                          **self.optimizer.ea_kwargs)
        self.optimizer.final_observe()

        # output results
        best = max(final_pop)
        index, opt_J = (
            max(enumerate(self.optimizer.J), key=operator.itemgetter(1))
            if self.optimizer.maximize else
            min(enumerate(self.optimizer.J), key=operator.itemgetter(1))
        )
        gid, cid = (index // self.optimizer.doe_size, index % self.optimizer.doe_size)
        logger.info(f"optimal J: {opt_J} (J_ins: {best.fitness}),\n"
                    f"D: {' '.join([str(d) for d in self.optimizer.inputs[gid][cid]])}\n"
                    f"D_ins: {' '.join([str(d) for d in best.candidate[:self.optimizer.n_design]])}"
                    f"\n[g{gid}, c{cid}]")
