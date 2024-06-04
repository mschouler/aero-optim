import logging
import operator as ope

from abc import ABC, abstractmethod
from typing import Type

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.optim.pymoo_optimizer import DebugOptimizer as PymooDebugOptimizer
from src.optim.pymoo_optimizer import WolfOptimizer as PymooWolfOptimizer
from src.optim.pymoo_optimizer import select_strategy as pymoo_select_strategy

from inspyred.ec import Bounder, terminators
from src.optim.inspyred_optimizer import DebugOptimizer as InspyredDebugOptimizer
from src.optim.inspyred_optimizer import WolfOptimizer as InspyredWolfOptimizer
from src.optim.inspyred_optimizer import select_strategy as inspyred_select_strategy

from src.optim.optimizer import Optimizer
from src.utils import get_custom_class

logger = logging.getLogger(__name__)


class Evolution(ABC):
    """
    This class implements an abstract evolution object.
    """
    def __init__(self, config: dict, debug: bool):
        self.custom_file: str = config["study"].get("custom_file", "")
        self.set_optimizer(debug=debug)
        self.optimizer: Type[Optimizer] = self.OptimizerClass(config)
        self.set_ea()

    @abstractmethod
    def set_optimizer(self, *args, **kwargs):
        """
        Sets the optimizer object.
        """
        self.OptimizerClass = (
            get_custom_class(self.custom_file, "CustomOptimizer") if self.custom_file else None
        )

    @abstractmethod
    def set_ea(self, *args, **kwargs):
        """
        Sets the evolutionary computation algorithm.
        """

    @abstractmethod
    def evolve(self, *args, **kwargs):
        """
        Defines how to execute the optimization.
        """


class PymooEvolution(Evolution):
    """
    This class implements a default pymoo based evolution object.
    """
    def __init__(self, config: dict, debug: bool = False):
        super().__init__(config, debug)

    def set_optimizer(self, debug: bool = False):
        """
        **Instantiates** the optimizer attribute as custom if any or from default classes.
        """
        super().set_optimizer()
        if not self.OptimizerClass:
            if debug:
                self.OptimizerClass = PymooDebugOptimizer
            else:
                self.OptimizerClass = PymooWolfOptimizer
            logger.info(f"optimizer set to {self.OptimizerClass}")

    def set_ea(self):
        """
        **Instantiates** the default algorithm attribute.
        """
        self.algorithm = pymoo_select_strategy(
            self.optimizer.strategy,
            self.optimizer.doe_size,
            self.optimizer.generator._pymoo_generator(),
            self.optimizer.ea_kwargs
        )

    def evolve(self):
        """
        **Executes** the default evolution method.
        """
        res = minimize(problem=self.optimizer,
                       algorithm=self.algorithm,
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


class InspyredEvolution(Evolution):
    """
    This class implements a default inspyred based evolution object.
    """
    def __init__(self, config: dict, debug: bool = False):
        super().__init__(config, debug)
        self.algorithm.observer = self.optimizer._observe
        self.algorithm.terminator = terminators.generation_termination

    def set_optimizer(self, debug: bool = False):
        """
        **Instantiates** the optimizer attribute as custom if any or from default classes.
        """
        super().set_optimizer()
        if not self.OptimizerClass:
            if debug:
                self.OptimizerClass = InspyredDebugOptimizer
            else:
                self.OptimizerClass = InspyredWolfOptimizer
            logger.info(f"optimizer set to {self.OptimizerClass}")

    def set_ea(self):
        """
        **Instantiates** the default algorithm attribute.
        """
        self.algorithm = inspyred_select_strategy(self.optimizer.strategy, self.optimizer.prng)

    def evolve(self):
        """
        **Executes** the default evolution method.
        """
        final_pop = self.algorithm.evolve(generator=self.optimizer.generator._ins_generator,
                                          evaluator=self.optimizer._evaluate,
                                          pop_size=self.optimizer.doe_size,
                                          max_generations=self.optimizer.max_generations,
                                          bounder=Bounder(*self.optimizer.bound),
                                          maximize=self.optimizer.maximize,
                                          **self.optimizer.ea_kwargs)

        self.optimizer.final_observe()

        # output results
        best = max(final_pop)
        index, opt_J = (
            max(enumerate(self.optimizer.J), key=ope.itemgetter(1)) if self.optimizer.maximize else
            min(enumerate(self.optimizer.J), key=ope.itemgetter(1))
        )
        gid, cid = (index // self.optimizer.doe_size, index % self.optimizer.doe_size)
        logger.info(f"optimal(J): {opt_J}, "
                    f"D: {' '.join([str(d) for d in best.candidate[:self.optimizer.n_design]])} "
                    f"[g{gid}, c{cid}]")
