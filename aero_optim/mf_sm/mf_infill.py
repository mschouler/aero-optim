import logging
import numpy as np

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from scipy.spatial.distance import cdist
from scipy.stats import norm

from aero_optim.mf_sm.mf_models import MfSMT, MultiObjectiveModel

logger = logging.getLogger(__name__)


class LCBProblem(Problem):
    """
    Lower Confidence Bound problem.
    """
    def __init__(self, model: MfSMT, n_var: int, bound: tuple[float]):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.model = model

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        f1 = self.model.evaluate(x) - self.model.evaluate_std(x)
        assert f1.shape == (x.shape[0], 1)
        out["F"] = f1


class EIProblem(LCBProblem):
    """
    Expected Improvement problem.
    """
    epsilon: float = 1e-6

    def __init__(self, model: MfSMT, n_var: int, bound: tuple[float]):
        super().__init__(model=model, n_var=n_var, bound=bound)

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        u = self.model.get_y_star() - self.model.evaluate(x)
        std = self.model.evaluate_std(x)
        f1 = u * norm.cdf(u / (std + self.epsilon)) + std * norm.pdf(u / (std + self.epsilon))
        zero_std_idx = np.argwhere(std < self.epsilon)
        f1[zero_std_idx] = 0.
        assert f1.shape == (x.shape[0], 1)
        out["F"] = - f1


class PIBOProblem(Problem):
    """
    Bi-objective Probability Improvement problem.
    """
    epsilon: float = 1e-6

    def __init__(self, model: MultiObjectiveModel, n_var: int, bound: tuple[float]):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.model = model

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        pareto = self.model.compute_pareto()
        model1 = self.model.models[0]
        model2 = self.model.models[1]

        u1 = model1.evaluate(x)
        u2 = model2.evaluate(x)
        std1 = model1.evaluate_std(x) + self.epsilon
        std2 = model2.evaluate_std(x) + self.epsilon

        PI_aug = norm.cdf((pareto[0, 0] - u1) / std1)
        for i in range(len(pareto) - 1):
            PI_aug += (
                norm.cdf((pareto[i + 1, 0] - u1) / std1)
                - norm.cdf((pareto[i, 0] - u1) / std1) * norm.cdf((pareto[i, 1] - u2) / std2)
                + (1 - norm.cdf((pareto[-1, 0] - u1) / std1))
                * norm.cdf((pareto[-1, 1] - u2) / std2)
            )
        out["F"] = - PI_aug


class EDProblem(Problem):
    """
    Euclidean Distance problem.
    """
    def __init__(
            self,
            DOE: np.ndarray,
            n_var: int,
            bound: tuple[float]
    ):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.DOE = DOE

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        f1 = np.min(cdist(np.atleast_2d(x), self.DOE, "euclidean"), axis=1)
        f1 = np.expand_dims(f1, axis=1)
        assert f1.shape == (x.shape[0], 1), f"f1 shape {f1.shape} x shape {x.shape}"
        out["F"] = - f1


def minimize_LCB(
        model: MfSMT, n_var: int, bound: tuple[float], seed: int, n_gen: int = 1000
) -> np.ndarray:
    """
    Lower Confidence Bound minimization function.
    """
    problem = LCBProblem(model, n_var=n_var, bound=bound)
    algorithm = PSO()
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False
    )
    logger.info(f"LCB adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_EI(
        model: MfSMT, n_var: int, bound: tuple[float], seed: int, n_gen: int = 1000
) -> np.ndarray:
    """
    Expected Improvement maximization function.
    """
    problem = EIProblem(model, n_var=n_var, bound=bound)
    algorithm = PSO()
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False
    )
    logger.info(f"EI adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_PI_BO(
        model: MultiObjectiveModel, n_var: int, bound: tuple[float], seed: int, n_gen: int = 1000
) -> np.ndarray:
    """
    Bi-objective Probability of Improvement maximization
    """
    problem = PIBOProblem(model, n_var=n_var, bound=bound)
    algorithm = PSO()
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False
    )
    logger.info(f"PIBO adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_ED(
        DOE: np.ndarray, n_var: int, bound: tuple[float], seed: int, n_gen: int = 1000
) -> np.ndarray:
    """
    Euclidean distance maximization function.

    Inputs:
        DOE (np.ndarray): the full low- and high-fidelity DOE.
    """
    problem = EDProblem(DOE, n_var=n_var, bound=bound)
    algorithm = PSO()
    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False
    )
    logger.info(f"ED adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X
