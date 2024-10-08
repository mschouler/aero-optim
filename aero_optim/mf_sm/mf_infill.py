import logging
import numpy as np

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from scipy.spatial.distance import cdist
from scipy.stats import norm

from aero_optim.mf_sm.mf_models import MfSMT

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


class EDProblem(Problem):
    """
    Euclidean Distance problem.
    """
    def __init__(self, DOE: np.ndarray, model: MfSMT, n_var: int, bound: tuple[float]):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.model = model
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

    Inputs:
        y_star (float): the current best fitness value.
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


def maximize_ED(
        model: MfSMT, DOE: np.ndarray, n_var: int, bound: tuple[float], seed: int, n_gen: int = 1000
) -> np.ndarray:
    """
    Expected Improvement maximization function.

    Inputs:
        DOE (np.ndarray): the full low- and high-fidelity DOE.
    """
    problem = EDProblem(DOE, model, n_var=n_var, bound=bound)
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
