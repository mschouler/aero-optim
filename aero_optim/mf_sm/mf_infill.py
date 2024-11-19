import logging
import numpy as np

from typing import Callable

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from scipy.spatial.distance import cdist
from scipy.stats import norm

from aero_optim.mf_sm.mf_models import MfLGP, MfKPLS, MfSMT, MultiObjectiveModel, SfSMT

logger = logging.getLogger(__name__)

EPSILON: float = 1e-6


def ED_acquisition_function(x: np.ndarray, DOE: np.ndarray) -> np.ndarray:
    """
    Euclidean Distance:
    see X. Zhang et al. (2021): https://doi.org/10.1016/j.cma.2020.113485
    """
    f1 = np.min(cdist(np.atleast_2d(x), DOE, "euclidean"), axis=1)
    f1 = np.expand_dims(f1, axis=1)
    assert f1.shape == (x.shape[0], 1), f"f1 shape {f1.shape} x shape {x.shape}"
    return f1


def LCB_acquisition_function(x: np.ndarray, model: MfSMT, alpha: float = 1) -> np.ndarray:
    """
    Lower Confidence Bound acquisition function.
    """
    return model.evaluate(x) - alpha * model.evaluate_std(x)


def EI_acquisition_function(x: np.ndarray, model: MfSMT) -> np.ndarray:
    """
    Expected Improvement acquisition function.
    """
    u = model.get_y_star() - model.evaluate(x)
    std = model.evaluate_std(x)
    f1 = u * norm.cdf(u / (std + EPSILON)) + std * norm.pdf(u / (std + EPSILON))
    zero_std_idx = np.argwhere(std < EPSILON)
    f1[zero_std_idx] = 0.
    return f1


def PI_acquisition_function(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Bi-objective Probability of Improvement:
    see A. J. Keane (2006): https://doi.org/10.2514/1.16875
    """
    assert model.models[0].y_hf_DOE is not None
    assert model.models[1].y_hf_DOE is not None
    pareto = compute_pareto(model.models[0].y_hf_DOE, model.models[1].y_hf_DOE)

    u1 = model.models[0].evaluate(x)
    u2 = model.models[1].evaluate(x)
    std1 = model.models[0].evaluate_std(x) + EPSILON
    std2 = model.models[1].evaluate_std(x) + EPSILON

    PIaug = norm.cdf((pareto[0, 0] - u1) / std1)
    for i in range(len(pareto) - 1):
        PIaug += (
            (norm.cdf((pareto[i + 1, 0] - u1) / std1) - norm.cdf((pareto[i, 0] - u1) / std1))
            * norm.cdf((pareto[i, 1] - u2) / std2)
        )
    PIaug += (1 - norm.cdf((pareto[-1, 0] - u1) / std1)) * norm.cdf((pareto[-1, 1] - u2) / std2)
    return PIaug


def MPI_acquisition_function(x: np.ndarray, model: MultiObjectiveModel) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement:
    see A. A. Rahat (2017): https://dl.acm.org/doi/10.1145/3071178.3071276
    """
    assert model.models[0].y_hf_DOE is not None
    assert model.models[1].y_hf_DOE is not None
    pareto = compute_pareto(model.models[0].y_hf_DOE, model.models[1].y_hf_DOE)

    u1 = model.models[0].evaluate(x)
    u2 = model.models[1].evaluate(x)
    std1 = model.models[0].evaluate_std(x) + EPSILON
    std2 = model.models[1].evaluate_std(x) + EPSILON

    MPI = np.min(np.column_stack(
        [1 - norm.cdf((u1 - pp[0]) / std1) * norm.cdf((u2 - pp[1]) / std2)
            for pp in pareto]
    ), axis=1)
    return MPI


class EDProblem(Problem):
    """
    Euclidean Distance problem.
    """
    def __init__(self, DOE: np.ndarray, n_var: int, bound: list):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.DOE = DOE

    def _evaluate(self, x: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["F"] = - ED_acquisition_function(x, self.DOE)


class AcquisitionFunctionProblem(Problem):
    """
    Generic class for acquisition function optimization problems.
    """
    def __init__(
            self,
            function: Callable,
            model: MfLGP | MfKPLS | MfSMT | SfSMT | MultiObjectiveModel,
            n_var: int,
            bound: list,
            minimize: bool = True
    ):
        Problem.__init__(
            self, n_var=n_var, n_ieq_constr=0, xl=bound[0], xu=bound[1]
        )
        self.model = model
        self.function = function
        self.minimize = minimize

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        out["F"] = self.function(X, self.model) if self.minimize else -self.function(X, self.model)


class RegCritProblem(Problem):
    """
    Regularized problem:
    see R. Grapin et al. (2022): https://doi.org/10.2514/6.2022-4053
    """
    def __init__(
            self,
            function: Callable,
            model: MultiObjectiveModel,
            n_var: int,
            bound: list,
            gamma: float
    ):
        super().__init__(n_var=n_var, n_obj=1, xl=bound[0], xu=bound[-1])
        self.model = model
        self.gamma = gamma
        self.function = function

    def _evaluate(self, x: np.ndarray, out: np.ndarray):
        u1 = self.model.models[0].evaluate(x)
        u2 = self.model.models[1].evaluate(x)
        out["F"] = - (self.gamma * self.function(x, self.model)
                      - np.sum(np.column_stack([u1, u2]), axis=1))


def maximize_ED(
        DOE: np.ndarray, n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Euclidean distance maximization function.

    Inputs:
        DOE (np.ndarray): the full low- and high-fidelity DOE.
    """
    problem = EDProblem(DOE, n_var=n_var, bound=bound)
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"ED adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def minimize_LCB(
        model: MfLGP | MfKPLS | MfSMT | SfSMT,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Lower Confidence Bound minimization function.
    """
    problem = AcquisitionFunctionProblem(
        LCB_acquisition_function, model, n_var=n_var, bound=bound
    )
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"LCB adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_EI(
        model: MfLGP | MfKPLS | MfSMT | SfSMT,
        n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Expected Improvement maximization function.
    """
    problem = AcquisitionFunctionProblem(
        EI_acquisition_function, model, n_var=n_var, bound=bound, minimize=False
    )
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"EI adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_PI_BO(
        model: MultiObjectiveModel, n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Bi-objective Probability of Improvement maximization.
    """
    problem = AcquisitionFunctionProblem(
        PI_acquisition_function, model, n_var=n_var, bound=bound, minimize=False
    )
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"PIBO adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_MPI_BO(
        model: MultiObjectiveModel, n_var: int, bound: list, seed: int, n_gen: int = 100
) -> np.ndarray:
    """
    Bi-objective Minimal Probability of Improvement maximization.
    """
    problem = AcquisitionFunctionProblem(
        MPI_acquisition_function, model, n_var=n_var, bound=bound, minimize=False
    )
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"MPIBO adaptive infill best solution:\n X = {res.X}\n F = {res.F}")
    return res.X


def maximize_RegCrit(
    acquisition_function: Callable,
    model: MultiObjectiveModel,
    n_var: int,
    bound: list,
    seed: int,
    n_gen: int = 100
) -> np.ndarray:
    """
    Sum regularized infill criterion for bi-objective acquisition functions.
    """
    # maximize the acquisition function
    problem = AcquisitionFunctionProblem(
        acquisition_function, model, n_var=n_var, bound=bound, minimize=False
    )
    algorithm = PSO()
    res = minimize(problem, algorithm, termination=get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    logger.info(f"acquisition function approximate maximizer:\n X = {res.X}\n F = {res.F}")
    # compute the regularization variable gamma
    X_max = res.X
    alpha = res.F
    xi = np.sum(model.evaluate(X_max.reshape(1, -1)))
    gamma = 100 * xi / alpha if alpha > 0 else 1
    # maximize the regularized infill criterion
    reg_problem = RegCritProblem(acquisition_function, model, n_var, bound, gamma)
    algorithm = PSO()
    reg_res = minimize(reg_problem, algorithm, termination=get_termination("n_gen", n_gen),
                       seed=seed, verbose=False)
    logger.info(f"regularized infill best solution:\n X = {res.X}\n F = {res.F}")
    return reg_res.X


def compute_pareto(J1: np.ndarray, J2: np.ndarray) -> np.ndarray:
    """
    Returns the current fitness pareto front.
    see implementation from the stackoverflow thread below:
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python/40239615#40239615 # noqa
    """
    costs = np.column_stack((J1, J2))
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    costs = costs[is_efficient]
    sorted_idx = np.argsort(costs, axis=0)[:, 0]
    return costs[sorted_idx]
