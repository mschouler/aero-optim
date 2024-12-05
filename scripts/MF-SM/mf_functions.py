import math
import numpy as np

from typing import Callable

from pymoo.core.problem import Problem


# metrics
def get_R2(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the coefficient of determination R2.
    Note: the higher the better.
    """
    return 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)


def get_RMSE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the root mean square error RMSE.
    Note: the lower the better.
    """
    return np.sqrt(np.sum((y_test - y_pred)**2) / len(y_test))


# 1D analytical functions from Meng 2020:
# https://doi.org/10.1016/j.jcp.2019.109020
def meng_f1d_lf(x: np.ndarray, A: float = 0.5, B: float = 10., C: float = -5):
    """
    1D low fidelity analytical function.
    Note: x in [0; 1]
    """
    return A * (6 * x - 2)**2 * np.sin(12 * x - 4) + B * (x - 0.5) + C


def meng_f1d_hf(x: np.ndarray):
    """
    1D high fidelity analytical function.
    Note: x in [0; 1]
    """
    return (6 * x - 2)**2 * np.sin(12 * x - 4)


# 1D analytical functions from Brevault 2020:
# https://doi.org/10.1016/j.ast.2020.106339
def brevault_f1d_lf(x: np.ndarray) -> float:
    """
    1D low fidelity analytical function.
    Note: x in [0; 1]
    """
    return np.sin(4 * math.pi * x)


def brevault_f1d_hf(x: np.ndarray, a: int) -> float:
    """
    1D high fidelity analytical function.
    Note: x in [0; 1]
    """
    return (x / 2. - math.sqrt(2)) * np.sin(4 * math.pi * x + a * math.pi)**a


# 2D analytical functions from Brevault 2020:
# https://doi.org/10.1016/j.ast.2020.106339
def brevault_f2d_hf(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    2D high fidelity analytical function.
    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    return (x2**2 - x1)**2 + (x1 - 1)**2


def brevault_f2d_lf(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    2D low fidelity analytical function.
    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    return 0.9 * x2**4 + 2.2 * x1**2 - 1.8 * x1 * x2**2 + 0.5


# nD analytical functions from Brevault 2020:
# https://doi.org/10.1016/j.ast.2020.106339
def brevault_fnd_hf(x: np.ndarray) -> np.ndarray:
    """
    nD high fidelity analytical function.
    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    x_t = x.transpose()  # x_t has shape (dim, n_lf)
    res: np.ndarray = np.zeros(x_t.shape[-1])
    for x_ip1, x_i in zip(x_t[1:], x_t[:-1]):
        res += (x_ip1**2 - x_i)**2 + (x_i - 1)**2
    res = np.expand_dims(res, axis=1)
    return res


def brevault_fnd_lf(x: np.ndarray) -> np.ndarray:
    """
    nD low fidelity analytical function.
    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    x_t = x.transpose()  # x_t has shape (dim, n_lf)
    res: np.ndarray = np.zeros(x_t.shape[-1])
    for x_ip1, x_i in zip(x_t[1:], x_t[:-1]):
        res += 0.9 * x_ip1**4 + 2.2 * x_i**2 - 1.8 * x_i * x_ip1**2 + 0.5
    res = np.expand_dims(res, axis=1)
    return res


# nD multi-fidelity analytical optimization problems
# from Charayron 2023:
# https://doi.org/10.1016/j.ast.2023.108673

# ZDT1
def zdt1_u(x: np.ndarray) -> np.ndarray:
    n_var = x.shape[-1]
    return 1 + 9.0 / (n_var - 1) * np.sum(x[:, 1:], axis=1)


def zdt1_v(x: np.ndarray) -> np.ndarray:
    return 1 - np.sqrt(x[:, 0] / zdt1_u(x))


def zdt1_hf(x: np.ndarray) -> np.ndarray:
    """
    ZDT1 high-fidelity function.
    """
    f1 = x[:, 0]
    f2 = zdt1_u(x) * zdt1_v(x)
    return np.column_stack([f1, f2])


def zdt1_lf(x) -> np.ndarray:
    """
    ZDT1 low-fidelity function.
    """
    f1 = 0.9 * x[:, 0] + 0.1
    f2 = (0.8 * zdt1_u(x) - 0.2) * (1.2 * zdt1_v(x) + 0.2)
    return np.column_stack([f1, f2])


# ZDT2
def zdt2_u(x: np.ndarray) -> np.ndarray:
    return zdt1_u(x)


def zdt2_v(x: np.ndarray) -> np.ndarray:
    return 1 - (x[:, 0] / zdt1_u(x))**2


def zdt2_hf(x) -> np.ndarray:
    """
    ZDT2 high-fidelity function.
    """
    f1 = x[:, 0]
    f2 = zdt2_u(x) * zdt2_v(x)
    return np.column_stack([f1, f2])


def zdt2_lf(x) -> np.ndarray:
    """
    ZDT2 low-fidelity function.
    """
    f1 = 0.8 * x[:, 0] + 0.2
    f2 = (0.9 * zdt2_u(x) + 0.2) * (1.1 * zdt2_v(x) - 0.2)
    return np.column_stack([f1, f2])


class SimpleProblem(Problem):
    """
    Pymoo simple bi-objective optimization problem.

    Note: function can either be an analytical function and return a numpy array
          or it can be the output of a MultiObjectiveModel evaluation and be a list
          of numpy arrays.
    """
    def __init__(self, function: Callable, dim: int, bound: list[float]):
        Problem.__init__(
            self, n_var=dim, n_obj=2, n_ieq_constr=0, xl=bound[0], xu=bound[1]
        )
        self.function = function
        self.gen_ctr = 0

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        eval = self.function(X)
        out["F"] = eval if isinstance(eval, np.ndarray) else np.column_stack(eval)
        if self.gen_ctr == 0:
            self.candidates = X
            self.fitnesses = out["F"]
        else:
            self.candidates = np.vstack((self.candidates, X))
            self.fitnesses = np.vstack((self.fitnesses, out["F"]))
        self.gen_ctr += 1
