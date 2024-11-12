import math
import numpy as np


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


# analytical functions
# 1D
def meng_f1d_lf(x: np.ndarray, A: float = 0.5, B: float = 10., C: float = -5):
    """
    1D low fidelity analytical function from Meng 2020:
    https://doi.org/10.1016/j.jcp.2019.109020

    Note: x in [0; 1]
    """
    return A * (6 * x - 2)**2 * np.sin(12 * x - 4) + B * (x - 0.5) + C


def meng_f1d_hf(x: np.ndarray):
    """
    1D high fidelity analytical function from Meng 2020:
    https://doi.org/10.1016/j.jcp.2019.109020

    Note: x in [0; 1]
    """
    return (6 * x - 2)**2 * np.sin(12 * x - 4)


def brevault_f1d_lf(x: np.ndarray) -> float:
    """
    1D low fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

    Note: x in [0; 1]
    """
    return np.sin(4 * math.pi * x)


def brevault_f1d_hf(x: np.ndarray, a: int) -> float:
    """
    1D high fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

    Note: x in [0; 1]
    """
    return (x / 2. - math.sqrt(2)) * np.sin(4 * math.pi * x + a * math.pi)**a


# 2D
def brevault_f2d_hf(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    2D high fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    return (x2**2 - x1)**2 + (x1 - 1)**2


def brevault_f2d_lf(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    2D low fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    return 0.9 * x2**4 + 2.2 * x1**2 - 1.8 * x1 * x2**2 + 0.5


# nD
def brevault_fnd_hf(x: np.ndarray) -> np.ndarray:
    """
    nD high fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

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
    nD low fidelity analytical function from Brevault 2020:
    https://doi.org/10.1016/j.ast.2020.106339

    Note: x_i in [-3; 3], i in [1, ..., d].
    """
    x_t = x.transpose()  # x_t has shape (dim, n_lf)
    res: np.ndarray = np.zeros(x_t.shape[-1])
    for x_ip1, x_i in zip(x_t[1:], x_t[:-1]):
        res += 0.9 * x_ip1**4 + 2.2 * x_i**2 - 1.8 * x_i * x_ip1**2 + 0.5
    res = np.expand_dims(res, axis=1)
    return res
