import numpy as np
import os

from src.FFD.ffd import FFD_2D

input: str = "input/naca12.dat"
Delta: np.ndarray = np.array([1., 1., 1., 1.])
nc: int = 2


def get_ffd2d(input: str, nc: int) -> FFD_2D:
    return FFD_2D(input, nc)


def test_to_lat():
    assert os.path.isfile(input)
    ffd = get_ffd2d(input, nc)
    assert np.sum(np.abs(ffd.from_lat(ffd.to_lat(ffd.pts)) - ffd.pts[:, :2])) < 1e-10


def test_dPij():
    ffd = get_ffd2d(input, nc)
    assert np.array_equal(ffd.dPij(0, 0, Delta), np.array([0, Delta[0]]))


def test_pad_Delta():
    ffd = get_ffd2d(input, nc)
    assert np.array_equal(ffd.pad_Delta(Delta), np.array([0., 1., 1., 0., 0., 1., 1., 0.]))


def test_apply_ffd():
    ffd = get_ffd2d(input, nc)
    null_Delta = np.zeros(4)
    deformed_profile = ffd.apply_ffd(null_Delta)
    assert np.sum(np.abs(ffd.pts[:, :2] - deformed_profile)) < 1e-10
