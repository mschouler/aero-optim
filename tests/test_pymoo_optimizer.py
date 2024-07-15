import numpy as np
import os
import pytest
from typing import cast

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from aero_optim.optim.pymoo_optimizer import PymooWolfOptimizer
from aero_optim.utils import check_file, check_config

sim_config_path: str = "tests/extras/test_optimizer_config.json"
mesh_file: str = "empty_file.mesh"
mesh_file_path: str = "tests/extras/" + mesh_file
executable_path: str = os.path.join(os.getcwd(), "tests", "extras", "dummy_wolf.py")


@pytest.fixture(scope='session')
def opt() -> PymooWolfOptimizer:
    check_file(sim_config_path)
    config, _, _ = check_config(sim_config_path)
    # as opposed to inspyred, pymoo considers the initial generation as the first one
    # test consistency is enforced by incrementing max_generations
    config["optim"]["max_generations"] = config["optim"]["max_generations"] + 1
    opt = PymooWolfOptimizer(config)
    idx = opt.simulator.exec_cmd.index("@path")
    opt.simulator.exec_cmd[idx] = executable_path
    return opt


def dummy_deform(Delta: np.ndarray, gid: int, cid: int) -> tuple[str, np.ndarray]:
    """dummy deform."""
    return "", np.array([])


def dummy_mesh(ffdfile: str) -> str:
    """dummy_mesh."""
    return mesh_file_path


def dummy_observe(pop_fitness: np.ndarray):
    """dummy _observe."""
    return


def test_pymoo_optimizer(opt: PymooWolfOptimizer):
    # simplify deform and mesh methods
    setattr(opt, "deform", dummy_deform)
    setattr(opt, "mesh", dummy_mesh)
    setattr(opt, "_observe", dummy_observe)
    ea = PSO(pop_size=opt.doe_size, sampling=opt.generator._pymoo_generator())
    res = minimize(problem=opt,
                   algorithm=ea,
                   termination=get_termination("n_gen", opt.max_generations),
                   seed=opt.seed,
                   verbose=True)
    index, opt_J = min(enumerate(opt.J), key=lambda x: abs(res.F - x[1]))
    gid, cid = (index // opt.doe_size, index % opt.doe_size)
    # CD, CL = [0.1 + (gid**cid + cid + gid) / 100. + cid / 200., 0.3 + (gid + cid) / 100.]
    # CL < 0.31 => g0, c0 becomes a bad candidate
    expected_J = [0.11, 0.115, 0.13, 0.12, 0.135, 0.15]
    assert opt.max_generations == opt.gen_ctr
    assert np.sum([abs(i - j) for i, j in zip(cast(list[float], opt.J), expected_J)]) < 1e-10
    assert type(opt.J[1]) is float and type(opt_J) is float
    assert abs(opt.J[1] - opt_J) < 1e-6
    assert (gid, cid) == (0, 1)
