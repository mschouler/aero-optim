import inspyred
import numpy as np
import operator
import os
import pytest

from aero_optim.optim.inspyred_optimizer import InspyredWolfOptimizer
from aero_optim.utils import check_file, check_config

sim_config_path: str = "tests/extras/test_optimizer_config.json"
mesh_file: str = "empty_file.mesh"
mesh_file_path: str = "tests/extras/" + mesh_file
executable_path: str = os.path.join(os.getcwd(), "tests", "extras", "dummy_wolf.py")


@pytest.fixture(scope='session')
def opt() -> InspyredWolfOptimizer:
    check_file(sim_config_path)
    config, _, _ = check_config(sim_config_path)
    opt = InspyredWolfOptimizer(config)
    idx = opt.simulator.exec_cmd.index("@path")
    opt.simulator.exec_cmd[idx] = executable_path
    return opt


def dummy_deform(Delta: np.ndarray, gid: int, cid: int) -> tuple[str, np.ndarray]:
    """dummy deform."""
    return "", np.array([])


def dummy_mesh(ffdfile: str) -> str:
    """dummy_mesh."""
    return mesh_file_path


def test_ins_optimizer(opt: InspyredWolfOptimizer):
    # simplify deform and mesh methods
    setattr(opt, "deform", dummy_deform)
    setattr(opt, "mesh", dummy_mesh)
    ea = inspyred.swarm.PSO(opt.prng)
    ea.terminator = inspyred.ec.terminators.generation_termination
    final_pop = ea.evolve(generator=opt.generator._ins_generator,
                          evaluator=opt._evaluate,
                          pop_size=opt.doe_size,
                          max_generations=opt.max_generations,
                          bounder=inspyred.ec.Bounder(*opt.bound),
                          maximize=opt.maximize)
    index, opt_J = (
        max(enumerate(opt.J), key=operator.itemgetter(1)) if opt.maximize else
        min(enumerate(opt.J), key=operator.itemgetter(1))
    )
    gid, cid = (index // opt.doe_size, index % opt.doe_size)
    # CD, CL = [0.1 + (gid**cid + cid + gid) / 100. + cid / 200., 0.3 + (gid + cid) / 100.]
    # CL < 0.31 => fitness = CD + 1
    print(opt.J)
    expected_J = [1.11, 0.115, 0.13, 0.12, 0.135, 0.15]
    assert opt.max_generations == opt.gen_ctr - 1
    assert np.sum([abs(i - j) for i, j in zip(opt.J, expected_J)]) < 1e-10
    assert min([c.fitness for c in final_pop]) == min(opt.J[-opt.doe_size:])
    assert (gid, cid) == (0, 1)
