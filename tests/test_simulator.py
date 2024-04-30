import os
import pytest
import time

from src.simulator import WolfSimulator
from src.utils import check_file, check_config

sim_config_path: str = "tests/test_simulator_config.json"
mesh_file: str = "empty_file.mesh"
mesh_file_path: str = "tests/" + mesh_file
executable_path: str = os.path.join(os.getcwd(), "tests", "dummy_wolf.py")


@pytest.fixture(scope='session')  # one server to rule'em all
def wolf() -> WolfSimulator:
    check_file(sim_config_path)
    config, _ = check_config(sim_config_path)
    wolf = WolfSimulator(config)
    idx = wolf.exec_cmd.index("@path")
    wolf.exec_cmd[idx] = executable_path
    return wolf


def test_execute_sim(wolf: WolfSimulator):
    wolf.execute_sim(meshfile=mesh_file_path)
    assert wolf.sim_pro[0][0] == {'cid': 0, 'gid': 0}
    assert wolf.sim_pro[0][-1].args == ["python3", executable_path, "-in", mesh_file]
    assert wolf.df_dict[0] == {}


def test_monitor_sim_progress(wolf: WolfSimulator):
    """this also tests post_process"""
    assert wolf.monitor_sim_progress() == 1
    time.sleep(1)
    assert wolf.monitor_sim_progress() == 0
    assert wolf.df_dict[0][0].columns.values.tolist() == ["ResTot", "CD", "CL", "ResCD", "ResCL"]
