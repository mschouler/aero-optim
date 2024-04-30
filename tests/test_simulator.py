from src.simulator import WolfSimulator
from src.utils import check_file, check_config

sim_config_path: str = "tests/test_simulator_config.json"
mesh_file: str = "empty_file.mesh"
mesh_file_path: str = "tests/" + mesh_file


def get_wolfsimulator(config_path: str) -> WolfSimulator:
    check_file(config_path)
    config, _ = check_config(config_path)
    return WolfSimulator(config)


def test_execute_sim():
    gid: int = 0
    wolf: WolfSimulator = get_wolfsimulator(sim_config_path)
    wolf.execute_sim(meshfile=mesh_file_path)
    assert wolf.sim_pro[0][0] == {'cid': 0, 'gid': 0}
    assert wolf.sim_pro[0][-1].args == ["python3", "tests/dummy_wolf.py", "-in", mesh_file]
    assert wolf.df_dict[gid] == {}
