import os

from aero_optim.mesh.naca_base_mesh import NACABaseMesh
from aero_optim.mesh.naca_block_mesh import NACABlockMesh
from aero_optim.utils import check_config, check_file

base_config_path: str = "tests/extras/test_base_config.json"
base_files: list[str] = ["naca_base.mesh", "naca_base.geo_unrolled", "naca_base.log"]
block_config_path: str = "tests/extras/test_block_config.json"
block_files: list[str] = ["naca_block.mesh", "naca_block.geo_unrolled", "naca_block.log"]


def get_NACABaseMesh(config_path: str) -> NACABaseMesh:
    check_file(config_path)
    config, _, _ = check_config(config_path, gmsh=True)
    return NACABaseMesh(config)


def get_NACABlockMesh(config_path: str) -> NACABlockMesh:
    check_file(config_path)
    config, _, _ = check_config(config_path, gmsh=True)
    return NACABlockMesh(config)


def test_base_mesh(tmpdir):
    base_mesh = get_NACABaseMesh(base_config_path)
    base_mesh.build_mesh()
    _ = base_mesh.write_mesh(tmpdir)
    for file in base_files:
        assert os.path.isfile(os.path.join(tmpdir, file))


def test_block_mesh(tmpdir):
    block_mesh = get_NACABlockMesh(block_config_path)
    block_mesh.build_mesh()
    _ = block_mesh.write_mesh(tmpdir)
    for file in block_files:
        assert os.path.isfile(os.path.join(tmpdir, file))
