import argparse

from src.utils import check_file, check_config
from src.naca_base_mesh import NACABaseMesh
from src.naca_block_mesh import NACABlockMesh


if __name__ == "__main__":
    """
    This program automatically generates a simple mesh with gmsh as part of a shape
    optimization routine. Depending on the configuration settings, the mesh is designed for:
    * an external airfoil shape optimization in supersonic conditions [airfoil]
        ** with a basic mesh [base]
        ** with a block mesh [block]
    * a compressor blade shape optimization in transonic conditions [compressor] (soon)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-f", "--file", type=str,
                        help="input dat file: --file=/path/to/file.dat", default=None)

    args = parser.parse_args()
    check_file(args.config)
    config, study_type = check_config(args.config, args.file)

    gmsh_mesh: NACABaseMesh | NACABlockMesh | None = None
    if study_type == "base":
        gmsh_mesh = NACABaseMesh(config)
    elif study_type == "block":
        gmsh_mesh = NACABlockMesh(config)

    if gmsh_mesh:
        gmsh_mesh.build_mesh()
