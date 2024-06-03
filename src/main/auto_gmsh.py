import argparse
import logging
import sys

from src.utils import check_config
from src.mesh.cascade_mesh import CascadeMesh
from src.mesh.naca_base_mesh import NACABaseMesh
from src.mesh.naca_block_mesh import NACABlockMesh

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
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
                        help="input dat file: --file=/path/to/file.dat", default="")
    args = parser.parse_args()

    config, study_type = check_config(args.config, gmsh=True)

    gmsh_mesh = None
    if study_type == "base":
        gmsh_mesh = NACABaseMesh(config, args.file)
    elif study_type == "block":
        gmsh_mesh = NACABlockMesh(config, args.file)
    elif study_type == "cascade":
        gmsh_mesh = CascadeMesh(config, args.file)
    else:
        raise Exception(f"ERROR -- incorrect study_type <{study_type}>")

    if gmsh_mesh:
        gmsh_mesh.build_mesh()
        _ = gmsh_mesh.write_mesh()


if __name__ == "__main__":
    main()
