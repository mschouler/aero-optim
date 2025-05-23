import argparse
import logging
import sys

from aero_optim.utils import check_config, get_custom_class
from aero_optim.mesh.mesh import MeshMusicaa
from aero_optim.mesh.cascade_mesh import CascadeMesh
from aero_optim.mesh.naca_base_mesh import NACABaseMesh
from aero_optim.mesh.naca_block_mesh import NACABlockMesh

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
    parser.add_argument("-o", "--outdir", type=str, help="mesh output directory", default="")
    args = parser.parse_args()

    config, custom_file, study_type = check_config(args.config, outdir=args.outdir, gmsh=True)

    MeshClass = get_custom_class(custom_file, "CustomMesh") if custom_file else None
    if not MeshClass:
        if study_type == "naca_base":
            MeshClass = NACABaseMesh
        elif study_type == "naca_block":
            MeshClass = NACABlockMesh
        elif study_type == "cascade":
            MeshClass = CascadeMesh
        elif study_type == "musicaa":
            MeshClass = MeshMusicaa
        else:
            raise Exception(f"ERROR -- incorrect study_type <{study_type}>")
    gmsh_mesh = MeshClass(config, args.file)

    if gmsh_mesh:
        gmsh_mesh.build_mesh()
        _ = gmsh_mesh.write_mesh()


if __name__ == "__main__":
    main()
