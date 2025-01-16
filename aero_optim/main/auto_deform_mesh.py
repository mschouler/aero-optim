import argparse
import logging
import sys
import os

from aero_optim.utils import check_config, get_custom_class
from aero_optim.mesh.cascade_mesh import CascadeMeshMusicaa

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    """
    This program automatically deforms a baseline mesh with musicaa as part of a shape
    optimization routine.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-f", "--file", type=str,
                        help="input dat file: --file=/path/to/file.dat", default="")
    parser.add_argument("-o", "--outdir", type=str, help="mesh output directory", default="")
    args = parser.parse_args()

    config, custom_file, study_type = check_config(args.config, outdir=args.outdir)

    MeshClass = get_custom_class(custom_file, "CustomMesh") if custom_file else None
    if not MeshClass:
        if study_type == "cascade":
            MeshClass = CascadeMeshMusicaa
        else:
            raise Exception(f"ERROR -- incorrect study_type <{study_type}>")
    musicaa_mesh = MeshClass(config)

    if musicaa_mesh:
        musicaa_mesh.write_deformed_mesh_edges()
        musicaa_mesh.deform_mesh()


if __name__ == "__main__":
    main()
