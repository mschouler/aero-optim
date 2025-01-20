import argparse
import logging
import sys

from aero_optim.utils import check_config, get_custom_class

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    """
    This program automatically deforms a baseline mesh with musicaa as part of a shape
    optimization routine.
    Note: a CustomMesh class MUST be provided, since mesh deformations are not default
    to the aero_optim module.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    args = parser.parse_args()

    config, custom_file, _ = check_config(args.config)

    # instantiate custom mesh class
    MeshClass = get_custom_class(custom_file, "CustomMesh") if custom_file else None
    if not MeshClass:
        raise Exception("ERROR -- either provide a 2D profile, or a valid CustomMesh class.")
    mesh = MeshClass(config)

    # build mesh
    mesh.build_mesh()


if __name__ == "__main__":
    main()
