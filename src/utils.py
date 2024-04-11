import json
import os.path


def from_dat(file: str, header_len: int = 2, scale: float = 1) -> list[list[float]]:
    """
    Return the cleaned list of data points.
    >> pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
       where N is the number of points describing the geometry
       and (z0, ..., zN) are null or identical.
    Note:
        When a geometry is closed, the .dat file may contain a redundancy i.e. the last
        point is also the first one in the list. This can become problematic for the
        compressor blade case but has no effect in other cases. Such duplicates are hence removed.
    """
    dat_file = [line.strip() for line in open(file, "r").read().splitlines()]
    pts = [list(map(float, line.split(" "))) for line in dat_file[header_len:]]
    pts = pts[:-1] if pts[0] == pts[-1] else pts
    pts = [[p[0], p[1], 0.] for p in pts] if len(pts[0]) == 2 else pts
    return pts if scale == 1 else [[coord * scale for coord in p] for p in pts]


def check_config(
        config: str,
        optim: bool = False, gmsh: bool = False, sim: bool = False) -> tuple[dict, str]:
    """
    Ensure the presence of all required entries in config, then return config and study type.
    """
    with open(config) as jfile:
        config_dict = json.load(jfile)

    # look for upper level categories
    if "study" not in config_dict:
        raise Exception(f"ERROR -- no <study>  upper entry in {config}")
    if optim and "optim" not in config_dict:
        raise Exception(f"ERROR -- no <optim>  upper entry in {config}")
    if (optim or gmsh) and "gmsh" not in config_dict:
        raise Exception(f"ERROR -- no <mesh>  upper entry in {config}")
    if (optim or sim) and "simulator" not in config_dict:
        raise Exception(f"ERROR -- no <simulator>  upper entry in {config}")

    # look for mandatory entries
    if (optim or gmsh) and "study_type" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <study_type> entry in {config}[study]")
    if (optim or gmsh) and "file" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <file>  entry in {config}[study]")
    if "outdir" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <outdir>  entry in {config}[study]")

    # check path correctness
    if (optim or gmsh) and not os.path.isfile(config_dict["study"]["file"]):
        raise Exception(f"ERROR -- <{config_dict['study']['file']}> could not be found")
    if (optim or gmsh) and config_dict["study"]["study_type"] not in ["base", "block"]:
        raise Exception(f"ERROR -- wrong <study_type> specification in {config}[study]")

    return config_dict, config_dict["study"]["study_type"]


def check_file(filename: str):
    """
    Make sure an existing file was given.
    """
    if not os.path.isfile(filename):
        raise Exception(f"ERROR -- <{filename}> could not be found")


def check_dir(dirname: str):
    """
    Make sure the directory exists and create one if not.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        print(f">> created {dirname} repository")
