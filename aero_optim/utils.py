import importlib.util
import glob
import json
import logging
import os.path
import shutil
import signal
import subprocess
import argparse

from types import FrameType

STUDY_TYPE = ["naca_base", "naca_block", "cascade"]
FFD_TYPE = ["ffd_2d", "ffd_pod_2d"]
logger = logging.getLogger(__name__)


def from_dat(file: str, header_len: int = 2, scale: float = 1) -> list[list[float]]:
    """
    Returns the cleaned list of data points.
    >> dat_fil: path to input_geometry.dat.
    >> pts:  the geometry coordinates in the original referential.
       pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
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
        custom_file: str = "", outdir: str = "",
        optim: bool = False, gmsh: bool = False, sim: bool = False) -> tuple[dict, str, str]:
    """
    Ensures the presence of all required entries in config,
    then returns config (dict), custom_file (str) and study type (str).
    """
    # check for config and open it
    check_file(config)
    with open(config) as jfile:
        config_dict = json.load(jfile)
    print("AERO-Optim: general check config..")

    # supersed outdir if given
    if outdir:
        config_dict["study"]["outdir"] = outdir

    # look for upper level categories
    if "study" not in config_dict:
        raise Exception(f"ERROR -- no <study>  upper entry in {config}")
    if optim and "optim" not in config_dict:
        raise Exception(f"ERROR -- no <optim>  upper entry in {config}")
    if (optim or gmsh) and "gmsh" not in config_dict:
        raise Exception(f"ERROR -- no <gmsh>  upper entry in {config}")
    if (optim or sim) and "simulator" not in config_dict:
        raise Exception(f"ERROR -- no <simulator>  upper entry in {config}")

    # look for mandatory entries
    if optim and "ffd_type" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <ffd_type> entry in {config}[study]")
    if (optim or gmsh) and "study_type" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <study_type> entry in {config}[study]")
    if (optim or gmsh) and "file" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <file>  entry in {config}[study]")
    if "outdir" not in config_dict["study"]:
        raise Exception(f"ERROR -- no <outdir>  entry in {config}[study]")
    if optim:
        check_dir(config_dict["study"]["outdir"])
        cp_filelist([config], [config_dict["study"]["outdir"]])

    # check path and study_type correctness
    if (optim or gmsh) and not os.path.isfile(config_dict["study"]["file"]):
        raise Exception(f"ERROR -- <{config_dict['study']['file']}> could not be found")

    # supersede custom_file entry
    if custom_file:
        config_dict["study"]["custom_file"] = custom_file

    return (
        config_dict, config_dict["study"].get("custom_file", ""), config_dict["study"]["study_type"]
    )


def check_parser(args: argparse.ArgumentParser):
    """
    Ensures the presence of all required entries in the command line (for auto_*.py).
    """
    # if the 2D profile is not available directly, need specific arguments
    if args.profile_or_mesh_files != 'profile':
        # indicate the blocks containing wall coordinates (connected the profile)
        if not args.wall_bl:
            raise Exception("ERROR -- must include [wall_bl] entry (see --help)")
        # if periodic blocks must be translated to reconstruct 2D profile
        # check if pitch is provided
        if args.periodic_bl and not args.pitch:
            raise Exception("ERROR -- if periodic_bl, must include blade pitch value (see --help)")


def check_file(filename: str):
    """
    Makes sure an existing file was given.
    """
    if not os.path.isfile(filename):
        raise Exception(f"ERROR -- <{filename}> could not be found")


def check_dir(dirname: str):
    """
    Makes sure the directory exists and create one if not.
    """
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logger.info(f"created {dirname} repository")


def configure_logger(logger: logging.Logger, log_filename: str, log_level: int = logging.INFO):
    """
    Configures logger.
    """
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_filename, mode="w")
    formatter = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_log_level_from_verbosity(verbosity: int) -> int:
    """
    Logger level setting taken from:
    https://gitlab.inria.fr/melissa/melissa/-/blob/develop/melissa/utility/logger.py
    """
    if verbosity >= 3:
        return logging.DEBUG
    elif verbosity == 2:
        return logging.INFO
    elif verbosity == 1:
        return logging.WARNING
    elif verbosity == 0:
        return logging.ERROR
    else:
        return logging.DEBUG


def set_logger(logger: logging.Logger, outdir: str, log_name: str, verb: int = 3) -> logging.Logger:
    """
    Returns the set logger.
    """
    log_level = get_log_level_from_verbosity(verb)
    configure_logger(logger, os.path.join(outdir, log_name), log_level)
    return logger


def handle_signal(signo: int, frame: FrameType | None):
    """
    Raises exception in case of interruption signal.
    """
    signame = signal.Signals(signo).name
    logger.info(f"clean handling of {signame} signal")
    raise Exception("Program interruption")


def catch_signal():
    """
    Makes sure interruption signals are catched.
    """
    signals = [signal.SIGINT, signal.SIGPIPE, signal.SIGTERM]
    for s in signals:
        signal.signal(s, handle_signal)


def get_custom_class(filename: str, module_name: str):
    """
    Returns a customized object (evolution, optimizer, simulator or mesh).
    """
    try:
        spec_class = importlib.util.spec_from_file_location(module_name, filename)
        if spec_class and spec_class.loader:
            custom_class = importlib.util.module_from_spec(spec_class)
            spec_class.loader.exec_module(custom_class)
            MyClass = getattr(custom_class, module_name)
            logger.info(f"successfully recovered {module_name}")
            return MyClass
    except Exception:
        logger.warning(f"could not find {module_name} in {filename}")
        return None


def run(cmd: list[str], output: str, timeout: float | None = None):
    """
    Wrapper around subprocess.run that executes cmd and redirects stdout/stderr to output.
    """
    with open(output, "wb") as out:
        subprocess.run(cmd, stdout=out, stderr=out, check=True, timeout=timeout)


def sed_in_file(fname: str, sim_args: dict):
    """
    Mimics the bash sed function.
    It updates the file fname by altering keywords according to sim_args which has the form:
    sim_args = {1st key: {"inplace": bool, "param": list[str]}, 2nd key: ...}.
    If "inplace" is set to True, the key is changed in place by the unique value in the list.
    If not, the key is unchanged and its following lines are replaced with the values in the list.
    """
    file_content = open(fname, "r").read().splitlines()
    for key, value in sim_args.items():
        idx = file_content.index(key)
        if value["inplace"]:
            file_content[idx] = value['param'][0]
        else:
            for ii, param in enumerate(value['param']):
                file_content[idx + 1 + ii] = param
    with open(fname, 'w') as ftw:
        ftw.write("\n".join(file_content))


def replace_in_file(fname: str, sim_args: dict):
    """
    Updates the file fname by replacing specific strings with others.
    """
    with open(fname, 'r') as file:
        filedata = file.read()
    # Replace the target string
    for key, value in sim_args.items():
        filedata = filedata.replace(key, value)
    # Write the file out again
    with open(fname, 'w') as file:
        file.write(filedata)


def rm_filelist(deletion_list: list[str]):
    """
    Wrapper around os.remove that deletes all files specified in deletion_list.
    """
    [os.remove(f) for f_pattern in deletion_list for f in glob.glob(f_pattern)]  # type: ignore


def cp_filelist(in_files: list[str], out_files: list[str], move: bool = False):
    """
    Wrapper around shutil.copy that mimics bash cp command.
    It copies all files specified in in_files to out_files if move is set to False.
    If move is set to True, the behavior is changed to the bash mv command.
    """
    for in_f, out_f in zip(in_files, out_files):
        try:
            shutil.copy(in_f, out_f) if not move else shutil.move(in_f, out_f)
        except FileNotFoundError:
            print(f"WARNING -- {in_f} not found")
        except shutil.SameFileError:
            print(f"WARNING -- {in_f} same file as {out_f}")


def mv_filelist(*args):
    """
    Wrapper around shutil.move that mimics bash mv command.
    """
    return cp_filelist(*args, move=True)


def ln_filelist(in_files: list[str], out_files: list[str]):
    """
    Wrapper around os.symlink that mimics bash ln -s command.
    If the symbolic link already exists, the behavior is changed to ln -sf.
    """
    for in_f, out_f in zip(in_files, out_files):
        try:
            os.symlink(in_f, out_f)
        except FileExistsError as e:
            print(f"WARNING -- {e}, symlink will be forced")
            os.symlink(in_f, "tmplink")
            os.rename("tmplink", out_f)
