import importlib.util
import json
import logging
import os.path
import shutil
import signal

from types import FrameType

STUDY_TYPE = ["base", "block", "cascade", "debug"]
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
        optim: bool = False, gmsh: bool = False, sim: bool = False) -> tuple[dict, str]:
    """
    Ensures the presence of all required entries in config, then return config and study type.
    """
    # check for config and open it
    check_file(config)
    with open(config) as jfile:
        config_dict = json.load(jfile)
    print("general check config..")

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
    if optim:
        check_dir(config_dict["study"]["outdir"])
        shutil.copy(config, config_dict["study"]["outdir"])

    # check path and study_type correctness
    if (optim or gmsh) and not os.path.isfile(config_dict["study"]["file"]):
        raise Exception(f"ERROR -- <{config_dict['study']['file']}> could not be found")

    return config_dict, config_dict["study"]["study_type"]


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
    spec_class = importlib.util.spec_from_file_location(module_name, filename)
    if spec_class and spec_class.loader:
        custom_class = importlib.util.module_from_spec(spec_class)
        spec_class.loader.exec_module(custom_class)
        MyClass = getattr(custom_class, module_name)
        logger.info(f"successfully recovered {module_name}")
        return MyClass
    else:
        raise Exception(f"ERROR -- could not find {module_name} in {filename}")
