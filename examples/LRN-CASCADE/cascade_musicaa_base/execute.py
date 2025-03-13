import logging
import subprocess
import functools
import numpy as np
import os
import shutil
import sys
import re
import pandas as pd
import json
from typing import Callable

from aero_optim.mesh.mesh import get_block_info
from custom_cascade_MUSICAA import get_time_info, get_niter_ftt
from aero_optim.utils import (cp_filelist, round_number, rm_filelist,
                              read_next_line_in_file, custom_input)

FAILURE: int = 1
SUCCESS: int = 0

MUSICAA: str = "mpiexec -n 51 /home/matar/bin/musicaa"

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)


def execute_steady(config: dict):
    """
    **Executes** a Reynolds-Averaged Navier-Stokes simulation with MUSICAA.
    """

    # execute computation
    config.update({"is_stats": False})
    proc = execute(MUSICAA)
    monitor_sim_progress(proc, config, "steady")

    # gather statistics
    config.update({"is_stats": True})
    pre_process_stats(config, "steady")
    proc = execute(MUSICAA)
    monitor_sim_progress(proc, config, "steady")


def execute_unsteady(config: dict):
    """
    **Executes** a Large Eddy Simulation with MUSICAA.
    """

    # initialize LES with a fully developed 2D field
    config.update({"is_stats": False})
    pre_process_init_unsteady("2D")
    proc = execute(MUSICAA)
    monitor_sim_progress(proc, config, "unsteady", "init_2D")

    # carry on with 3D transient
    pre_process_init_unsteady("3D")
    proc = execute(MUSICAA)
    monitor_sim_progress(proc, config, "unsteady", "init_3D")

    # gather statistics
    config.update({"is_stats": True})
    pre_process_stats(config, "unsteady")
    proc = execute(MUSICAA)
    monitor_sim_progress(proc, config, "unsteady")


def execute(exec_cmd: str):
    """
    **Pre-processes** and **Executes** a simulation with musicaa.
    """
    # submit computation
    with open("musicaa.out", "wb") as out:
        with open("musicaa.err", "wb") as err:
            proc = subprocess.Popen(exec_cmd.split(),
                                    env=os.environ,
                                    stdin=subprocess.DEVNULL,
                                    stdout=out,
                                    stderr=err,
                                    universal_newlines=True)
    return proc


def monitor_sim_progress(proc: subprocess.Popen,
                         config: dict,
                         computation_type: str,
                         unsteady_step: str = ""):
    """
    **Monitors** a simulation.
    """
    # get simulation arguments
    restart = config["simulator"].get("restart", 5)
    divide_CFL_by = config["simulator"].get("divide_CFL_by", 1.2)
    if unsteady_step == "init_2D":
        max_niter = config["simulator"].get("max_niter_init_2D", 200000)
    elif unsteady_step == "init_3D":
        max_niter = config["simulator"].get("max_niter_init_3D", 200000)
    if config["is_stats"]:
        max_niter = config["simulator"].get("max_niter_stats", 200000)
    else:
        config.update({"max_niter_steady_reached": False})
        max_niter = config["simulator"].get("max_niter_steady", 100000)

    current_restart = 0
    while True:
        returncode = proc.poll()
        # computation still running
        if returncode is None:
            converged, niter = check_convergence(config, computation_type)
            if converged or niter >= max_niter:
                stop_MUSICAA("./")
                if computation_type == "steady":
                    config.update({"max_niter_steady_reached": True})
                del config["n_convergence_check"]

        # computation has crashed
        elif returncode > 0:
            if current_restart < restart:
                # reduce CFL
                CFL = float(read_next_line_in_file("param.ini", "CFL"))
                lower_CFL = CFL / divide_CFL_by
                logger.error((f"ERROR -- init_unsteady_2D crashed with CFL={CFL} "
                              f"and will be restarted with lower CFL="
                              f"{lower_CFL}"))
                rm_filelist(["plane*"])
                proc = execute(MUSICAA)
            else:
                raise Exception(f"ERROR -- {computation_type} simulation crashed")

        # computation has completed
        elif returncode == 0:
            break


def stop_MUSICAA(sim_outdir: str):
    """
    **Stops** MUSICAA during execution.
    """
    # send signal to MUSICAA if convergence reached
    with open(f"{sim_outdir}/stop", "w") as stop:
        stop.write("stop")


def get_sensors(sim_outdir: str,
                block_info: dict,
                just_get_niter: bool = False) -> dict:
    """
    **Returns** the sensors time signals as a dictionnary, both point and line types.
    """
    # iterate over sensors
    sensors: dict = {}
    sensors["total_nb_points"] = block_info["total_nb_points"]
    sensors["total_nb_lines"] = block_info["total_nb_lines"]
    nbl = block_info["nbl"]
    for bl in range(nbl):
        bl += 1
        for sensor_type, nb_sensors in [("point", "nb_points"), ("line", "nb_lines")]:
            for sensor_nb in range(block_info[f"block_{bl}"][nb_sensors]):
                sensor_nb += 1
                sensors = read_sensor(sim_outdir, block_info, sensors,
                                      sensor_type, bl, sensor_nb, just_get_niter)
                if just_get_niter:
                    return sensors
    return sensors


def read_sensor(sim_outdir: str,
                block_info: dict,
                sensors: dict,
                sensor_type: str,
                bl: int,
                sensor_nb: int,
                just_get_niter: bool = False) -> dict:
    """
    **Returns** data from a sensor.
    """
    dtype = np.dtype("f8")
    nvars = block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nvars"]
    freq = block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["freq"]
    nz1 = block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nz1"]
    nz2 = block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nz2"]
    nz = 1 + nz2 - nz1

    # open file and collect data
    f = open(
        os.path.join(sim_outdir,
                     f"{sensor_type}_{str(sensor_nb).rjust(3, '0')}_bl{bl}.bin"),
        "r")
    data = np.fromfile(f, dtype=dtype, count=-1)
    niter_sensor = data.size // nvars // nz
    niter = data.size // nvars // nz * freq
    sensors["niter"] = int(niter)
    if just_get_niter:
        return sensors
    data = data[:niter_sensor * nvars * nz].reshape((niter_sensor, nvars, nz))
    sensors[f"block_{bl}"] = {}
    sensors[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"] = data.copy()
    return sensors


def get_residuals(sim_outdir: str, nprint: int, ndeb_RANS: int) -> dict:
    """
    **Returns** residuals from ongoing computations.
    """
    res: dict = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': [], 'RANS_var': []}
    f = open(f"{sim_outdir}/residuals.bin", "rb")
    i4_dtype = np.dtype('<i4')
    i8_dtype = np.dtype('<i8')
    f8_dtype = np.dtype('<f8')
    it = 1
    while True:
        try:
            if it == 1:
                np.fromfile(f, dtype=i4_dtype, count=1)
            else:
                np.fromfile(f, dtype=i8_dtype, count=1)
            res['nn'].append(np.fromfile(f, dtype=i4_dtype, count=1)[0])
            # read arg
            np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rho'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            # read arg
            np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rhou'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            # read arg
            np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rhov'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            # read arg
            np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rhoe'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            if it * nprint > ndeb_RANS:
                # read arg
                np.fromfile(f, dtype=f8_dtype, count=1)
                res['RANS_var'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            it += 1
        except IndexError:
            break
    return res


def check_residuals(config: dict) -> tuple[bool, int]:
    """
    **Returns** True if residual convergence criterion is met. If so,
    the iteration at which statistics are started is returned, otherwise
    the current iteration is returned.
    """
    # get simulation arguments
    convergence_criteria = config["simulator"]["convergence_criteria"]
    residual_convergence_order = convergence_criteria.get("residual_convergence_order", 4)
    if config["is_stats"]:
        return False, 0
    nprint = int(re.findall(r'\b\d+\b',
                            read_next_line_in_file("param.ini",
                                                   "screen"))[0])
    ndeb_RANS = int(read_next_line_in_file("param.ini", "ndeb_RANS"))
    if "n_convergence_check" not in config.keys():
        config.update({"n_convergence_check": 1})

    # if residuals.bin file exists
    try:
        res = get_residuals("./", nprint, ndeb_RANS)
        unwanted = ["nn"]
        # if file not empty
        try:
            niter = int(res["nn"][-1])
            if niter <= ndeb_RANS:
                nvars = 4
                unwanted.append("RANS_var")
            else:
                nvars = 5
        except IndexError:
            return False, 0

        # proceed if this criterion has not already been checked
        if niter // nprint >= config["n_convergence_check"]:

            # compute current order of convergence for each variable
            config["n_convergence_check"] += 1
            nvars_converged = 0
            res_max = []
            for var in set(res) - set(unwanted):
                res_max.append(max(res[var]) - min(res[var]))
                if max(res[var]) - min(res[var]) > residual_convergence_order:
                    nvars_converged += 1

            if niter > ndeb_RANS:
                print((f"it: {niter}; "
                       f"lowest convergence order = {round_number(min(res_max), 'down', 2)}"))
                if nvars_converged == nvars:
                    return True, niter
                else:
                    return False, niter
            else:
                print((f"it: {niter}; RANS starts at it: {ndeb_RANS}"))
                return False, niter
        else:
            return False, niter

    except FileNotFoundError:
        return False, 0


def check_unsteady_crit(config: dict) -> tuple[bool, int]:
    """
    **Returns** (True, iteration) if unsteady ending criterion
    (either transient or stats) is met.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]
    nb_ftt_before_criterion = convergence_criteria["nb_ftt_before_criterion"]
    block_info = get_block_info("./")
    try:
        # read the first sensor to get current iteration
        sensors = get_sensors("./", block_info, just_get_niter=True)
    except FileNotFoundError:
        # computation not started
        return False, 0

    # proceed if this criterion has not already been checked
    niter_ftt = get_niter_ftt("./", config["plot3D"]["mesh"]["chord_length"])
    try:
        if sensors["niter"] // niter_ftt >= config["n_convergence_check"]:
            if config["is_stats"]:
                # check if QoIs have converged
                config["n_convergence_check"] += 1
                return QoI_convergence("./", config, block_info), sensors["niter"]
            else:
                # check if unsteady criteria are met
                if sensors["niter"] // niter_ftt >= nb_ftt_before_criterion:
                    config["n_convergence_check"] += 1
                    return unsteady_crit("./", config, block_info, niter_ftt)

    except KeyError:
        if config["is_stats"]:
            config.update({"n_convergence_check": 0})
        else:
            config.update({"n_convergence_check": nb_ftt_before_criterion})

    return False, sensors["niter"]


def QoI_convergence(sim_outdir: str,
                    config: dict,
                    block_info: dict) -> bool:
    """
    **Returns** (True, iteration) if statistics of the QoIs have converged.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]
    nb_ftt_mov_avg = convergence_criteria["nb_ftt_mov_avg"]
    QoIs_convergence_order = convergence_criteria["QoIs_convergence_order"]

    # compute QoIs and save to file
    new_QoIs_df = compute_QoIs(config)
    filename = os.path.join(sim_outdir, "QoI_convergence.csv")
    if config["n_convergence_check"] == 1:
        # first time computing the QoIs (need at least 2 steps to compute residual)
        new_QoIs_df.to_csv(filename, index=False)
        return False
    try:
        QoIs_df = pd.read_csv(filename)
        QoIs_df = pd.concat([QoIs_df, new_QoIs_df], axis=0)
    except FileNotFoundError:
        QoIs_df = new_QoIs_df
    QoIs_df.to_csv(filename, index=False)
    QoIs = np.array(QoIs_df)

    # clear directory of unused restart<time_stamp>_bl*.bin files
    time_info = get_time_info(sim_outdir)
    restarts = [f"restart{time_stamp}_bl*" for time_stamp in time_info.keys()]
    rm_filelist([os.path.join(sim_outdir, restart) for restart in restarts])

    # compute QoI residuals
    res = 2 * (QoIs[1:, :] - QoIs[:-1, :]) / (QoIs[1:, :] + QoIs[:-1, :])
    res = -np.log10(np.abs(res))
    try:
        mov_avg = np.mean(res[-nb_ftt_mov_avg:, :], axis=0)
    except IndexError:
        mov_avg = np.mean(res, axis=0)
    sensors = get_sensors(sim_outdir, block_info, just_get_niter=True)
    print((f"it: {sensors['niter']}; "
           f"lowest QoI convergence order = {round_number(min(mov_avg),'down', 2)}"))
    if np.sum(mov_avg > QoIs_convergence_order) >= QoIs.shape[-1]:
        return True
    else:
        return False


def unsteady_crit(sim_outdir: str,
                  config: dict,
                  block_info: dict,
                  niter_ftt: int,) -> tuple[bool, int]:
    """
    **Returns** (True, iteration) if ending criterion is met.
    - for the numerical transient, see:
        J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
    - for stats convergence, it is based on mean and rms evolutions obtained
        from the same sensors used to detect the end of the transient.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]
    nb_ftt_mov_avg = convergence_criteria.get("nb_ftt_mov_avg", 4)
    only_compute_mean_crit = convergence_criteria.get("only_compute_mean_crit", True)
    unsteady_convergence_percent_mean = \
        convergence_criteria.get("unsteady_convergence_percent_rms", 1)
    unsteady_convergence_percent_rms = \
        convergence_criteria.get("unsteady_convergence_percent_rms", 1)

    # define criterion to compute
    if config["is_stats"]:
        criterion = "stats_crit"
    else:
        criterion = "Boudet_crit"
    compute_crit: Callable = globals()[f"compute_{criterion}"]

    # get sensors
    sensors = get_sensors(sim_outdir, block_info)
    converged: list[int] = []
    global_crit: list[float] = []
    for block in (key for key in sensors if key.startswith("block")):
        bl = int(block.split("_")[-1])
        for sensor_type, nb_sensors in [("point", "nb_points"), ("line", "nb_lines")]:
            for sensor_nb in range(block_info[f"block_{bl}"][nb_sensors]):
                sensor_nb += 1
                sensor_copy = sensors[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"].copy()
                # moving average to avoid temporary convergence
                mov_avg_crit: list[float] = []
                for sample_nb in range(nb_ftt_mov_avg):
                    sample_size = sample_nb * niter_ftt + 1
                    sensor = sensor_copy[:-sample_size]
                    if sensors["niter"] < sample_size + niter_ftt:
                        break
                    crit = compute_crit(sensor)
                    mov_avg_crit.append(crit)

                # if threshold w.r.t mean only or rms too
                if only_compute_mean_crit:
                    unsteady_convergence_percent = unsteady_convergence_percent_mean
                else:
                    unsteady_convergence_percent = unsteady_convergence_percent_rms

                # check if converged
                mov_avg = sum(mov_avg_crit) / len(mov_avg_crit)
                if mov_avg < unsteady_convergence_percent:
                    is_converged = True
                else:
                    is_converged = False
                converged.append(is_converged)
                global_crit.append(mov_avg)
    if config["is_stats"]:
        print((f"it: {sensors['niter']}; "
               f"max statistics variation = {round_number(max(global_crit), 'closest', 2)}%"))
    else:
        print((f"it: {sensors['niter']}; "
               f"max transient variation = {round_number(max(global_crit), 'closest', 2)}%"))
    if sum(converged) == sensors["total_nb_points"] + sensors["total_nb_lines"]:
        return True, sensors["niter"]
    else:
        return False, sensors["niter"]


def compute_QoIs(config: dict) -> pd.DataFrame:
    """
    **Returns** the QoIs during computation in a DataFrame.
    """
    qty_list: list[list[float]] = []
    head_list: list[str] = []
    # loop over the post-processing arguments to extract from the results
    for qty in config["post_process"]["outputs"]:
        # check if the method for computing qty exists
        try:
            # get arguments
            get_args: Callable = globals()[f"args_{qty}"]
            args = get_args("./", config)
            get_value: Callable = globals()[qty]
            value = get_value("./", args)
        except AttributeError:
            raise Exception(f"ERROR -- method for computing {qty} does not exist")
        try:
            # compute simulation results
            qty_list.append(value)
            head_list.append(qty)
        except Exception as e:
            logger.warning(f"could not compute {qty}")
            logger.warning(f"exception {e} was raised")
    # pd.Series allows columns of different lengths
    df = pd.DataFrame({head_list[i]: pd.Series(qty_list[i]) for i in range(len(qty_list))})
    return df


def compute_Boudet_crit(config: dict, sensor: np.ndarray) -> float:
    """
    **Returns** (True, iteration to start statistics) if transient ending criterion is met:
    see J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
    This criterion is here modified in an attempt to reduce the unused simulation
    time from 3 additional transients to 1 (in practice ~2) by considering thirds
    of the second half rather than quarters of the whole.
    The longer the transient, the more efficient this approach.
    In the end, a geometric mean of both criteria is retained.
    """
    # get simulation arguments
    convergence_criteria = config["simulator"]["convergence_criteria"]
    only_compute_mean_crit = convergence_criteria.get("only_compute_mean_crit", True)
    Boudet_criterion_type = convergence_criteria.get("Boudet_criterion_type", "mean")

    # shape of array: (niter, nvars, nz)
    niter = sensor.shape[0]
    nvars = sensor.shape[1]
    nz = sensor.shape[2]

    # get mean, rms and quarters
    quarter = niter // 4
    mean = np.mean(sensor, axis=(0, 2))
    sensor_squared = (sensor - mean[np.newaxis, :, np.newaxis])**2
    if niter % 4 == 0:
        mean_quarters_ = sensor.reshape((4, quarter, nvars, nz))
        rms_quarters_ = sensor_squared.reshape((4, quarter, nvars, nz))
    else:
        mean_quarters_ = np.zeros((4, quarter, nvars, nz))
        rms_quarters_ = np.zeros((4, quarter, nvars, nz))
        mean_quarters_[0] = sensor[:quarter]
        mean_quarters_[1] = sensor[quarter:2 * quarter]
        mean_quarters_[2] = sensor[2 * quarter:3 * quarter]
        mean_quarters_[3] = sensor[3 * quarter:4 * quarter]
        rms_quarters_[0] = sensor_squared[:quarter]
        rms_quarters_[1] = sensor_squared[quarter:2 * quarter]
        rms_quarters_[2] = sensor_squared[2 * quarter:3 * quarter]
        rms_quarters_[3] = sensor_squared[3 * quarter:4 * quarter]
    mean_quarters = np.mean(mean_quarters_, axis=(1, 3))
    rms_quarters = np.mean(np.sqrt(np.mean(rms_quarters_, axis=1)), axis=-1)

    # compute criterion
    crit = []
    for var in range(nvars):
        mean_crit = [abs((mean_quarters[1, var] - mean_quarters[2, var])
                     / mean_quarters[3, var]),
                     abs((mean_quarters[1, var] - mean_quarters[3, var])
                     / mean_quarters[3, var]),
                     abs((mean_quarters[2, var] - mean_quarters[3, var])
                     / mean_quarters[3, var])
                     ]
        rms_crit = [abs((rms_quarters[1, var] - rms_quarters[2, var])
                    / rms_quarters[3, var]),
                    abs((rms_quarters[1, var] - rms_quarters[3, var])
                    / rms_quarters[3, var]),
                    abs((rms_quarters[2, var] - rms_quarters[3, var])
                    / rms_quarters[3, var])
                    ]
        if only_compute_mean_crit:
            crit.append(max(mean_crit) * 100)
        else:
            crit.append(max(max(mean_crit), max(rms_crit)) * 100)
    # return if original Boudet criterion requested
    if Boudet_criterion_type == "original":
        return max(crit)

    # get mean, rms and quarters
    half = niter // 2
    sixth = half // 3
    mean = np.mean(sensor, axis=(0, 2))
    sensor_squared = (sensor - mean[np.newaxis, :, np.newaxis])**2
    sensor = sensor[half:]
    sensor_squared = sensor_squared[half:]
    if niter % 6 == 0:
        mean_sixths_ = sensor.reshape((3, sixth, nvars, nz))
        rms_sixths_ = sensor_squared.reshape((3, sixth, nvars, nz))
    else:
        mean_sixths_ = np.zeros((3, sixth, nvars, nz))
        rms_sixths_ = np.zeros((3, sixth, nvars, nz))
        mean_sixths_[0] = sensor[:sixth]
        mean_sixths_[1] = sensor[sixth:2 * sixth]
        mean_sixths_[2] = sensor[2 * sixth:3 * sixth]
        rms_sixths_[0] = sensor_squared[:sixth]
        rms_sixths_[1] = sensor_squared[sixth:2 * sixth]
        rms_sixths_[2] = sensor_squared[2 * sixth:3 * sixth]
    mean_sixths = np.mean(mean_sixths_, axis=(1, 3))
    rms_sixths = np.mean(np.sqrt(np.mean(rms_sixths_, axis=1)), axis=-1)

    # compute criterion
    crit_modif = []
    for var in range(nvars):
        mean_crit = [abs((mean_sixths[0, var] - mean_sixths[1, var])
                     / mean_sixths[2, var]),
                     abs((mean_sixths[0, var] - mean_sixths[2, var])
                     / mean_sixths[2, var]),
                     abs((mean_sixths[1, var] - mean_sixths[2, var])
                     / mean_sixths[2, var])
                     ]
        rms_crit = [abs((rms_sixths[0, var] - rms_sixths[1, var])
                    / rms_sixths[2, var]),
                    abs((rms_sixths[0, var] - rms_sixths[2, var])
                    / rms_sixths[2, var]),
                    abs((rms_sixths[1, var] - rms_sixths[2, var])
                    / rms_sixths[2, var])
                    ]
        if only_compute_mean_crit:
            crit_modif.append(max(mean_crit) * 100)
        else:
            crit_modif.append(max(max(mean_crit), max(rms_crit)) * 100)
    # return if modified Boudet criterion requested
    if Boudet_criterion_type == "modified":
        return max(crit_modif)

    # compute geometric mean if requested
    crit_mean = [abs(crit_Boudet * crit_modif[i]) / (crit_Boudet * crit_modif[i])
                 * np.sqrt(abs(crit_Boudet * crit_modif[i])) for
                 i, crit_Boudet in enumerate(crit)]
    return max(crit_mean)


def compute_stats_crit(config: dict, sensor: np.ndarray) -> float:
    """
    **Returns** (True, iteration to start statistics) if statistics convergence
    criterion is met. It is based on mean and rms evolutions obtained
    from the same sensors used to detect the end of the transient.
    """
    # get simulation arguments
    convergence_criteria = config["simulator"]["convergence_criteria"]
    only_compute_mean_crit = convergence_criteria.get("only_compute_mean_crit", True)

    # shape of array: (niter, nvars, nz)
    niter = sensor.shape[0]
    nvars = sensor.shape[1]
    nz = sensor.shape[2]

    # get mean, rms and quarters
    half = niter // 2
    mean = np.mean(sensor, axis=(0, 2))
    sensor_squared = (sensor - mean[np.newaxis, :, np.newaxis])**2
    if niter % 2 == 0:
        mean_halves_ = sensor.reshape((2, half, nvars, nz))
        rms_halves_ = sensor_squared.reshape((2, half, nvars, nz))
    else:
        mean_halves_ = np.zeros((2, half, nvars, nz))
        rms_halves_ = np.zeros((2, half, nvars, nz))
        mean_halves_[0] = sensor[:half]
        mean_halves_[1] = sensor[half:2 * half]
        rms_halves_[0] = sensor_squared[:half]
        rms_halves_[1] = sensor_squared[half:2 * half]
    mean_halves = np.mean(mean_halves_, axis=(1, 3))
    rms_halves = np.mean(np.sqrt(np.mean(rms_halves_, axis=1)), axis=-1)

    # compute criterion
    crit = []
    for var in range(nvars):
        mean_var = 0.5 * (mean_halves[0, var] + mean_halves[1, var])
        rms_var = 0.5 * (rms_halves[0, var] + rms_halves[1, var])
        mean_crit = abs((mean_halves[0, var] - mean_halves[1, var])
                        / mean_var)
        rms_crit = abs((rms_halves[0, var] - rms_halves[1, var])
                       / rms_var)
        if only_compute_mean_crit:
            crit.append(mean_crit * 100)
        else:
            crit.append(max(mean_crit, rms_crit) * 100)

    return max(crit)


def check_convergence(config: dict,
                      computation_type: str) -> tuple[bool, int]:
    """
    **Returns** True if MUSICAA computation convergence criterion is met. If so,
    returns current iteration number to start statistics.
    """
    if computation_type == "steady":
        return check_residuals(config)
    else:
        return check_unsteady_crit(config)


def pre_process_stats(config: dict, computation_type: str):
    """
    **Pre-processes** computation for statistics.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]

    # get current iteration from time.ini
    time_info = get_time_info("./")
    ndeb_stats = time_info["niter_total"]

    # modify param.ini file
    args = {}
    args.update({"from_field": "2"})
    args.update({"Iteration number to start statistics": f"{ndeb_stats + 1}"})
    if computation_type == "steady":
        # if steady computation did not fully converge
        if config["max_niter_steady_reached"]:
            niter_stats = convergence_criteria.get("niter_stats_steady", 10000)
        else:
            niter_stats = 2
    elif computation_type == "unsteady":
        niter_stats = 999999
        # add frequency for QoI convergence check: will ask MUSICAA to output stats files
        niter_ftt = get_niter_ftt("./", config["plot3D"]["mesh"]["chord_length"])
        freqs = read_next_line_in_file("param.ini",
                                       "Output frequencies: screen / stats / fields")
        freqs = freqs.split()
        args.update({"Output frequencies: screen / stats / fields":
                     f"{freqs[0]} {freqs[1]} {niter_ftt}"})
    args.update({"Max number of temporal iterations": f"{niter_stats} 3000.0"})
    custom_input("param.ini", args)


def change_dimensions_param_blocks():
    """
    **Modifies** the param_blocks.ini file of a given simulation to 2D.
    """
    # file and block info
    block_info = get_block_info("./")
    nbl = block_info["nbl"]

    with open("param_blocks.ini", "r") as f:
        filedata = f.readlines()
    # iterate over each block
    for bl in range(nbl):
        bl += 1
        pattern = f"! Block #{bl}"
        for i, line in enumerate(filedata):
            if pattern in line:
                # change mesh dimension
                filedata[i + 5] = "      1          1     |  K-direction\n"
                # if point sensors located at nz!=1, change to nz=1
                nb_points = block_info[f"block_{bl}"]["nb_points"]
                if nb_points > 0:
                    for point_nb in range(nb_points):
                        point_nb += 1
                        if block_info[f"block_{bl}"][f"point_{point_nb}"]["nz1"] != 1:
                            position \
                                = block_info[f"block_{bl}"][(f"point_"
                                                             f"{point_nb}")]["position"]
                            var_list \
                                = block_info[f"block_{bl}"][(f"point_"
                                                             f"{point_nb}")]["var_list"]
                            info_snap = [int(dim) for dim in
                                         re.findall(r"\d+", filedata[i + 21 + position])]
                            info_snap[4], info_snap[5] = 1, 1
                            replacement = "   " + "    ".join([str(j) for j in info_snap]) \
                                + "    " + " ".join(var_list) + "\n"
                            filedata[i + 21 + position] = replacement
                # if line sensors, change to points
                nb_lines = block_info[f"block_{bl}"]["nb_lines"]
                if nb_lines > 0:
                    for line_nb in range(nb_lines):
                        line_nb += 1
                        position \
                            = block_info[f"block_{bl}"][f"line_{line_nb}"]["position"]
                        var_list \
                            = block_info[f"block_{bl}"][f"line_{line_nb}"]["var_list"]
                        info_snap = [int(dim) for dim in
                                     re.findall(r"\d+", filedata[i + 21 + position])]
                        info_snap[4], info_snap[5] = 1, 1
                        replacement = "   " + "    ".join([str(j) for j in info_snap]) \
                            + "    " + " ".join(var_list) + "\n"
                        filedata[i + 21 + position] = replacement
    with open("param_blocks.ini", "w") as f:
        f.writelines(filedata)


def pre_process_init_unsteady(dimension: str):
    """
    **Pre-processes** unsteady computation to initialize with 2D or 3D simulation.
    """
    args = {}
    if dimension == "2D":
        # save inputs to temporary files
        cp_filelist(["param.ini", "param_blocks.ini"],
                    ["param.ini_3D", "param_blocks.ini_3D"])
        # modify param.ini file
        args.update({"Implicit Residual Smoothing": "2"})
        args.update({"Residual smoothing parameter": "0.42 0.1 0.005 0.00025 0.0000125"})
        is_SF = read_next_line_in_file("param.ini", ("Selective Filtering: is_SF"
                                                     "[T:SF; F:artifical viscosity]"))
        if is_SF:
            coeff = 0.2
            coeff_shock = 0.2
        else:
            coeff = 1.0
            coeff_shock = 1.0
        args.update({("(between 0 and 1 for SF, recommended"
                    "value 0.1 / around 1.0 for art.visc)"): f"{coeff}"})
        args.update({"Indicator is_shock and Coefficient of low-order term":
                     f"T {coeff_shock}"})
        args.update({"Switch of Edoh": "T"})
        args.update({"Shock sensor: Ducros sensor": "T 0.5"})
        custom_input("param.ini", args)
        # modify param_blocks.ini file: 3D->2D
        change_dimensions_param_blocks()
    else:
        # recover original input files
        cp_filelist(["param.ini_3D", "param_blocks.ini_3D"],
                    ["param.ini", "param_blocks.ini"])
        # change to restart
        args.update({"from_interp": "2"})
        custom_input("param.ini", args)


def get_residual(res_file: str = "residual.dat", entry: int = -2) -> float:
    res_line = open(res_file, "r").read().splitlines()[-1]
    res_list = list(map(float, res_line.split()))
    return float(res_list[entry])


def get_turbocoef(res_file: str = "turbocoef.dat") -> list[float]:
    # Iter (0) CPU (1)  DebitIn (2)  DebitOut (3)  DebitRat (4)  PreTotIn (5)  PreTotOut (6)
    # PreTotRat (7) TemTotIn (8) TemTotOut (9) TemTotRat (10) IsentropicEff (11) LossCoef (12)
    # HeatFlux (13) TempCoef (14)
    res_line = open(res_file, "r").read().splitlines()[-1]
    return list(map(float, res_line.split()))


def main() -> int:
    """
    This program runs a WOLF CFD simulation at ADP.
    In multisimulation mode, OP1 (+5°) and OP2 (-5°) simulations are also executed.
    """
    cwd = os.getcwd()

    # read config file
    with open("config.json") as jfile:
        config = json.load(jfile)
    # set computation type
    computation_type: str = read_next_line_in_file("param.ini",
                                                   "DES without subgrid")
    computation_type = "unsteady" if computation_type == "N" else "steady"
    execute_computation: Callable = globals()[f"execute_{computation_type}"]

    print("** ADP SIMULATION **")
    print("** -------------- **")
    sim_dir = "ADP"
    shutil.rmtree(sim_dir, ignore_errors=True)
    os.mkdir(sim_dir)
    args = {}
    # add simulation files
    cp_filelist(["param.ini", "param_blocks.ini", "param_rans.ini", "feos_air.ini"],
                [sim_dir] * 4)
    os.chdir(sim_dir)
    # specify path for mesh files
    old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
    dir_grid: str = "'" + os.path.join("../", old_dir_grid) + "'"
    args.update({"Directory for grid files": dir_grid})
    custom_input("param.ini", args)
    # execute computation
    sim_pro = []
    config_ADP = config.copy()
    sim_pro.append(execute_computation(config_ADP))

    # try:
    #     execute_LES()
    #     turbocoef = get_turbocoef()
    #     print(f">> residual : {get_residual()}")
    #     print(f">> debit ratio: {turbocoef[4]}")
    #     print(f">> total pressure ratio: {turbocoef[7]}")
    #     print(f">> total temperature ratio: {turbocoef[10]}")
    #     print(f">> isentropic efficiency: {turbocoef[11]}")
    #     print(f">> loss coefficient: {turbocoef[12]}\n")
    # except CalledProcessError:
    #     print(f">> ADP failed after {time.time() - t0} seconds.")
    #     return FAILURE
    # print(f">> ADP finished successfully in {time.time() - t0} seconds.\n")

    os.chdir(cwd)
    print("** OP1 SIMULATION (+5 deg.) **")
    print("** ------------------------ **")
    sim_dir = "OP1"
    os.mkdir(sim_dir)
    # add simulation files
    cp_filelist(["param.ini", "param_blocks.ini", "param_rans.ini", "feos_air.ini"],
                [sim_dir] * 4)
    os.chdir(sim_dir)
    args = {}
    # specify path for mesh files
    old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
    dir_grid: str = "'" + os.path.join("../", old_dir_grid) + "'"
    args.update({"Directory for grid files": dir_grid})
    # change flow angle
    args.update({"Flow angles": "48. 0."})
    custom_input("param.ini", args)

    # EXECUTE 2nd LES
    config_OP1 = config.copy()
    sim_pro.append(execute_computation(config_OP1))

    # try:
    #     run(musicaa_cmd, "musicaa.job")
    #     turbocoef = get_turbocoef()
    #     print(f">> residual : {get_residual()}")
    #     print(f">> debit ratio: {turbocoef[4]}")
    #     print(f">> total pressure ratio: {turbocoef[7]}")
    #     print(f">> total temperature ratio: {turbocoef[10]}")
    #     print(f">> isentropic efficiency: {turbocoef[11]}")
    #     print(f">> loss coefficient: {turbocoef[12]}\n")
    # except CalledProcessError:
    #     print(f">> OP1 failed after {time.time() - t0} seconds.")
    #     return FAILURE
    # print(f">> OP1 finished successfully in {time.time() - t0} seconds.\n")

    os.chdir(cwd)
    print("** OP2 SIMULATION (-5 deg.) **")
    print("** ------------------------ **")
    sim_dir = "OP2"
    os.mkdir(sim_dir)
    # add simulation files
    cp_filelist(["param.ini", "param_blocks.ini", "param_rans.ini", "feos_air.ini"],
                [sim_dir] * 4)
    os.chdir(sim_dir)
    args = {}
    # specify path for mesh files
    old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
    dir_grid: str = "'" + os.path.join("../", old_dir_grid) + "'"
    args.update({"Directory for grid files": dir_grid})
    # change flow angle
    args.update({"Flow angles": "38. 0."})
    custom_input("param.ini", args)

    # EXECUTE 2nd LES
    config_OP2 = config.copy()
    sim_pro.append(execute_computation(config_OP2))

    # try:
    #     run(musicaa_cmd, "musicaa.job")
    #     turbocoef = get_turbocoef()
    #     print(f">> residual : {get_residual()}")
    #     print(f">> debit ratio: {turbocoef[4]}")
    #     print(f">> total pressure ratio: {turbocoef[7]}")
    #     print(f">> total temperature ratio: {turbocoef[10]}")
    #     print(f">> isentropic efficiency: {turbocoef[11]}")
    #     print(f">> loss coefficient: {turbocoef[12]}\n")
    # except CalledProcessError:
    #     print(f">> OP2 failed after {time.time() - t0} seconds.")
    #     return FAILURE
    # print(f">> OP2 finished successfully in {time.time() - t0} seconds.")

    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
