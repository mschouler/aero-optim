import argparse
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
import time
import glob
import traceback

from aero_optim.mesh.mesh import get_block_info
from custom_cascade_musicaa import get_time_info, get_niter_ftt, compute_QoIs
from aero_optim.utils import (cp_filelist, rm_filelist, read_next_line_in_file,
                              custom_input, submit_popen_process, wait_for_it)

EPSILON: float = 1e-6
FAILURE: int = 1
SUCCESS: int = 0
MUSICAA: str = "mpiexec -n @nproc /home/mschouler/bin/musicaa"

print = functools.partial(print, flush=True)

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def save_simu(sim_outdir: str, step_name: str):
    """
    **Saves** the simulation results to step_name in sim_outdir.
    """
    file_list = (
        glob.glob(os.path.join(sim_outdir, "restart*"))
        + glob.glob(os.path.join(sim_outdir, "plane_00*"))
        + glob.glob(os.path.join(sim_outdir, "stats*"))
        + glob.glob(os.path.join(sim_outdir, "*.dat"))
        + glob.glob(os.path.join(sim_outdir, "*.ini"))
        + glob.glob(os.path.join(sim_outdir, "line_*"))
        + glob.glob(os.path.join(sim_outdir, "point_*"))
        + glob.glob(os.path.join(sim_outdir, "grid_bl*"))
    )
    os.makedirs(os.path.join(sim_outdir, step_name), exist_ok=True)
    cp_filelist(file_list, len(file_list) * [os.path.join(sim_outdir, step_name)])


def get_nproc(sim_outdir: str) -> int:
    """
    **Computes** the number of proc required by the simulation from param_blocks.ini.
    """
    f = open(os.path.join(sim_outdir, "param_blocks.ini"), "r").read().splitlines()
    nproc = 0
    for ll, line in enumerate(f):
        if "Nb procs" in line:
            nproc += np.prod(np.array([int(re.findall(r'\d+', f[ll + j])[1]) for j in range(1, 4)]))
    return nproc


def execute_steady(config: dict, sim_outdir: str):
    """
    **Executes** a Reynolds-Averaged Navier-Stokes simulation with MUSICAA.
    """
    # execute computation
    config.update({"is_stats": False})
    exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
    print(f"INFO -- submit popen steady: {exec_cmd}")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc, config, sim_outdir, "steady")

    # gather statistics
    config.update({"is_stats": True})
    pre_process_stats(config, sim_outdir, "steady")
    nproc = get_nproc(sim_outdir)
    exec_cmd = MUSICAA.replace("@nproc", str(nproc))
    print(f"INFO -- submit popen steady: {exec_cmd}")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc, config, sim_outdir, "steady")


def execute_unsteady(config: dict, sim_outdir: str):
    """
    **Executes** a Large Eddy Simulation with MUSICAA.
    """
    # initialize LES with a fully developed 2D field
    config.update({"is_stats": False})
    pre_process_init_unsteady("2D", sim_outdir)
    exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
    print(f"INFO -- submit popen unsteady 2D: {exec_cmd}")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc, config, sim_outdir, "unsteady", unsteady_step="init_2D")
    # save results
    save_simu(sim_outdir, "init_2D")

    # carry on with 3D transient
    pre_process_init_unsteady("3D", sim_outdir)
    exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
    print(f"INFO -- submit popen unsteady 3D: {exec_cmd}")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc, config, sim_outdir, "unsteady", unsteady_step="init_3D")
    save_simu(sim_outdir, "init_3D")

    # gather statistics
    config.update({"is_stats": True})
    pre_process_stats(config, sim_outdir, "unsteady")
    exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
    print(f"INFO -- submit popen unsteady 3D + stats: {exec_cmd}")
    _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
    monitor_sim_progress(proc, config, sim_outdir, "unsteady")


def monitor_sim_progress(proc: subprocess.Popen,
                         config: dict,
                         sim_outdir: str,
                         computation_type: str,
                         unsteady_step: str = ""):
    """
    **Monitors** a simulation.
    """
    # get simulation arguments
    restart = config["simulator"].get("restart_musicaa", 1)
    divide_CFL_by = config["simulator"].get("divide_CFL_by", 1.2)
    if unsteady_step == "init_2D":
        max_niter = config["simulator"]["convergence_criteria"].get("max_niter_init_2D", 200000)
    elif unsteady_step == "init_3D":
        max_niter = config["simulator"]["convergence_criteria"].get("max_niter_init_3D", 200000)
    if config["is_stats"]:
        max_niter = config["simulator"]["convergence_criteria"].get("max_niter_stats", 200000)
    elif computation_type == "steady":
        config.update({"max_niter_steady_reached": False})
        max_niter = config["simulator"]["convergence_criteria"].get("max_niter_steady", 100000)

    current_restart = 0
    while True:
        returncode = proc.poll()
        # computation still running
        if returncode is None:
            converged, niter = check_convergence(config, sim_outdir, computation_type)
            if converged or niter >= max_niter:
                print(f"INFO -- simulation converged: {converged}, "
                      f"niter: {niter}, max_niter: {max_niter}")
                print("INFO -- musicaa will be stopped")
                stop_MUSICAA(sim_outdir, proc)
                if computation_type == "steady":
                    config.update({"max_niter_steady_reached": True})
                del config["n_convergence_check"]

        # computation has crashed
        elif returncode > 0:
            if current_restart < restart:
                # reduce CFL
                param_ini = os.path.join(sim_outdir, "param.ini")
                CFL = float(read_next_line_in_file(param_ini, "CFL"))
                lower_CFL = CFL / divide_CFL_by
                print(f"ERROR -- init_unsteady_2D crashed with CFL={CFL} "
                      f"and will be restarted with lower CFL={lower_CFL}")
                custom_input(param_ini, {"CFL": lower_CFL})
                # clean directory
                rm_filelist(["plane*", "line*", "point*", "restart*"])
                exec_cmd = MUSICAA.replace("@nproc", str(get_nproc(sim_outdir)))
                _, proc = submit_popen_process("musicaa", exec_cmd.split(), sim_outdir)
                current_restart += 1
            else:
                raise Exception(f"ERROR -- {computation_type} simulation crashed")

        # computation has completed
        elif returncode == 0:
            break


def stop_MUSICAA(sim_outdir: str, proc: subprocess.Popen):
    """
    **Stops** MUSICAA during execution.
    """
    # send signal to MUSICAA if convergence reached
    with open(f"{sim_outdir}/stop", "w") as stop:
        stop.write("stop")
    # wait for the simulation to finish
    while True:
        returncode = proc.poll()
        if returncode is None:
            time.sleep(1.)
        elif returncode == 0:
            print("INFO -- simulation stopped successfully")
            break
        else:
            print("INFO -- simulation stopped with an error")
            break


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
    sensors[f"block_{bl}"] = {} if f"block_{bl}" not in sensors.keys() else sensors[f"block_{bl}"]
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


def check_residuals(config: dict, sim_outdir: str) -> tuple[bool, int]:
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
    param_ini = os.path.join(sim_outdir, "param.ini")
    nprint = int(re.findall(r'\b\d+\b',
                            read_next_line_in_file(param_ini,
                                                   "screen"))[0])
    ndeb_RANS = int(read_next_line_in_file(param_ini, "ndeb_RANS"))
    if "n_convergence_check" not in config.keys():
        config.update({"n_convergence_check": 1})

    # if residuals.bin file exists
    try:
        res = get_residuals(sim_outdir, nprint, ndeb_RANS)
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
                if not res[var]:
                    return False, niter
                res_max.append(max(res[var]) - min(res[var]))
                if max(res[var]) - min(res[var]) > residual_convergence_order:
                    nvars_converged += 1

            if niter > ndeb_RANS:
                print((f"it: {niter}; "
                       f"lowest convergence order = {min(res_max):.2f}"))
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


def check_unsteady_crit(config: dict, sim_outdir: str) -> tuple[bool, int]:
    """
    **Returns** (True, iteration) if unsteady ending criterion
    (either transient or stats) is met.
    """
    # get simulation args
    convergence_criteria: dict = config["simulator"]["convergence_criteria"]
    nb_ftt_before_criterion = convergence_criteria["nb_ftt_before_criterion"]
    block_info = get_block_info(sim_outdir)
    try:
        # read the first sensor to get current iteration
        sensors = get_sensors(sim_outdir, block_info, just_get_niter=True)
    except FileNotFoundError:
        # computation not started
        return False, 0

    # proceed if this criterion has not already been checked
    niter_ftt = get_niter_ftt(sim_outdir, config["gmsh"]["chord_length"])
    try:
        if (sensors["niter"] - config["niter_0"]) // niter_ftt >= config["n_convergence_check"]:
            if config["is_stats"]:
                # check if QoIs have converged
                config["n_convergence_check"] += 1
                return (
                    QoI_convergence(sim_outdir, config, block_info),
                    sensors["niter"] - config["niter_0"]
                )
            else:
                # check if unsteady criteria are met
                if (sensors["niter"] - config["niter_0"]) // niter_ftt >= nb_ftt_before_criterion:
                    config["n_convergence_check"] += 1
                    return unsteady_crit(sim_outdir, config, block_info)

    except KeyError:
        traceback.print_exc()
        if config["is_stats"]:
            config.update({"n_convergence_check": 1})
        else:
            config.update({"n_convergence_check": nb_ftt_before_criterion})
        config["niter_0"] = sensors["niter"] - 1 if sensors["niter"] > 0 else 0
        print(f"INFO -- niter_ftt: {niter_ftt}")
        print(f"INFO -- niter_0: {config['niter_0']}")

    return False, sensors["niter"] - config["niter_0"]


def QoI_convergence(sim_outdir: str,
                    config: dict,
                    block_info: dict) -> bool:
    """
    **Returns** (True, iteration) if statistics of the QoIs have converged.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]
    QoIs_convergence_order = convergence_criteria["QoIs_convergence_order"]

    # compute QoIs and save to file
    sensors = get_sensors(sim_outdir, block_info, just_get_niter=True)
    new_QoIs_df = compute_QoIs(config, sim_outdir)
    filename = os.path.join(sim_outdir, "QoI_convergence.csv")
    if not os.path.isfile(filename):
        # first time computing the QoIs
        new_QoIs_df.to_csv(filename, index=False)
        print(f"it: {sensors['niter']}, QoI: {new_QoIs_df.to_numpy()}")
        return False
    QoIs_df = pd.concat([pd.read_csv(filename), new_QoIs_df], axis=0)
    QoIs_df.to_csv(filename, index=False)
    QoIs = QoIs_df.to_numpy()

    # clear directory of unused restart<time_stamp>_bl*.bin files
    time_info = get_time_info(sim_outdir)
    restarts = [f"restart{time_stamp}_bl*" for time_stamp in time_info.keys()]
    rm_filelist([os.path.join(sim_outdir, restart) for restart in restarts])

    # compute QoI residuals
    print(f"it: {sensors['niter'] - config['niter_0']}, QoI: {QoIs}")
    if len(QoIs) < 3:
        return False
    else:
        delta_1 = abs((QoIs[-1] - QoIs[-2]) / QoIs[-1])
        delta_2 = abs((QoIs[-1] - QoIs[-3]) / QoIs[-1])
        order = -np.log10(max(delta_1, delta_2))[0]
        print(f"QoI convergence order = {order:.2f}")
        return order < QoIs_convergence_order


def unsteady_crit(sim_outdir: str,
                  config: dict,
                  block_info: dict) -> tuple[bool, int]:
    """
    **Returns** (True, iteration) if ending criterion is met.
    - for the numerical transient, see:
        J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
    - for stats convergence, it is based on mean and rms evolutions obtained
        from the same sensors used to detect the end of the transient.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]
    only_compute_mean_crit = convergence_criteria.get("only_compute_mean_crit", True)
    unsteady_convergence_percent_mean = \
        convergence_criteria.get("unsteady_convergence_percent_mean", 1)
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
                # compute sensor crit
                crit = compute_crit(config, sensor_copy)

                # if threshold w.r.t mean only or rms too
                if only_compute_mean_crit:
                    unsteady_cv_pt = unsteady_convergence_percent_mean
                else:
                    unsteady_cv_pt = unsteady_convergence_percent_rms

                # check if converged
                if crit < unsteady_cv_pt:
                    is_converged = True
                else:
                    is_converged = False
                converged.append(is_converged)
                global_crit.append(crit)
    if config["is_stats"]:
        print((f"it: {sensors['niter']}; "
               f"max statistics variation = {max(global_crit):.2f}%"))
    else:
        print((f"it: {sensors['niter']}; "
               f"max transient variation = {max(global_crit):.2f}%"))
    if sum(converged) == sensors["total_nb_points"] + sensors["total_nb_lines"]:
        print(f"INFO -- convergence reached after {sensors['niter']} steps")
        return True, sensors["niter"] - config["niter_0"]
    else:
        return False, sensors["niter"] - config["niter_0"]


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
    monitored_variables = convergence_criteria.get("monitored_variables", 0)

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
    mean_quarters = np.mean(mean_quarters_, axis=(1, 3)) + EPSILON
    rms_quarters = np.mean(np.sqrt(np.mean(rms_quarters_, axis=1)), axis=-1) + EPSILON

    # compute criterion
    crit = []
    nvars = monitored_variables if monitored_variables else nvars
    for var in range(nvars):
        mean_crit = [abs((mean_quarters[1, var] - mean_quarters[2, var])
                     / mean_quarters[3, var]),
                     abs((mean_quarters[1, var] - mean_quarters[3, var])
                     / mean_quarters[3, var]),
                     abs((mean_quarters[2, var] - mean_quarters[3, var])
                     / mean_quarters[3, var])]
        rms_crit = [abs((rms_quarters[1, var] - rms_quarters[2, var])
                    / rms_quarters[3, var]),
                    abs((rms_quarters[1, var] - rms_quarters[3, var])
                    / rms_quarters[3, var]),
                    abs((rms_quarters[2, var] - rms_quarters[3, var])
                    / rms_quarters[3, var])]
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
    mean_sixths = np.mean(mean_sixths_, axis=(1, 3)) + EPSILON
    rms_sixths = np.mean(np.sqrt(np.mean(rms_sixths_, axis=1)), axis=-1) + EPSILON

    # compute criterion
    crit_modif = []
    for var in range(nvars):
        mean_crit = [abs((mean_sixths[0, var] - mean_sixths[1, var])
                     / mean_sixths[2, var]),
                     abs((mean_sixths[0, var] - mean_sixths[2, var])
                     / mean_sixths[2, var]),
                     abs((mean_sixths[1, var] - mean_sixths[2, var])
                     / mean_sixths[2, var])]
        rms_crit = [abs((rms_sixths[0, var] - rms_sixths[1, var])
                    / rms_sixths[2, var]),
                    abs((rms_sixths[0, var] - rms_sixths[2, var])
                    / rms_sixths[2, var]),
                    abs((rms_sixths[1, var] - rms_sixths[2, var])
                    / rms_sixths[2, var])]
        if only_compute_mean_crit:
            crit_modif.append(max(mean_crit) * 100)
        else:
            crit_modif.append(max(max(mean_crit), max(rms_crit)) * 100)
    # return if modified Boudet criterion requested
    if Boudet_criterion_type == "modified":
        return max(crit_modif)

    # compute geometric mean if requested
    crit_mean = [abs(crit_Boudet * crit_modif[i]) / (crit_Boudet * crit_modif[i] + EPSILON)
                 * np.sqrt(abs(crit_Boudet * crit_modif[i])) for
                 i, crit_Boudet in enumerate(crit)]
    return max(crit_mean)


def compute_stats_crit(config: dict, sensor: np.ndarray) -> float:
    """
    **Returns** (True, iteration to start statistics) if statistics convergence
    criterion is met. It is based on mean and rms evolutions obtained
    from the same sensors used to detect the end of the transient.

    /!\ this function is never called
        because "is_stat" = True calls QoI_convergence (see check_unsteady_crit).
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
        mean_var = 0.5 * (mean_halves[0, var] + mean_halves[1, var]) + EPSILON
        rms_var = 0.5 * (rms_halves[0, var] + rms_halves[1, var]) + EPSILON
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
                      sim_outdir: str,
                      computation_type: str) -> tuple[bool, int]:
    """
    **Returns** True if MUSICAA computation convergence criterion is met. If so,
    returns current iteration number to start statistics.
    """
    if computation_type == "steady":
        return check_residuals(config, sim_outdir)
    else:
        return check_unsteady_crit(config, sim_outdir)


def pre_process_stats(config: dict, sim_outdir: str, computation_type: str):
    """
    **Pre-processes** computation for statistics.
    """
    # get simulation args
    convergence_criteria = config["simulator"]["convergence_criteria"]

    # get current iteration from time.ini
    time_info = get_time_info(sim_outdir)
    ndeb_stats = time_info["niter_total"]

    # modify param.ini file
    args = {}
    args.update({"from_field": "2"})
    args.update({"Iteration number to start statistics": f"{ndeb_stats + 1}"})
    param_ini = os.path.join(sim_outdir, "param.ini")
    if computation_type == "steady":
        # if steady computation did not fully converge
        if config["max_niter_steady_reached"]:
            niter_stats = convergence_criteria.get("niter_stats_steady", 10000)
        else:
            niter_stats = 1
    elif computation_type == "unsteady":
        niter_stats = 999999
        # add frequency for QoI convergence check: will ask MUSICAA to output stats files
        niter_ftt = get_niter_ftt(sim_outdir, config["gmsh"]["chord_length"])
        freqs = read_next_line_in_file(param_ini,
                                       "Output frequencies: screen / stats / fields").split()
        args.update({"Output frequencies: screen / stats / fields":
                     f"{freqs[0]} {freqs[1]} {niter_ftt}"})
    args.update({"Max number of temporal iterations": f"{niter_stats} 3000.0"})
    custom_input(param_ini, args)


def change_dimensions_param_blocks(sim_outdir: str):
    """
    **Modifies** the param_blocks.ini file of a given simulation to 2D.
    """
    # file and block info
    block_info = get_block_info(sim_outdir)
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


def pre_process_init_unsteady(dimension: str, sim_outdir: str):
    """
    **Pre-processes** unsteady computation to initialize with 2D or 3D simulation.

    Note: it is assumed that param.ini corresponds to the LES (i.e. 3D) input
          and param_blocks.ini corresponds to the 2D input.
    """
    args = {}
    param_ini = os.path.join(sim_outdir, "param.ini")
    param_blocks_ini = os.path.join(sim_outdir, "param_blocks.ini")
    if dimension == "2D":
        # save original inputs
        param_ini_copy = os.path.join(sim_outdir, "param.ini_3D")
        param_blocks_ini_copy = os.path.join(sim_outdir, "param_blocks.ini_2D")
        cp_filelist([param_ini, param_blocks_ini], [param_ini_copy, param_blocks_ini_copy])
        # modify param.ini file
        args.update({"Implicit Residual Smoothing": "2"})
        args.update({"Residual smoothing parameter": "0.42 0.1 0.005 0.00025 0.0000125"})
        is_SF = read_next_line_in_file(param_ini, ("Selective Filtering: is_SF "
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
        custom_input(param_ini, args)
    else:
        # copy 3D inputs
        param_ini_copy = os.path.join(sim_outdir, "param.ini_3D")
        param_blocks_ini_copy = os.path.join(sim_outdir, "param_blocks.ini_3D")
        cp_filelist([param_ini_copy, param_blocks_ini_copy], [param_ini, param_blocks_ini])
        # change to restart
        args.update({"from_interp": "2"})
        custom_input(param_ini, args)


def main() -> int:
    """
    This program runs a WOLF CFD simulation at ADP.
    In multisimulation mode, OP1 (+5°) and OP2 (-5°) simulations are also executed.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ms", type=int, help="number of parallel simulations", default=-1)
    parser.add_argument("-adp", action="store_true", help="simulate ADP")
    parser.add_argument("-op1", action="store_true", help="simulate OP1")
    parser.add_argument("-op2", action="store_true", help="simulate OP2")

    args_parse = parser.parse_args()
    t0 = time.time()
    print(f"simulations performed with: {args_parse}\n")

    # read config file
    with open("sim_config.json") as jfile:
        config = json.load(jfile)
    # set computation type
    computation_type: str = read_next_line_in_file("param.ini", "DES without subgrid")
    computation_type = "unsteady" if computation_type == "N" else "steady"
    execute_computation: Callable = globals()[f"execute_{computation_type}"]

    if args_parse.ms > 0:
        # tracking variables
        l_proc = []
        # process cmd line
        exec_args = sys.argv
        ms_idx = exec_args.index("-ms")
        del exec_args[ms_idx + 1]
        del exec_args[ms_idx]
        # adp
        exec_cmd = [sys.executable] + exec_args + ["-adp"]
        l_proc.append(submit_popen_process("ADP", exec_cmd))
        # op1
        exec_cmd = [sys.executable] + exec_args + ["-op1"]
        if len(l_proc) < args_parse.ms:
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        else:
            wait_for_it(l_proc, args_parse.ms)
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        # op2
        exec_cmd = [sys.executable] + exec_args + ["-op2"]
        if len(l_proc) < args_parse.ms:
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        else:
            wait_for_it(l_proc, args_parse.ms)
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        # wait for all processes to finish
        wait_for_it(l_proc, 1)

    if args_parse.adp:
        print("** ADP SIMULATION **")
        print("** -------------- **")
        sim_dir = "ADP"
        shutil.rmtree(sim_dir, ignore_errors=True)
        os.mkdir(sim_dir)
        args = {}
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_ADP = config.copy()
        execute_computation(config_ADP, sim_dir)

    if args_parse.op1:
        print("** OP1 SIMULATION (+5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP1"
        os.mkdir(sim_dir)
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        args = {}
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        # change flow angle
        args.update({"Flow angles": "48. 0."})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_OP1 = config.copy()
        execute_computation(config_OP1, sim_dir)

    if args_parse.op2:
        print("** OP2 SIMULATION (-5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP2"
        os.mkdir(sim_dir)
        # add simulation files
        cp_filelist(config["simulator"]["cp_list"], [sim_dir] * len(config["simulator"]["cp_list"]))
        args = {}
        # specify path for mesh files
        old_dir_grid = read_next_line_in_file("param.ini", "Directory for grid files")[1:-1]
        dir_grid = "'" + os.path.join("../", old_dir_grid) + "'"
        args.update({"Directory for grid files": dir_grid})
        # change flow angle
        args.update({"Flow angles": "38. 0."})
        param_ini = os.path.join(sim_dir, "param.ini")
        custom_input(param_ini, args)
        # execute computation
        config_OP2 = config.copy()
        execute_computation(config_OP2, sim_dir)

    print(f"INFO -- simulations finished successfully in {time.time() - t0} seconds.")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
