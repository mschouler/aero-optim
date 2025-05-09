import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import subprocess
import scipy.interpolate as si
import sys
import time

from aero_optim.simulator.simulator import WolfSimulator
from aero_optim.utils import (custom_input, find_closest_index, check_dir,
                              read_next_line_in_file, cp_filelist)

from typing import Callable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cascade_wolf_base.custom_cascade_wolf import CustomEvolution as WolfCustomEvolution # noqa
from cascade_wolf_base.custom_cascade_wolf import CustomOptimizer as WolfCustomOptimizer # noqa

logger = logging.getLogger(__name__)


def get_feos_info(sim_outdir: str) -> dict:
    """
    **Reads** the feos_*.ini file from MUSICAA.
    """
    # regular expression to match lines with a name and a corresponding value
    pattern = re.compile(r"([A-Za-z\s]+)\.{2,}\s*(\S+)")

    # iterate over each line and apply the regex pattern
    feos_info: dict = {}
    with open(os.path.join(sim_outdir, "feos_air.ini"), "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                key = match.group(1).strip()
                value = match.group(2)
                feos_info[key] = float(value)
    return feos_info


def get_sim_info(sim_outdir: str):
    """
    **Returns** a dictionnary containing relevant information on the mesh
    used by MUSICAA: number of blocks, block size, number of ghost points.

    Note: this element is automatically produced by MUSICAA, and need not be
          created manually.
    """
    # read relevant lines
    sim_info: dict = {}
    # info.ini creation may take time
    if not os.path.isfile(os.path.join(sim_outdir, 'info.ini')):
        logger.warning("info.ini not found")
        time.sleep(2.)
    with open(os.path.join(sim_outdir, "info.ini"), "r") as f:
        lines = f.readlines()
    try:
        sim_info["nbloc"] = int(lines[0].split()[4])
        for bl in range(sim_info["nbloc"]):
            bl += 1
            sim_info[f"block_{bl}"] = {}
            sim_info[f"block_{bl}"]["nx"] = int(lines[bl].split()[5])
            sim_info[f"block_{bl}"]["ny"] = int(lines[bl].split()[6])
            sim_info[f"block_{bl}"]["nz"] = int(lines[bl].split()[7])
        sim_info["ngh"] = int(lines[bl + 8].split()[-1])
        sim_info["dt"] = float(lines[bl + 9].split()[-1])
    except IndexError:
        logger.warning("info.ini incomplete")
        time.sleep(1.)
        return get_sim_info(sim_outdir)
    return sim_info


def get_time_info(sim_outdir: str) -> dict:
    """
    **Reads** the time.ini file from MUSICAA.
    """
    # regular expression to match lines with a name and a corresponding value
    pattern = re.compile(r"(\d{4}_\d{4})\s*=\s*(\d+)\s*([\d\.]+)\s*([\d\.E+-]+)")

    # iterate over each line and apply the regex pattern
    time_info: dict = {}
    with open(os.path.join(sim_outdir, "time.ini"), "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp = match.group(1)
                iter = int(match.group(2))
                cputot = float(match.group(3))
                time = float(match.group(4))

                # Store the extracted values in the dictionary
                time_info[timestamp] = {
                    'iter': iter,
                    'cputot': cputot,
                    'time': time
                }
        time_info["niter_total"] = iter
    return time_info


def get_niter_ftt(sim_outdir: str, L_ref: float) -> int:
    """
    **Returns** the number of iterations per flow-through time (ftt)
    """
    # f.t.t = L_ref/u_ref
    feos_info = get_feos_info(sim_outdir)
    Mach_ref = float(read_next_line_in_file("param.ini",
                                            "Reference Mach"))
    T_ref = float(read_next_line_in_file("param.ini",
                                         "Reference temperature"))
    c_ref = np.sqrt(feos_info["Equivalent gamma"] * feos_info["Gas constant"] * T_ref)
    u_ref = c_ref * Mach_ref
    Lgrid = float(read_next_line_in_file("param.ini",
                                         "Scaling value for the grid Lgrid"))
    if Lgrid != 0:
        L_ref = L_ref * Lgrid
    ftt = L_ref / u_ref
    sim_info = get_sim_info(sim_outdir)
    return int(ftt / sim_info["dt"])


def read_bl(sim_outdir: str, bl: int) -> tuple[np.ndarray, np.ndarray]:
    """
    **Reads** simulation grid coordinates.
    """
    # get sim_info
    sim_info = get_sim_info(sim_outdir)
    nx = sim_info[f"block_{bl}"]["nx"]
    ny = sim_info[f"block_{bl}"]["ny"]
    ngh = sim_info["ngh"]
    nx_ext = nx + 2 * ngh
    ny_ext = ny + 2 * ngh

    # read coordinates extended by ghost cells
    filename = os.path.join(sim_outdir, f"grid_bl{bl}_ngh{ngh}.bin")
    f = open(filename, "r")
    x = np.fromfile(f, dtype=("<f8"), count=nx_ext * ny_ext).reshape((nx_ext, ny_ext),
                                                                     order="F")
    y = np.fromfile(f, dtype=("<f8"), count=nx_ext * ny_ext).reshape((nx_ext, ny_ext),
                                                                     order="F")
    x = x[ngh:-ngh, ngh:-ngh]
    y = y[ngh:-ngh, ngh:-ngh]
    return x, y


def mixed_out(data: dict) -> dict:
    """
    **Computes** mixed-out quantities:
    see A. Prasad (2004): https://doi.org/10.1115/1.1928289
    """
    # conservation of mass
    m_bar = np.nanmean(data["rhou_interp"])
    v_bar = np.nanmean(data["rho*uv_interp"]) / m_bar
    w_bar = np.nanmean(data["rho*uw_interp"]) / m_bar
    vv_bar = v_bar**2
    ww_bar = w_bar**2

    # conservation of momentum
    x_mom = np.nanmean(data["rho*uu_interp"] + data["p_interp"])
    y_mom = np.nanmean(data["rho*uv_interp"])
    z_mom = np.nanmean(data["rho*uw_interp"])

    # conservation of energy
    gam = data["gam"]
    R = data["R"]
    e = data["R"] * data["gam"] / (data["gam"] - 1) *\
        np.nanmean(data["rhou_interp"] * data["T_interp"]) +\
        0.5 * np.nanmean(data["rhou_interp"] * (data["uu_interp"]
                                                + data["vv_interp"]
                                                + data["ww_interp"]))

    # quadratic equation
    Q = 1 / m_bar**2 * (1 - 2 * gam / (gam - 1))
    L = 2 / m_bar**2 * (gam / (gam - 1) * x_mom - x_mom)
    C = 1 / m_bar**2 * (x_mom**2 + y_mom**2 + z_mom**2) - 2 * e / m_bar

    # select subsonic root
    p_bar = (-L - np.sqrt(L**2 - 4 * Q * C)) / 2 / Q
    u_bar = (x_mom - p_bar) / m_bar
    V2_bar = u_bar**2 + vv_bar + ww_bar
    rho_bar = m_bar / u_bar
    T_bar = p_bar / rho_bar / R
    c_bar = np.sqrt(gam * R * T_bar)
    M_bar = np.sqrt(V2_bar) / c_bar
    p0_bar = p_bar * (1 + (gam - 1) / 2 * M_bar**2)**(gam / (gam - 1))

    # store
    mixed_out_state = {"p_bar": p_bar,
                       "rho_bar": rho_bar,
                       "T_bar": T_bar,
                       "V2_bar": V2_bar,
                       "M_bar": M_bar,
                       "p0_bar": p0_bar}

    return mixed_out_state


def read_stats_bl(sim_outdir: str, bl_list: list[int], var_list: list[str]) -> dict:
    """
    **Reads** statistics of MUSICAA computation block from stats*_bl*.bin.
    """
    # get simulation information
    sim_info = get_sim_info(sim_outdir)

    # list of variables in file
    vars1 = ["rho", "u", "v", "w", "p", "T", "rhou", "rhov", "rhow", "rhoe",
             "rho**2", "uu", "vv", "ww", "uv", "uw", "vw", "vT", "p**2", "T**2",
             "mu", "divloc", "divloc**2"]
    vars2 = ["e", "h", "c", "s", "M", "0.5*q", "g", "la", "cp", "cv",
             "prr", "eck", "rho*dux", "rho*duy", "rho*duz", "rho*dvx", "rho*dvy",
             "rho*dvz", "rho*dwx", "rho*dwy", "rho*dwz", "p*div", "rho*div", "b1",
             "b2", "b3", "rhoT", "uT", "vT", "e**2", "h**2", "c**2", "s**2",
             "qq/cc2", "g**2", "mu**2", "la**2", "cv**2", "cp**2", "prr**2", "eck**2",
             "p*u", "p*v", "s*u", "s*v", "p*rho", "h*rho", "T*p", "p*s", "T*s", "rho*s",
             "g*rho", "g*p", "g*s", "g*T", "g*u", "g*v", "p*dux", "p*dvy", "p*dwz",
             "p*duy", "p*dvx", "rho*div**2", "dux**2", "duy**2", "duz**2", "dvx**2",
             "dvy**2", "dvz**2", "dwx**2", "dwy**2", "dwz**2", "b1**2", "b2**2", "b3**2",
             "rho*b1", "rho*b2", "rho*b3", "rho*uu", "rho*vv", "rho*ww",
             "rho*T**2", "rho*b1**2", "rho*b2**2", "rho*b3**2", "rho*uv", "rho*uw",
             "rho*vw", "rho*vT", "rho*u**2*v", "rho*v**3", "rho*w**2*v", "rho*v**2*u",
             "rho*dux**2", "rho*dvy**2", "rho*dwz**2", "rho*duy*dvx", "rho*duz*dwx",
             "rho*dvz*dwy", "u**3", "p**3", "u**4", "p**4", "Frhou", "Frhov", "Frhow",
             "Grhov", "Grhow", "Hrhow", "Frhovu", "Frhouu", "Frhovv", "Frhoww",
             "Grhovu", "Grhovv", "Grhoww", "Frhou_dux", "Frhou_dvx", "Frhov_dux",
             "Frhov_duy", "Frhov_dvx", "Frhov_dvy", "Frhow_duz", "Frhow_dvz",
             "Frhow_dwx", "Grhov_duy", "Grhov_dvy", "Grhow_duz", "Grhow_dvz",
             "Grhow_dwy", "Hrhow_dwz", "la*dTx", "la*dTy", "la*dTz",
             "h*u", "h*v", "h*w", "rho*h*u", "rho*h*v", "rho*h*w", "rho*u**3",
             "rho*v**3", "rho*w**3", "rho*w**2*u",
             "h0", "e0", "s0", "T0", "p0", "rho0", "mut"]
    all_vars = [vars1, vars2]

    # loop over blocks
    data: dict = {}
    stats = 1
    for vars in all_vars:
        for bl in bl_list:
            # get filename
            filename = os.path.join(sim_outdir, f"stats{stats}_bl{bl}.bin")

            # get block dimensions
            nx = sim_info[f"block_{bl}"]["nx"]
            ny = sim_info[f"block_{bl}"]["ny"]

            # read and store
            if stats == 1:
                data[f"block_{bl}"] = {}
            logger.info(f"wait for {filename} to be produced..")
            while not os.path.isfile(filename):
                time.sleep(1.)
            logger.info(f"{filename} found")
            f = open(filename, "rb")
            dtype = np.dtype("f8")
            for var in vars:
                data[f"block_{bl}"][var] = np.fromfile(
                    f, dtype=dtype, count=nx * ny).reshape((nx, ny), order="F")
            f.close()

            # keep only those provided in var_list
            unwanted = set(data[f"block_{bl}"]) - set(var_list)
            for unwanted_key in unwanted:
                if unwanted_key != "x" and unwanted_key != "y":
                    del data[f"block_{bl}"][unwanted_key]

        stats += 1

    # add fluid properties
    feos_info = get_feos_info(sim_outdir)
    data["gam"] = feos_info["Equivalent gamma"]
    data["R"] = feos_info["Gas constant"]

    return data


def extract_measurement_line(sim_outdir: str,
                             bl_list: list[int],
                             lims: list[float]) -> dict:
    """
    **Extracts** data along measurement line.
    """
    # get time-averaged block data
    var_list_interp = ["uu", "vv", "ww", "rhou", "rhov", "rho*uu", "rho*uv", "rho*uw", "p", "T"]
    data = read_stats_bl(sim_outdir, bl_list, var_list_interp)

    # get coordinates
    for bl in bl_list:
        data[f"block_{bl}"]["x"], data[f"block_{bl}"]["y"] = read_bl(sim_outdir, bl)

    # interpolate all variables
    for var in var_list_interp:
        data[f"{var}_interp"] = line_interp(data, var, lims, bl_list)

    return data


def line_interp(data: dict, var: str, lims: list[float], bl_list: list[int]) -> np.ndarray:
    """
    **Interpolates** data along a line defined by lims.
    """
    # flatten data
    var_flat_ = []
    x_flat_ = []
    y_flat_ = []
    for bl in bl_list:
        var_flat_.append(data[f"block_{bl}"][f"{var}"].flatten())
        x_flat_.append(data[f"block_{bl}"]["x"].flatten())
        y_flat_.append(data[f"block_{bl}"]["y"].flatten())
    var_flat = np.hstack(var_flat_)
    x_flat = np.hstack(x_flat_)
    y_flat = np.hstack(y_flat_)

    # create line
    x1, y1 = lims[0], lims[1]
    x2, y2 = lims[2], lims[3]
    x_interp = np.linspace(x1, x2, 1000)
    y_interp = np.linspace(y1, y2, 1000)

    # interpolate
    var_interp = si.griddata((x_flat, y_flat), var_flat,
                             (x_interp, y_interp), method="linear")

    return var_interp


def compute_QoIs(config: dict, sim_outdir: str) -> pd.DataFrame:
    """
    **Returns** the QoIs during computation in a DataFrame.
    """
    qty_list: list[list[float]] = []
    head_list: list[str] = []
    post_process_args: dict = config["simulator"]["post_process"]
    # loop over the post-processing arguments to extract from the results
    for qty in post_process_args["outputs"]:
        # check if the method for computing qty exists
        try:
            # get arguments
            get_args: Callable = globals()[f"args_{qty}"]
            args = get_args(sim_outdir, config)
            get_value: Callable = globals()[qty]
            value = get_value(sim_outdir, args)
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


def args_MixedoutLossCoef(sim_outdir: str, config: dict) -> dict:
    """
    **Returns** a dictionary containing the required arguments for MixedoutLossCoef.
    """
    args: dict = {}

    # inlet
    bl_list = config["gmsh"]["inlet_bl"]
    x1 = config["simulator"]["post_process"]["measurement_lines"]["inlet_x1"]
    x2 = config["simulator"]["post_process"]["measurement_lines"]["inlet_x2"]
    x, y = read_bl(sim_outdir, bl_list[0])
    closest_index = find_closest_index(x[:, 0], x1)
    y1 = y[closest_index, :].min()
    y2 = y1 + config["gmsh"]["pitch"]
    inlet_lims = [x1, y1, x2, y2]
    args["inlet_bl"] = bl_list
    args["inlet_lims"] = inlet_lims

    # outlet
    bl_list = config["gmsh"]["outlet_bl"]
    x1 = config["simulator"]["post_process"]["measurement_lines"]["outlet_x1"]
    x2 = config["simulator"]["post_process"]["measurement_lines"]["outlet_x2"]
    x, y = read_bl(sim_outdir, bl_list[0])
    closest_index = find_closest_index(x[:, 0], x1)
    y1 = y[closest_index, :].min()
    y2 = y1 + config["gmsh"]["pitch"]
    outlet_lims = [x1, y1, x2, y2]
    args["outlet_bl"] = bl_list
    args["outlet_lims"] = outlet_lims

    return args


def MixedoutLossCoef(sim_outdir: str, args: dict) -> float:
    """
    **Post-processes** the results of a terminated simulation.
    **Returns** the extracted results in a DataFrame.
    """
    # extract arguments
    inlet_bl = args["inlet_bl"]
    inlet_lims = args["inlet_lims"]
    outlet_bl = args["outlet_bl"]
    outlet_lims = args["outlet_lims"]

    # compute inlet mixed-out pressure
    inlet_data = extract_measurement_line(sim_outdir, inlet_bl, inlet_lims)
    inlet_mixed_out_state = mixed_out(inlet_data)

    # compute oulet mixed-out pressure
    outlet_data = extract_measurement_line(sim_outdir, outlet_bl, outlet_lims)
    outlet_mixed_out_state = mixed_out(outlet_data)

    return (inlet_mixed_out_state["p0_bar"] - outlet_mixed_out_state["p0_bar"]) /\
           (inlet_mixed_out_state["p0_bar"] - inlet_mixed_out_state["p_bar"])


def args_OutflowAngle(sim_outdir: str, config: dict) -> dict:
    return args_MixedoutLossCoef(sim_outdir, config)


def OutflowAngle(sim_outdir: str, args: dict) -> float:
    """
    **Post-processes** the results of a terminated simulation.
    **Returns** the extracted results in a DataFrame.
    """
    # extract arguments
    outlet_bl = args["outlet_bl"]
    outlet_lims = args["outlet_lims"]

    # compute oulet mixed-out pressure
    outlet_data = extract_measurement_line(sim_outdir, outlet_bl, outlet_lims)
    outflow_angle = np.nanmean(np.arctan(outlet_data["rhov_interp"] / outlet_data["rhou_interp"]))

    return outflow_angle / np.pi * 180


# class CustomSimulator(Simulator):
class CustomSimulator(WolfSimulator):
    """
    This class implements a simulator for the CFD code MUSICAA.
    """
    def process_config(self):
        """
        **Makes sure** the config file contains the required information and extracts it:

        - computation_type (str): type of computation (steady/unsteady)
        """
        logger.debug("processing config..")
        if "exec_cmd" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <exec_cmd> entry in {self.config['simulator']}")
        if "ref_input" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <ref_input> entry in {self.config['simulator']}")
        if "post_process" not in self.config["simulator"]:
            logger.debug(f"no <post_process> entry in {self.config['simulator']}")
        self.computation_type = read_next_line_in_file("param.ini", "DES without subgrid")
        self.computation_type = "unsteady" if self.computation_type == "N" else "steady"

    def set_solver_name(self):
        """
        **Sets** the solver name to musicaa.
        """
        self.solver_name = "musicaa"

    def pre_process(self, meshfile: str, gid: int, cid: int) -> tuple[str, list[str]]:
        """
        **Pre-processes** the simulation execution
        and **returns** the execution command and directory.
        """
        # get the simulation meshfile
        if meshfile:
            full_meshfile = meshfile
            path_to_meshfile = "/".join(full_meshfile.split("/")[:-1])
            meshfile = full_meshfile.split("/")[-1]
        else:
            path_to_meshfile = self.config["gmsh"]["mesh_dir"]
            meshfile = self.config["gmsh"]["mesh_name"]

        # name of simulation directory
        sim_outdir = self.get_sim_outdir(gid=gid, cid=cid)
        check_dir(sim_outdir)

        # copy files and executable to directory
        cp_filelist(self.config["simulator"]["cp_list"],
                    [sim_outdir] * len(self.config["simulator"]["cp_list"]))
        logger.info((f"param.ini and "
                     f"param_blocks.ini "
                     f"copied to {sim_outdir}"))

        # modify solver input file: delete half-cell at block boundaries
        args: dict = {}
        args.update({"from_interp": "0"})
        args.update({"Max number of temporal iterations": "9999999 3000.0"})
        args.update({"Iteration number to start statistics": "9999999"})
        args.update({"Half-cell": "T", "Coarse grid": "F 0", "Perturb grid": "F"})
        args.update({"Directory for grid files":
                     f"'{os.path.relpath(path_to_meshfile, sim_outdir)}'"})
        args.update({"Name for grid files": meshfile})
        if self.computation_type == "steady":
            args.update({"Compute residuals": "T"})
        custom_input(os.path.join(sim_outdir, "param.ini"), args)

        # execute MUSICAA to delete half-cell
        os.chdir(sim_outdir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"].split()
        with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.out", "wb") as out:
            with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.err", "wb") as err:
                logger.info(f"delete mesh half-cell for g{gid}, c{cid} with {self.solver_name}")
                subprocess.run(preprocess_cmd,
                               env=os.environ,
                               stdin=subprocess.DEVNULL,
                               stdout=out,
                               stderr=err,
                               universal_newlines=True)

        # modify solver input file: mode from_scratch
        args.update({"from_interp": "1"})
        custom_input("param.ini", args)
        logger.info(f"changed execution mode to 1 in {sim_outdir}")
        os.chdir(self.cwd)

        # create local config file
        sim_config = {
            "gmsh": self.config["gmsh"],
            "optim": self.config["optim"],
            "simulator": self.config["simulator"],
        }
        with open(os.path.join(sim_outdir, "sim_config.json"), "w") as jfile:
            json.dump(sim_config, jfile)

        return sim_outdir, self.exec_cmd

    def _post_process(self, dict_id: dict, sim_outdir: str) -> str:
        """
        **Post-processes** the results of a terminated simulation.
        **Returns** the extracted results in a DataFrame.
        """
        df = compute_QoIs(self.config, sim_outdir)
        logger.info(
            f"g{dict_id['gid']}, c{dict_id['cid']} converged in {len(df)} it."
        )
        logger.info(f"last values:\n{df.tail(n=1).to_string(index=False)}")
        return df

    def post_process(self, dict_id: dict, sim_out_dir: str) -> dict[str, pd.DataFrame]:
        """
        **Post-processes** the results of a terminated triple simulation.</br>
        **Returns** the extracted results in a dictionary of DataFrames.

        Note:
            there are two oIoIs: loss_ADP and loss_OP = 1/2(loss_OP1 + loss_OP2)
            to be extracted from: sim_out_dir/ADP, sim_out_dir/OP1  and sim_out_dir/OP2
        """
        df_sub_dict: dict[str, pd.DataFrame] = {}
        for fname in ["ADP", "OP1", "OP2"]:
            logger.debug(f"post_process g{dict_id['gid']}, c{dict_id['cid']} {fname}..")
            df_sub_dict[fname] = self._post_process(dict_id, os.path.join(sim_out_dir, fname))
        return df_sub_dict


class CustomOptimizer(WolfCustomOptimizer):
    def _observe(self, pop_fitness: np.ndarray):
        """
        **Plots** some results each time a generation has been evaluated:</br>
        > the simulations residuals,</br>
        > the candidates fitnesses,</br>
        > the baseline and deformed profiles.
        """
        gid = self.gen_ctr

        # plot settings
        baseline: np.ndarray = self.ffd.pts
        profiles: list[np.ndarray] = self.ffd_profiles[gid]
        res_dict = self.simulator.df_dict[gid]
        df_key = res_dict[self.feasible_cid[gid][0]]["ADP"].columns  # ResTot, LossCoef, x, y, Mis
        cmap = mpl.colormaps[self.cmap].resampled(self.doe_size)
        colors = cmap(np.linspace(0, 1, self.doe_size))
        # subplot construction
        fig = plt.figure(figsize=(16, 16))
        ax1 = plt.subplot(2, 1, 1)  # profiles
        ax2 = plt.subplot(2, 3, 4)  # loss_ADP
        ax3 = plt.subplot(2, 3, 5)  # loss_OP
        ax4 = plt.subplot(2, 3, 6)  # fitness (loss_ADP vs loss_OP)
        plt.subplots_adjust(wspace=0.25)
        ax1.plot(baseline[:, 0], baseline[:, 1], color="k", lw=2, ls="--", label="baseline")
        # loop over candidates through the last generated profiles
        for cid in self.feasible_cid[gid]:
            ax1.plot(profiles[cid][:, 0], profiles[cid][:, 1], color=colors[cid], label=f"c{cid}")
            res_dict[cid]["ADP"][df_key[0]].plot(ax=ax2, color=colors[cid], label=f"c{cid}")
            vsize = min(len(res_dict[cid]["OP1"][df_key[0]]), len(res_dict[cid]["OP2"][df_key[0]]))
            ax3.plot(
                range(vsize),
                0.5 * (res_dict[cid]["OP1"][df_key[0]].values[-vsize:]
                       + res_dict[cid]["OP2"][df_key[0]].values[-vsize:]),
                color=colors[cid],
                label=f"c{cid}"
            )
            ax4.scatter(pop_fitness[cid, 0], pop_fitness[cid, 1],
                        color=colors[cid], label=f"c{cid}")
        ax4.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="baseline")
        # legend and title
        fig.suptitle(
            f"Generation {gid} results", size="x-large", weight="bold", y=0.93
        )
        # top
        ax1.set_title("FFD profiles", weight="bold")
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        # bottom left
        ax2.set_title(f"{df_key[0]} ADP", weight="bold")
        ax2.set_xlabel('it. #')
        ax2.set_ylabel('$w_\\text{ADP}$')
        # bottom center
        ax3.set_title(f"{df_key[0]} OP", weight="bold")
        ax3.set_xlabel('it. #')
        ax3.set_ylabel('$w_\\text{OP}$')
        # bottom right
        ax4.set_title(f"{self.QoI} ADP vs {self.QoI} OP", weight="bold")
        ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax4.set_xlabel('$w_\\text{ADP}$')
        ax4.set_ylabel('$w_\\text{OP}$')
        # save figure as png
        fig_name = f"pymoo_g{gid}.png"
        logger.info(f"saving {fig_name} to {self.figdir}")
        plt.savefig(os.path.join(self.figdir, fig_name), bbox_inches='tight')
        plt.close()


class CustomEvolution(WolfCustomEvolution):
    """Same custom class as the one defined in cascade_adap"""
