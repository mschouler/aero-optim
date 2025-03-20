import logging
import subprocess
import os
import numpy as np
import scipy.interpolate as si
import re
import pandas as pd
from typing import Callable
import signal
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)
from aero_optim.optim.evolution import PymooEvolution
from aero_optim.optim.optimizer import WolfOptimizer
from aero_optim.optim.pymoo_optimizer import PymooWolfOptimizer
from aero_optim.simulator.simulator import WolfSimulator
from aero_optim.utils import (custom_input, find_closest_index, check_dir,
                              read_next_line_in_file, cp_filelist)

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

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
    with open(os.path.join(sim_outdir, "info.ini"), "r") as f:
        lines = f.readlines()
    sim_info["nbloc"] = int(lines[0].split()[4])
    for bl in range(sim_info["nbloc"]):
        bl += 1
        sim_info[f"block_{bl}"] = {}
        sim_info[f"block_{bl}"]["nx"] = int(lines[bl].split()[5])
        sim_info[f"block_{bl}"]["ny"] = int(lines[bl].split()[6])
        sim_info[f"block_{bl}"]["nz"] = int(lines[bl].split()[7])
    sim_info["ngh"] = int(lines[bl + 8].split()[-1])
    sim_info["dt"] = float(lines[bl + 9].split()[-1])

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
    m_bar = np.mean(data["rhou_interp"])
    v_bar = np.mean(data["rho*uv_interp"]) / m_bar
    w_bar = np.mean(data["rho*uw_interp"]) / m_bar
    vv_bar = v_bar**2
    ww_bar = w_bar**2

    # conservation of momentum
    x_mom = np.mean(data["rho*uu_interp"] + data["p_interp"])
    y_mom = np.mean(data["rho*uv_interp"])
    z_mom = np.mean(data["rho*uw_interp"])

    # conservation of energy
    gam = data["gam"]
    R = data["R"]
    e = data["R"] * data["gam"] / (data["gam"] - 1) *\
        np.mean(data["rhou_interp"] * data["T_interp"]) +\
        0.5 * np.mean(data["rhou_interp"] * (data["uu_interp"]
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
    var_list_interp = ["uu", "vv", "ww", "rhou", "rho*uu", "rho*uv", "rho*uw", "p", "T"]
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


def args_LossCoef(sim_outdir: str, config: dict) -> dict:
    """
    **Returns** a dictionary containing the required arguments for LossCoef.
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


def LossCoef(sim_outdir: str, args: dict) -> float:
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


# class CustomSimulator(Simulator):
class CustomSimulator(WolfSimulator):
    """
    This class implements a simulator for the CFD code MUSICAA.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfSimulator object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - sim_pro (list[tuple[dict, subprocess.Popen[str]]]): list to track simulations
          and their associated subprocess.

            It has the following form:
            ({'gid': gid, 'cid': cid, 'meshfile': meshfile, 'restart': restart}, subprocess).

        # Simulation related
        # ------------------
        - sim_pro (list[tuple[dict, subprocess.Popen[str]]]): list containing current running
          processes.
        - restart (int): how many times a simulation is allowed to be restarted in case of failure.
        - CFL (float): CFL number read from the param.ini file in the working directory.
        - lower_CFL (float): lower CFL number in case computation crashes, computed with
                             lower_CFL = CFL / divide_CFL_by in config.json file.
        - ndeb_RANS (int): iteration from which the RANS model is computed.
        - nprint (int): every iteration at which MUSICAA prints information to screen. Used for
                        the residuals frequency !!! hard-coded in MUSICAA !!!
        - computation_type (str): type of computation (steady/unsteady)

        # Convergence criteria
        # --------------------
        - residual_convergence_order (float): order of convergence required for steady computations.
        - QoIs_convergence_order (float): order of convergence required for unsteady computation
                                          QoIs statistics.
        - unsteady_convergence_percent (float): percentage of variation allowed for mean and rms
                                                 to find end of unsteady transient.
        - unsteady_convergence_percent_mean (float): same for mean
        - unsteady_convergence_percent_rms (float): same for rms
        - Boudet_criterion_type (str): the type of the unsteady transient convergence criterion.
        - nb_ftt_before_criterion (int): number of flow-through times (f.t.t) to pass before
                                         computing the Boudet criterion
        - nb_ftt_mov_avg (int): number of flow-through times (f.t.t) to perform Boudet convergence
                                criterion moving average on.
        - only_compute_mean_crit (bool): if True, only the mean Boudet criterion is computed to
                                         asses transient convergence. This is the case when
                                         initializing the 2D computation since rms makes no sense

        # Mesh related
        # ------------
        - sim_info (dict): dictionary containing information about the computations.
                           Only accessible once the simulation has at least started or finished.
        """
        super().__init__(config)
        self.config: dict = config
        # simulator related
        self.sim_pro: list[tuple[dict, subprocess.Popen[str]]] = []
        self.restart: int = config["simulator"].get("restart", 0)
        self.CFL: float = float(read_next_line_in_file("param.ini", "CFL"))
        self.lower_CFL: float = self.CFL
        self.ndeb_RANS: int = int(read_next_line_in_file("param.ini",
                                                         "ndeb_RANS"))
        self.nprint: int = int(re.findall(r'\b\d+\b',
                                          read_next_line_in_file("param.ini",
                                                                 "screen"))[0])
        self.computation_type: str = read_next_line_in_file("param.ini",
                                                            "DES without subgrid")
        self.computation_type = "unsteady" if self.computation_type == "N" else "steady"
        # convergence criteria
        self.residual_convergence_order: float \
            = config["simulator"]["convergence_criteria"].get("residual_convergence_order", 4)
        self.QoIs_convergence_order: float \
            = config["simulator"]["convergence_criteria"].get("QoIs_convergence_order", 4)
        self.unsteady_convergence_percent_mean: float \
            = config["simulator"]["convergence_criteria"].get("unsteady_convergence_percent_mean",
                                                              1)
        self.unsteady_convergence_percent_rms: float \
            = config["simulator"]["convergence_criteria"].get("unsteady_convergence_percent_rms",
                                                              10)
        self.Boudet_criterion_type: bool \
            = config["simulator"]["convergence_criteria"].get("Boudet_criterion_type", "original")
        self.nb_ftt_before_criterion: int \
            = config["simulator"]["convergence_criteria"].get("nb_ftt_before_criterion", 10)
        self.nb_ftt_mov_avg: int \
            = config["simulator"]["convergence_criteria"].get("nb_ftt_mov_avg", 4)
        self.max_niter_init_2D: int \
            = config["simulator"]["convergence_criteria"].get("max_niter_init_2D", 999999)
        self.max_niter_init_3D: int \
            = config["simulator"]["convergence_criteria"].get("max_niter_init_3D", 999999)
        self.max_niter_stats: int \
            = config["simulator"]["convergence_criteria"].get("max_niter_stats", 999999)
        self.only_compute_mean_crit = False
        # mesh related
        self.block_info: dict = {}

    def process_config(self):
        """
        **Makes sure** the config file contains the required information and extracts it.
        """
        logger.debug("processing config..")
        if "exec_cmd" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <exec_cmd> entry in {self.config['simulator']}")
        if "ref_input" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <ref_input> entry in {self.config['simulator']}")
        if "post_process" not in self.config["simulator"]:
            logger.debug(f"no <post_process> entry in {self.config['simulator']}")

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
        full_meshfile = meshfile if meshfile else self.config["simulator"]["file"]
        path_to_meshfile: str = "/".join(full_meshfile.split("/")[:-1])
        meshfile = full_meshfile.split("/")[-1]

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
        custom_input(os.path.join(sim_outdir,
                                  "param.ini"), args)

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
            "simulator": self.config["simulator"],
        }
        with open(os.path.join(sim_outdir, "sim_config.json"), "w") as jfile:
            json.dump(sim_config, jfile)

        return sim_outdir, self.exec_cmd

    def execute(
            self,
            sim_outdir: str,
            exec_cmd: list[str],
            gid: int,
            cid: int,
            meshfile: str,
            restart: int
    ):
        """
        **Submits** the simulation subprocess and **updates** sim_pro.
        """
        # move to the output directory, execute musicaa and move back to the main directory
        os.chdir(sim_outdir)
        with open(f"{self.solver_name}_g{gid}_c{cid}.out", "wb") as out:
            with open(f"{self.solver_name}_g{gid}_c{cid}.err", "wb") as err:
                logger.info((f"execute {self.computation_type} simulation "
                             f"g{gid}, c{cid} with {self.solver_name}"))
                proc = subprocess.Popen(exec_cmd,
                                        env=os.environ,
                                        stdin=subprocess.DEVNULL,
                                        stdout=out,
                                        stderr=err,
                                        universal_newlines=True)
        os.chdir(self.cwd)
        # append simulation to the list of active processes
        self.sim_pro.append(
            ({"gid": gid, "cid": cid, "meshfile": meshfile, "restart": restart}, proc)
        )

    def _post_process(self, dict_id: dict, sim_outdir: str) -> str:
        """
        **Post-processes** the results of a terminated simulation.
        **Returns** the extracted results in a DataFrame.
        """
        qty_list: list[list[float]] = []
        head_list: list[str] = []
        # loop over the post-processing arguments to extract from the results
        for qty in self.post_process_args["outputs"]:
            # check if the method for computing qty exists
            try:
                # get arguments
                get_args: Callable = globals()[f"args_{qty}"]
                args = get_args(sim_outdir, self.config)
                get_value: Callable = globals()[qty]
                value = get_value(sim_outdir, args)
            except AttributeError:
                raise Exception(f"ERROR -- method for computing {qty} does not exist")
            try:
                # compute simulation results
                qty_list.append(value)
                head_list.append(qty)
            except Exception as e:
                logger.warning(f"could not compute {qty} in {sim_outdir}")
                logger.warning(f"exception {e} was raised")
        # pd.Series allows columns of different lengths
        df = pd.DataFrame({head_list[i]: pd.Series(qty_list[i]) for i in range(len(qty_list))})
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

    def get_sim_outdir(self, gid: int = 0, cid: int = 0) -> str:
        """
        **Returns** the path to the folder containing the simulation results.
        """
        return os.path.join(
            self.outdir, f"{self.solver_name.upper()}",
            f"{self.solver_name}_g{gid}_c{cid}"
        )


class CustomOptimizer(PymooWolfOptimizer):
    def __init__(self, config: dict):
        """
         **Inner**

        - feasible_cid (dict[int, list[int]]): dictionary containing feasible cid of each gid.
        """
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=2, n_ieq_constr=4, xl=self.bound[0], xu=self.bound[1]
        )
        self.feasible_cid: dict[int, list[int]] = {}

    def set_inner(self):
        """
        **Sets** some baseline quantities required to compute the relative constraints:

        - bsl_w_ADP (float)
        - bsl_w_OP (float)
        - bsl_camber_th (tuple[np.ndarray, float, float, np.ndarray])
        - bsl_area (float)
        - bsl_c (float)
        - bsl_c_ax (float)
        - bsl_cog (np.ndarray)
        - bsl_cog_x (float)
        - constraint (bool): whether to apply constraints (True) or not (False).
        """
        self.bsl_w_ADP = self.config["optim"].get("baseline_w_ADP", 0.03161)
        self.bsl_w_OP = self.config["optim"].get("baseline_w_OP", 0.03756)
        bsl_pts = self.ffd.pts
        self.bsl_c, self.bsl_c_ax = get_chords(bsl_pts)
        logger.info(f"baseline chord = {self.bsl_c} m, baseline axial chord = {self.bsl_c_ax}")
        bsl_upper, bsl_lower = split_profile(bsl_pts)
        self.bsl_camber_th = get_camber_th(bsl_upper, bsl_lower, interpolate=True)
        self.bsl_th_over_c = self.bsl_camber_th[1] / self.bsl_c
        self.bsl_Xth_over_cax = self.bsl_camber_th[2] / self.bsl_c_ax
        logger.info(f"baseline th_max = {self.bsl_camber_th[1]} m, "
                    f"Xth_max {self.bsl_camber_th[2]} m, "
                    f"th_max / c = {self.bsl_th_over_c}, "
                    f"Xth_max / c_ax = {self.bsl_Xth_over_cax}")
        self.bsl_area = get_area(bsl_pts)
        self.bsl_area_over_c2 = self.bsl_area / self.bsl_c**2
        logger.info(f"baseline area = {self.bsl_area} m2, "
                    f"baseline area / (c * c) = {self.bsl_area_over_c2}")
        self.bsl_cog = get_cog(bsl_pts)
        self.bsl_Xcg_over_cax = self.bsl_cog[0] / self.bsl_c_ax
        logger.info(f"baseline X_cg over c_ax = {self.bsl_Xcg_over_cax}")
        self.constraint: bool = self.config["optim"].get("constraint", True)

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Computes** the objective function and constraints for each candidate in the generation.

        Note:
            for this use-case, constraints can be computed before simulations.
            Unfeasible candidates are not simulated.
        """
        gid = self.gen_ctr
        self.feasible_cid[gid] = []

        # compute candidates constraints and execute feasible candidates only
        out["G"] = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        for cid in range(len(X)):
            if cid in self.feasible_cid[gid]:
                loss_ADP = self.simulator.df_dict[gid][cid]["ADP"][self.QoI].iloc[-1]
                loss_OP1 = self.simulator.df_dict[gid][cid]["OP1"][self.QoI].iloc[-1]
                loss_OP2 = self.simulator.df_dict[gid][cid]["OP2"][self.QoI].iloc[-1]
                logger.info(f"g{gid}, c{cid}: "
                            f"w_ADP = {loss_ADP}, w_OP = {0.5 * (loss_OP1 + loss_OP2)}")
                self.J.append([loss_ADP, 0.5 * (loss_OP1 + loss_OP2)])
            else:
                self.J.append([float("nan"), float("nan")])

        out["F"] = np.row_stack(self.J[-self.doe_size:])
        self._observe(out["F"])
        self.gen_ctr += 1

    def execute_constrained_candidates(self, candidates: np.ndarray, gid: int) -> np.ndarray:
        """
        **Executes** feasible candidates only and **waits** for them to finish.
        """
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])
        constraint = []
        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            ffd_file, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)
            logger.info(f"candidate g{gid}, c{cid} constraint computation..")
            constraint.append(self.apply_candidate_constraints(ffd_profile, gid, cid))
            # only mesh and execute feasible candidates
            if len([v for v in constraint[cid] if v > 0.]) == 0:
                self.feasible_cid[gid].append(cid)
                # meshing with proper sigint management
                # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/842
                ORIGINAL_SIGINT_HANDLER = signal.signal(signal.SIGINT, signal.SIG_DFL)
                mesh_file = self.mesh(ffd_file)
                signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
                while self.simulator.monitor_sim_progress() * self.nproc_per_sim >= self.budget:
                    time.sleep(1)
                self.simulator.execute_sim(meshfile=mesh_file, gid=gid, cid=cid)
            else:
                logger.info(f"unfeasible candidate g{gid}, c{cid} not simulated")

        # wait for last candidates to finish
        while self.simulator.monitor_sim_progress() > 0:
            time.sleep(0.1)
        return np.row_stack(constraint)

    def apply_candidate_constraints(self, profile: np.ndarray, gid: int, cid: int) -> list[float]:
        """
        **Computes** various relative and absolute constraints of a given candidate
        and **returns** their values as a list of floats.

        Note:
            when some constraint is violated, a graph is also generated.
        """
        if not self.constraint:
            return [-1.] * 4
        # relative constraints
        # thmax / c:        +/- 30%
        # Xthmax / c_ax:    +/- 20%
        upper, lower = split_profile(profile)
        c, c_ax = get_chords(profile)
        camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
        th_over_c = thmax / c
        Xth_over_cax = Xthmax / c_ax
        logger.debug(f"th_max = {thmax} m, Xth_max {Xthmax} m")
        logger.debug(f"th_max / c = {th_over_c}, Xth_max / c_ax = {Xth_over_cax}")
        th_cond = abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3
        logger.debug(f"th_max / c: {'violated' if th_cond > 0 else 'not violated'} ({th_cond})")
        Xth_cond = abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2
        logger.debug(f"Xth_max / c_ax: {'violated' if Xth_cond > 0 else 'not violated'} "
                     f"({Xth_cond})")
        # area / (c * c):   +/- 20%
        area = get_area(profile)
        area_over_c2 = area / c**2
        area_cond = abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2
        logger.debug(f"area / (c * c): {'violated' if area_cond > 0 else 'not violated'} "
                     f"({area_cond})")
        # X_cg / c_ax:      +/- 20%
        cog = get_cog(profile)
        Xcg_over_cax = cog[0] / c_ax
        cog_cond = abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2
        logger.debug(f"X_cg / c_ax: {'violated' if cog_cond > 0 else 'not violated'} ({cog_cond})")
        # absolute constraints
        O_le, O_te = get_circle_centers(upper[:, :2], lower[:, :2])
        le_circle = get_circle(O_le, 0.005 * c)
        te_circle = get_circle(O_te, 0.005 * c)
        le_radius_cond = get_radius_violation(profile, O_le, 0.005 * c)
        logger.debug(f"le radius: {'violated' if le_radius_cond > 0 else 'not violated'} "
                     f"({le_radius_cond})")
        te_radius_cond = get_radius_violation(profile, O_te, 0.005 * c)
        logger.debug(f"te radius: {'violated' if te_radius_cond > 0 else 'not violated'} "
                     f"({te_radius_cond})")
        if cog_cond > 0:
            fig_name = os.path.join(self.figdir, f"profile_g{gid}_c{cid}.png")
            plot_profile(profile, cog, fig_name)
        if th_cond > 0 or Xth_cond > 0 or area_cond > 0:
            fig_name = os.path.join(self.figdir, f"sides_g{gid}_c{cid}.png")
            plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
        return [th_cond, Xth_cond, area_cond, cog_cond]

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

    def final_observe(self, best_candidates: np.ndarray):
        """
        **Plots** convergence progress by plotting the fitness values
        obtained with the successive generations.
        """
        logger.info(f"plotting populations statistics after {self.gen_ctr} generations..")

        # plot construction
        _, ax = plt.subplots(figsize=(8, 8))
        gen_fitness = np.row_stack(self.J)

        # plotting data
        cmap = mpl.colormaps[self.cmap].resampled(self.max_generations)
        colors = cmap(np.linspace(0, 1, self.max_generations))
        for gid in range(self.max_generations):
            ax.scatter(gen_fitness[gid * self.doe_size: (gid + 1) * self.doe_size][:, 0],
                       gen_fitness[gid * self.doe_size: (gid + 1) * self.doe_size][:, 1],
                       color=colors[gid], label=f"g{gid}")
        ax.scatter(self.bsl_w_ADP, self.bsl_w_OP, marker="*", color="red", label="baseline")
        sorted_idx = np.argsort(best_candidates, axis=0)[:, 0]
        ax.plot(best_candidates[sorted_idx, 0], best_candidates[sorted_idx, 1],
                color="black", linestyle="dashed", label="pareto estimate")
        ax.plot()
        ax.set_axisbelow(True)
        plt.grid(True, color="grey", linestyle="dashed")

        # legend and title
        ax.set_title(f"Optimization evolution ({self.gen_ctr} g. x {self.doe_size} c.)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('$w_\\text{ADP}$')
        ax.set_ylabel('$w_\\text{OP}$')

        # save figure as png
        fig_name = f"pymoo_optim_g{self.gen_ctr}_c{self.doe_size}.png"
        logger.info(f"saving {fig_name} to {self.outdir}")
        plt.savefig(os.path.join(self.outdir, fig_name), bbox_inches='tight')
        plt.close()


class CustomEvolution(PymooEvolution):
    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = NSGA2(
            pop_size=self.optimizer.doe_size,
            sampling=self.optimizer.generator._pymoo_generator(),
            **self.optimizer.ea_kwargs
        )

    def evolve(self):
        logger.info("EXECUTE CUSTOM EVOLVE")
        res = minimize(problem=self.optimizer,
                       algorithm=self.ea,
                       termination=get_termination("n_gen", self.optimizer.max_generations),
                       seed=self.optimizer.seed,
                       verbose=True)

        self.optimizer.final_observe(res.F)

        # output results
        best_QoI, best_cand = res.F, res.X
        np.set_printoptions(linewidth=np.nan)
        logger.info(f"optimal QoIs:\n{best_QoI}")
        logger.info(f"optimal candidates:\n{best_cand}")
