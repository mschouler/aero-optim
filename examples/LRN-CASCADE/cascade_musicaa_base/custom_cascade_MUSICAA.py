import logging
import subprocess
import os
import numpy as np
import signal
import time
import shutil
import scipy.interpolate as si
import re

from aero_optim.utils import custom_input, find_closest_index
from aero_optim.mesh.mesh import MeshMusicaa
from aero_optim.simulator.simulator import Simulator
from aero_optim.optim.optimizer import Optimizer
from aero_optim.utils import from_dat, check_dir
from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle, get_circle_centers,
                             get_cog, get_radius_violation, split_profile, plot_profile, plot_sides)

"""
This script contains various customizations of the aero_optim module
to run with the solver MUSICAA.
"""

logger = logging.getLogger(__name__)


class CustomMesh(MeshMusicaa):
    """
    This class implements a mesh routine for a compressor cascade geometry when using MUSICAA.
    This solver requires strctured coincident blocks with a unique frontier on each boundary.
    """
    def __init__(self, config: dict):
        """
        Instantiates the CascadeMeshMusicaa object.

        - **kwargs: optional arguments listed upon call

        **Input**

        - config (dict): the config file dictionary.
        - datfile (str, optional): path/to/ffd_profile.dat file.

        **Inner**

        - pitch (float): blade-to-blade pitch
        - periodic_bl (list[int]): list containing the blocks that must be translated DOWN to reform
                        the blade geometry fully. Not needed if the original mesh already
                        surrounds the blade

        """
        super().__init__(config)

    def build_mesh(self):
        """
        **Orchestrates** the required steps to deform the baseline mesh using the new
        deformed profile for MUSICAA.
        """
        # read profile
        profile = from_dat(self.dat_file)

        # create musicaa_<outfile>_pert_bl*.x files
        mesh_dir = self.write_deformed_mesh_edges(profile, self.outdir)

        # deform mesh with MUSICAA
        self.deform_mesh(mesh_dir)

    def deform_mesh(self, mesh_dir: str):
        """
        **Executes** the MUSICAA mesh deformation routine.
        """
        args: dict = {}
        # set MUSICAA to pre-processing mode: 0
        args.update({"from_field": "0"})

        # set to perturb grid
        args.update({"Half-cell": "F", "Coarse grid": "F 0", "Perturb grid": "T"})

        # indicate in/output directory and name
        musicaa_mesh_dir = os.path.relpath(mesh_dir, self.dat_dir)
        args.update({"Directory for perturbed grid files": f"'{musicaa_mesh_dir}/'"})
        args.update({"Name for perturbed grid files", f"'{self.outfile}'"})

        # modify param.ini
        custom_input(self.config["simulator"]["ref_input_num"], args)

        # execute MUSICAA to deform mesh
        os.chdir(self.dat_dir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"]
        subprocess.Popen(preprocess_cmd, env=os.environ)
        os.chdir(self.cwd)


# class CustomOptimizer(Optimizer):
#     """
#     This class implements a Custom Optimizer.
#     """
#     def __init__(self, config: dict):
#         """
#         Instantiates the Optimizer object.

#         **Input**

#         - config (dict): the config file dictionary.
#         """
#         super().__init__(config)
#         self.feasible_cid: dict[int, list[int]] = {}

#     def set_simulator_class(self):
#         """
#         **Sets** the simulator class as custom.
#         """
#         super().set_simulator_class()

#     def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
#         """
#         **Computes** the objective function and constraints for each candidate in the generation.
#         """
#         gid = self.gen_ctr
#         self.feasible_cid[gid] = []

#         # compute candidates constraints and execute feasible candidates only
#         out["G"] = self.execute_constrained_candidates(X, gid)

#         # update candidates fitness
#         for cid in range(len(X)):
#             if cid in self.feasible_cid[gid]:
#                 loss_ADP = self.simulator.df_dict[gid][cid]["ADP"][self.QoI].iloc[-1]
#                 loss_OP1 = self.simulator.df_dict[gid][cid]["OP1"][self.QoI].iloc[-1]
#                 loss_OP2 = self.simulator.df_dict[gid][cid]["OP2"][self.QoI].iloc[-1]
#                 logger.info(f"g{gid}, c{cid}: "
#                             f"w_ADP = {loss_ADP}, w_OP = {0.5 * (loss_OP1 + loss_OP2)}")
#                 self.J.append([loss_ADP, 0.5 * (loss_OP1 + loss_OP2)])
#             else:
#                 self.J.append([float("nan"), float("nan")])

#         out["F"] = np.row_stack(self.J[-self.doe_size:])
#         self._observe(out["F"])
#         self.gen_ctr += 1

#     def execute_constrained_candidates(self, candidates: np.ndarray, gid: int) -> np.ndarray:
#         """
#         **Executes** feasible candidates only and **waits** for them to finish.
#         """
#         logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
#         self.ffd_profiles.append([])
#         self.inputs.append([])
#         constraint = []
#         for cid, cand in enumerate(candidates):
#             self.inputs[gid].append(np.array(cand))
#             ffd_file, ffd_profile = self.deform(cand, gid, cid)
#             self.ffd_profiles[gid].append(ffd_profile)
#             logger.info(f"candidate g{gid}, c{cid} constraint computation..")
#             constraint.append(self.apply_candidate_constraints(ffd_profile, gid, cid))
#             # only mesh and execute feasible candidates
#             if len([v for v in constraint[cid] if v > 0.]) == 0:
#                 self.feasible_cid[gid].append(cid)
#                 # meshing with proper sigint management
#                 # see https://gitlab.onelab.info/gmsh/gmsh/-/issues/842
#                 ORIGINAL_SIGINT_HANDLER = signal.signal(signal.SIGINT, signal.SIG_DFL)
#                 mesh_file = self.mesh(ffd_file)
#                 signal.signal(signal.SIGINT, ORIGINAL_SIGINT_HANDLER)
#                 while self.simulator.monitor_sim_progress() * self.nproc_per_sim >= self.budget:
#                     time.sleep(1)
#                 self.simulator.execute_sim(meshfile=mesh_file, gid=gid, cid=cid)
#             else:
#                 logger.info(f"unfeasible candidate g{gid}, c{cid} not simulated")

#         # wait for last candidates to finish
#         while self.simulator.monitor_sim_progress() > 0:
#             time.sleep(0.1)
#         return np.row_stack(constraint)

#     def apply_candidate_constraints(self, profile: np.ndarray, gid: int, cid: int) -> list[float]:
#         """
#         **Computes** various relative and absolute constraints of a given candidate
#         and **returns** their values as a list of floats.

#         Note:
#             when some constraint is violated, a graph is also generated.
#         """
#         if not self.constraint:
#             return [-1.] * 4
#         # relative constraints
#         # thmax / c:        +/- 30%
#         # Xthmax / c_ax:    +/- 20%
#         upper, lower = split_profile(profile)
#         c, c_ax = get_chords(profile)
#         camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
#         th_over_c = thmax / c
#         Xth_over_cax = Xthmax / c_ax
#         logger.debug(f"th_max = {thmax} m, Xth_max {Xthmax} m")
#         logger.debug(f"th_max / c = {th_over_c}, Xth_max / c_ax = {Xth_over_cax}")
#         th_cond = abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3
#         logger.debug(f"th_max / c: {'violated' if th_cond > 0 else 'not violated'} ({th_cond})")
#         Xth_cond = abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2
#         logger.debug(f"Xth_max / c_ax: {'violated' if Xth_cond > 0 else 'not violated'} "
#                      f"({Xth_cond})")
#         # area / (c * c):   +/- 20%
#         area = get_area(profile)
#         area_over_c2 = area / c**2
#         area_cond = abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2
#         logger.debug(f"area / (c * c): {'violated' if area_cond > 0 else 'not violated'} "
#                      f"({area_cond})")
#         # X_cg / c_ax:      +/- 20%
#         cog = get_cog(profile)
#         Xcg_over_cax = cog[0] / c_ax
#         cog_cond = abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2
#         logger.debug(f"X_cg / c_ax: {'violated' if cog_cond > 0 else 'not violated'} ({cog_cond})")
#         # absolute constraints
#         O_le, O_te = get_circle_centers(upper[:, :2], lower[:, :2])
#         le_circle = get_circle(O_le, 0.005 * c)
#         te_circle = get_circle(O_te, 0.005 * c)
#         le_radius_cond = get_radius_violation(profile, O_le, 0.005 * c)
#         logger.debug(f"le radius: {'violated' if le_radius_cond > 0 else 'not violated'} "
#                      f"({le_radius_cond})")
#         te_radius_cond = get_radius_violation(profile, O_te, 0.005 * c)
#         logger.debug(f"te radius: {'violated' if te_radius_cond > 0 else 'not violated'} "
#                      f"({te_radius_cond})")
#         if cog_cond > 0:
#             fig_name = os.path.join(self.figdir, f"profile_g{gid}_c{cid}.png")
#             plot_profile(profile, cog, fig_name)
#         if th_cond > 0 or Xth_cond > 0 or area_cond > 0:
#             fig_name = os.path.join(self.figdir, f"sides_g{gid}_c{cid}.png")
#             plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
#         return [th_cond, Xth_cond, area_cond, cog_cond]


class CustomSimulator(Simulator):
    """
    This class implements a simulator for the CFD code MUSICAA.
    """
    def __init__(self, config: dict):
        """
        Instantiates the MusicaaSimulator object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - sim_pro (list[tuple[dict, subprocess.Popen[str]]]): list to track simulations
          and their associated subprocess.

            It has the following form:
            ({'gid': gid, 'cid': cid, 'meshfile': meshfile, 'restart': restart}, subprocess).

        - restart (int): how many times a simulation is allowed to be restarted in case of failure.
        - mesh_info (dict): contains information on multi-block mesh used by MUSICAA
        """
        super().__init__(config)
        self.sim_pro: list[tuple[dict, subprocess.Popen[str]]] = []
        self.restart: int = config["simulator"].get("restart", 0)
        # get mesh info
        self.mesh_info: dict = self.get_mesh_info()

    def process_config(self):
        """
        **Makes sure** the config file contains the required information and extracts it.
        """
        logger.debug("processing config..")
        if "exec_cmd" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <exec_cmd> entry in {self.config['simulator']}")
        if "ref_input" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <ref_input> entry in {self.config['simulator']}")
        if "sim_args" not in self.config["simulator"]:
            logger.debug(f"no <sim_args> entry in {self.config['simulator']}")
        if "post_process" not in self.config["simulator"]:
            logger.debug(f"no <post_process> entry in {self.config['simulator']}")

    def set_solver_name(self):
        """
        **Sets** the solver name to musicaa.
        """
        self.solver_name = "musicaa"

    def execute_sim(self, meshfile: str = "", gid: int = 0, cid: int = 0, restart: int = 0):
        """
        **Pre-processes** and **executes** a MUSICAA simulation.
        """
        # add gid entry to the results dictionary
        if gid not in self.df_dict:
            self.df_dict[gid] = {}

        try:
            sim_outdir = self.get_sim_outdir(gid, cid)
            dict_id: dict = {"gid": gid, "cid": cid, "meshfile": meshfile}
            self.df_dict[dict_id["gid"]][dict_id["cid"]] = self.post_process(
                dict_id, sim_outdir
            )
            logger.info(f"g{gid}, c{cid}: loaded pre-existing results from files")
        except FileNotFoundError:
            # Pre-process
            sim_outdir = self.pre_process(meshfile, gid, cid)
            # Execution
            self.execute(sim_outdir, gid, cid, meshfile, restart)

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
        shutil.copy(self.config["simulator"]["ref_input_num"], sim_outdir)
        shutil.copy(self.config["simulator"]["ref_input_mesh"], sim_outdir)
        shutil.copy(self.solver_name, sim_outdir)
        logger.info(f"{self.config['simulator']['ref_input_num']},     \
                      {self.config['simulator']['ref_input_mesh']} and \
                      {self.solver_name} copied to {sim_outdir}")

        # modify solver input file: delete half-cell at block boundaries
        args: dict = {}
        args.update({"from_interp": "0", "Coarse grid": "F 0", "Perturb grid": "F"})
        args.update({"Half-cell": "T", "Coarse grid": "F 0", "Perturb grid": "F"})
        args.update({"Directory for grid files":
                     f"{os.path.relpath(path_to_meshfile, sim_outdir)}"})
        args.update({"Name for grid files": meshfile})
        custom_input(os.path.join(sim_outdir,
                                  self.config["simulator"]["ref_input_num"]).split("/")[-1], args)

        # execute MUSICAA to delete half-cell
        os.chdir(sim_outdir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"]
        subprocess.Popen(preprocess_cmd, env=os.environ)
        logger.info(f"Half-cell deleted in {sim_outdir}")

        # modify solver input file: mode from_scratch
        args.update({"from_interp": "1"})
        custom_input(self.config["simulator"]["ref_input_num"].split("/")[-1], args)
        logger.info(f"changed execution mode to 1 in {sim_outdir}")
        os.chdir(self.cwd)

        # copy any other solver expected files
        shutil.copy(self.config["simulator"]["ref_input_num_rans"], sim_outdir)
        shutil.copy(self.config["simulator"]["ref_input_feos"], sim_outdir)

        return sim_outdir

    def execute(
            self,
            sim_outdir: str,
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
                logger.info(f"execute simulation g{gid}, c{cid} with {self.solver_name}")
                proc = subprocess.Popen(self.exec_cmd,
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

    def monitor_sim_progress(self) -> str:
        """
        **Updates** the list of simulations under execution and **returns** its length.
        """
        # will call post-process if simulation ended, otherwise it waits
        logger.debug("Function 'monitor_sim_progress' not yet implemented")

    def post_process(self, dict_id: dict, sim_outdir: str) -> str:
        """
        **Post-processes** the results of a terminated simulation.
        Here, the mixed-out oultet pressure coefficient is the QoI.
        """
        # compute inlet mixed-out pressure
        inlet_data = self.extract_measurement_line(sim_outdir, "inlet")
        inlet_mixed_out_state = self.mixed_out(inlet_data)

        # compute oulet mixed-out pressure
        outlet_data = self.extract_measurement_line(sim_outdir, "outlet")
        outlet_mixed_out_state = self.mixed_out(outlet_data)

        # return pressure loss coefficient
        return (inlet_mixed_out_state["p0_bar"] - outlet_mixed_out_state["p0_bar"]) /\
            (inlet_mixed_out_state["p0_bar"] - inlet_mixed_out_state["p_bar"])

    def get_sim_outdir(self, gid: int = 0, cid: int = 0) -> str:
        """
        **Returns** the path to the folder containing the simulation results.
        """
        return os.path.join(
            self.outdir, f"{self.solver_name.upper()}",
            f"{self.solver_name}_g{gid}_c{cid}"
        )

    def read_bl(self, sim_outdir: str, bl: int) -> tuple[np.ndarray, np.ndarray]:
        """
        **Reads** simulation grid coordinates.
        """
        # get mesh dimensions
        nx = self.mesh_info[f"nx_bl{bl}"]
        ny = self.mesh_info[f"ny_bl{bl}"]
        ngh = self.mesh_info["ngh"]
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

    def mixed_out(self, data: dict) -> dict:
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
        feos_dict = self.get_feos_info()
        gam = feos_dict["Equivalent gamma"]
        R = feos_dict["Gas constant"]
        e = R * gam / (gam - 1) * np.mean(data["rhou_interp"] * data["T_interp"]) +\
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

    def kill_all(self):
        """
        **Kills** all active processes.
        """
        logger.debug("Function 'kill_all' not yet implemented")

    def read_stats_bl(self, sim_outdir: str, bl_list: list[str]) -> dict:
        """
        **Reads** statistics of MUSICAA computation block from stats*_bl*.bin.
        """
        data = {}

        # loop over blocks
        for bl in bl_list:
            # get filename
            filename = os.path.join(sim_outdir, f"stats1_bl{bl}.bin")
            f = open(filename, "r")

            # get block dimensions
            nx = self.mesh_info[f"nx_bl{bl}"]
            ny = self.mesh_info[f"ny_bl{bl}"]

            # list of variables in file
            var_list = ["rho", "u", "v", "w", "p", "T", "rhou", "rhov", "rhow", "rhoe",
                        "rho**2", "uu", "vv", "ww", "uv", "uw", "vw", "vT", "p**2", "T**2",
                        "mu", "divloc", "divloc**2"]

            # read and store
            data[f"block_{bl}"] = {}
            f = open(filename, "rb")
            dtype = np.dtype("f8")
            for var in var_list:
                data[f"block_{bl}"][var] = np.fromfile(
                    f, dtype=dtype, count=nx * ny).reshape((nx, ny), order="F")
            f.close()

        return data

    def extract_measurement_line(self, sim_outdir: str, location: str) -> dict:
        """
        Extract data along measurement line.
        """
        if location == "inlet":
            bl_list = self.config["mesh"]["inlet_bl"]
        else:
            bl_list = self.config["mesh"]["outlet_bl"]

        # get time-averaged block data
        data = self.read_stats_bl(sim_outdir, bl_list)

        # get coordinates
        for bl in bl_list:
            data[f"block_{bl}"]["x"], data[f"block_{bl}"]["y"] = self.read_bl(sim_outdir, bl)

        # data useful to compute mixed-out state
        var_list_interp = ["uu", "vv", "ww", "rhou", "rhouv", "rhouw", "p", "T"]
        unwanted = set(data) - set(var_list_interp)
        for unwanted_key in unwanted:
            del data[unwanted_key]

        # interpolation line definition
        x1 = self.config["simulator"]["post_process"][f"{location}_measurements_x1"]
        x2 = self.config["simulator"]["post_process"][f"{location}_measurements_x2"]
        closest_index = find_closest_index(data[f"block_{bl_list[0]}"]["x"][:, 0], x1)
        y1 = data[f"block_{bl_list[0]}"]["y"][closest_index, :].min()
        y2 = y1 + self.config["mesh"]["pitch"]
        lims = [x1, y1, x2, y2]

        # interpolate all variables
        for var in var_list_interp:
            data[f"{var}_interp"] = self.line_interp(data, var, lims, bl_list)

        return data

    def line_interp(self, data: dict, var: str, lims: list[float], bl_list: list[int]) -> tuple:
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

    def get_feos_info(self, sim_outdir: str) -> dict:
        """
        **Reads** the feos_*.ini file from MUSICAA.
        """
        # read file
        with open(os.path.join(sim_outdir, "feos_air.ini"), "r") as f:
            lines = f.readlines()

        # regular expression to match lines with a name and a corresponding value
        pattern = re.compile(r"([A-Za-z\s]+)\.{2,}\s*(\S+)")

        # iterate over each line and apply the regex pattern
        feos_dict = {}
        for match in pattern.finditer(lines):
            name = match.group(1).strip()
            value = match.group(2)
            feos_dict[name] = value

        return feos_dict

    def get_mesh_info(self) -> dict:
        """
        **Returns** a dictionnary containing relevant information on the mesh
        used by MUSICAA: number of blocks, block size, number of ghost points.
        Note: this element is automatically produced by MUSICAA, and need not be
        created manually.
        """
        mesh_info = {}

        # read relevant lines
        with open(os.path.join(self.dat_dir, "info.ini"), "r") as f:
            lines = f.readlines()
        mesh_info["nbloc"] = int(lines[0].split()[4])
        for ind in range(mesh_info["nbloc"]):
            mesh_info["nx_bl" + str(ind + 1)] = int(lines[1 + ind].split()[5])
            mesh_info["ny_bl" + str(ind + 1)] = int(lines[1 + ind].split()[6])
            mesh_info["nz_bl" + str(ind + 1)] = int(lines[1 + ind].split()[7])
        mesh_info["ngh"] = int(lines[ind + 8].split()[-1])

        return mesh_info
