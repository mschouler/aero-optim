import logging
import subprocess
import os
import numpy as np
import shutil
import scipy.interpolate as si
import re
import pandas as pd
from typing import Callable

from aero_optim.utils import (custom_input, find_closest_index, from_dat, check_dir,
                              read_next_line_in_file, round_number, cp_filelist, rm_filelist)
from aero_optim.mesh.mesh import MeshMusicaa
from aero_optim.simulator.simulator import Simulator

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
    def __init__(self, config: dict, just_get_block_info: bool = False):
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
        super().__init__(config, just_get_block_info=just_get_block_info)

    def build_mesh(self):
        """
        **Orchestrates** the required steps to deform the baseline mesh using the new
        deformed profile for MUSICAA.
        """
        # read profile
        profile = from_dat(self.dat_file)

        # create musicaa_<outfile>_bl*.x files
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
        args.update({"Directory for grid files": "'.'"})
        args.update({"Name for grid files": self.config['plot3D']['mesh']['mesh_name']})
        args.update({"Directory for perturbed grid files": f"'{musicaa_mesh_dir}'"})
        args.update({"Name for perturbed grid files": self.outfile})

        # modify param.ini
        custom_input(self.config["simulator"]["ref_input"], args)

        # execute MUSICAA to deform mesh
        os.chdir(self.dat_dir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"]
        out_message = os.path.join(musicaa_mesh_dir, f"musicaa_{self.outfile}.out")
        err_message = os.path.join(musicaa_mesh_dir, f"musicaa_{self.outfile}.err")
        with open(out_message, "wb") as out:
            with open(err_message, "wb") as err:
                logger.info(f"deform mesh for {mesh_dir}")
                proc = subprocess.Popen(preprocess_cmd,
                                        env=os.environ,
                                        stdin=subprocess.DEVNULL,
                                        stdout=out,
                                        stderr=err,
                                        universal_newlines=True)
        os.chdir(self.cwd)

        # wait to finish
        proc.communicate()


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
        - residual_convergence_order (float): order of convergence required for steady computations.
        - computation_type (str): type of computation (steady/unsteady)

        # Convergence criteria
        # --------------------
        - residual_convergence_order (float): order of convergence required for steady computations.
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
        self.CFL: float = float(read_next_line_in_file(config["simulator"]["ref_input"], "CFL"))
        self.lower_CFL: float = self.CFL
        self.ndeb_RANS: int = int(read_next_line_in_file(config["simulator"]["ref_input"],
                                                         "ndeb_RANS"))
        self.nprint: int = int(re.findall(r'\b\d+\b',
                                          read_next_line_in_file(config["simulator"]["ref_input"],
                                                                 "screen"))[0])
        self.residual_convergence_order: float \
            = config["simulator"].get("residual_convergence_order", 4)
        self.computation_type: str = read_next_line_in_file(config["simulator"]["ref_input"],
                                                            "DES without subgrid")
        self.computation_type = "unsteady" if self.computation_type == "N" else "steady"
        # convergence criteria
        self.residual_convergence_order: float \
            = config["simulator"]["convergence_criteria"].get("residual_convergence_order", 4)
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
        self.sim_info: dict = {}
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
        if "mesh" not in self.config["plot3D"]:
            logger.debug(f"no <mesh> entry in {self.config['plot3D']}")
        if "post_process" not in self.config["simulator"]:
            logger.debug(f"no <post_process> entry in {self.config['simulator']}")

    def set_solver_name(self):
        """
        **Sets** the solver name to musicaa.
        """
        self.solver_name = "musicaa"

    def execute_sim(self,
                    meshfile: str = "",
                    gid: int = 0,
                    cid: int = 0,
                    restart: int = 0,
                    is_stats: bool = False,
                    is_init_unsteady_3D: bool = False):
        """
        **Pre-processes** and **executes** a MUSICAA simulation.
        """
        # add gid entry to the results dictionary
        if gid not in self.df_dict:
            self.df_dict[gid] = {}

        if is_stats:
            # continue computation for statistics
            sim_outdir = self.get_sim_outdir(gid, cid)
            self.pre_process_stats(sim_outdir, ndeb_stats)
            self.execute(sim_outdir, gid, cid, meshfile, restart)
            # update new simulation info
            self.sim_pro[-1][0].update({"is_stats": True})
            self.sim_pro[-1][0].update({"ndeb_stats": ndeb_stats})
            self.sim_pro[-1][0].update({"is_init_unsteady_3D": False})

        elif is_init_unsteady_3D:
            # initialize unsteady computation from 2D to 3D
            sim_outdir = self.get_sim_outdir(gid, cid)
            self.pre_process_init_unsteady(sim_outdir, "3D")
            self.execute(sim_outdir, gid, cid, meshfile, restart)
            # update new simulation info
            self.sim_pro[-1][0].update({"is_init_unsteady_2D": False})
            self.sim_pro[-1][0].update({"is_init_unsteady_3D": True})

        else:
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

                # if starting unsteady computation, pre-process to initialize in 2D
                if self.computation_type == "unsteady":
                    self.pre_process_init_unsteady(sim_outdir, "2D")

                # if computation has crashed, restart with lower CFL
                if restart > 0:
                    args: dict = {"CFL": self.lower_CFL}
                    custom_input(os.path.join(sim_outdir, "param.ini"), args)
                    self.CFL = self.lower_CFL
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
        shutil.copy(self.config["simulator"]["ref_input"], sim_outdir)
        shutil.copy(self.config["simulator"]["ref_input_mesh"], sim_outdir)
        shutil.copy(os.path.join(self.config["simulator"]["path_to_solver"],
                                 self.solver_name), sim_outdir)
        logger.info((f"{self.config['simulator']['ref_input']}, "
                     f"{self.config['simulator']['ref_input_mesh']} and "
                     f"{self.solver_name} copied to {sim_outdir}"))

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
                                  self.config["simulator"]["ref_input"].split("/")[-1]), args)

        # copy any other solver expected files
        shutil.copy(self.config["simulator"]["ref_input_rans"], sim_outdir)
        shutil.copy(self.config["simulator"]["ref_input_feos"], sim_outdir)

        # execute MUSICAA to delete half-cell
        os.chdir(sim_outdir)
        preprocess_cmd = self.config["simulator"]["preprocess_cmd"]
        with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.out", "wb") as out:
            with open(f"{self.solver_name}_g{gid}_c{cid}_half-cell.err", "wb") as err:
                logger.info(f"delete mesh half-cell for g{gid}, c{cid} with {self.solver_name}")
                proc = subprocess.Popen(preprocess_cmd,
                                        env=os.environ,
                                        stdin=subprocess.DEVNULL,
                                        stdout=out,
                                        stderr=err,
                                        universal_newlines=True)
        # wait to finish
        proc.communicate()

        # modify solver input file: mode from_scratch
        args.update({"from_interp": "1"})
        custom_input(self.config["simulator"]["ref_input"].split("/")[-1], args)
        logger.info(f"changed execution mode to 1 in {sim_outdir}")
        os.chdir(self.cwd)

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
                logger.info((f"execute {self.computation_type} simulation "
                             f"g{gid}, c{cid} with {self.solver_name}"))
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
        finished_sim = []
        # loop over the list of simulation processes
        for id, (dict_id, p_id) in enumerate(self.sim_pro):
            returncode = p_id.poll()
            sim_outdir = self.get_sim_outdir(dict_id["gid"], dict_id["cid"])
            # get mesh info
            mesh = CustomMesh({"dat_dir": sim_outdir}, just_get_block_info=True)
            self.block_info = mesh.block_info
            # initialize computation type
            if "is_stats" not in dict_id:
                dict_id.update({"is_stats": False})
            if "is_init_unsteady_2D" not in dict_id and self.computation_type == "unsteady":
                dict_id.update({"is_init_unsteady_2D": True})
                dict_id.update({"is_init_unsteady_3D": False})

            if returncode is None and ("stop" not in dict_id):
                # check results convergence
                self.only_compute_mean_crit = False
                if "is_init_unsteady_2D" in dict_id and dict_id["is_init_unsteady_2D"]:
                    self.only_compute_mean_crit = True
                converged, niter = self.check_convergence(sim_outdir, dict_id)
                dict_id.update({"niter": niter})
                if converged:
                    self.stop_MUSICAA(sim_outdir)
                    dict_id.update({"stop": True})
                else:
                    pass  # simulation still running

            elif returncode == 0:
                # check if statistics files exist
                if os.path.isfile(os.path.join(sim_outdir, "stats1_bl1.bin")):
                    logger.info(f"{self.computation_type} simulation {dict_id} finished")
                    finished_sim.append(id)
                    # clean directory of line sensors
                    rm_filelist([os.path.join(sim_outdir, "line*")])
                    self.df_dict[dict_id["gid"]][dict_id["cid"]] = self.post_process(
                        dict_id, sim_outdir
                    )
                else:
                    finished_sim.append(id)
                    # if unsteady computation initialized in 2D
                    if self.computation_type == "unsteady" and dict_id["is_init_unsteady_2D"]:
                        logger.info((f"unsteady 2D initialization {dict_id} finished, "
                                     "executing 3D initialization"))
                        dict_id.update({"is_init_unsteady_2D": False})
                        dict_id.update({"is_init_unsteady_3D": True})
                        # clean directory of point sensors
                        rm_filelist([os.path.join(sim_outdir, "point*")])
                        self.execute_sim(
                            dict_id["meshfile"], dict_id["gid"], dict_id["cid"],
                            dict_id["restart"],
                            is_init_unsteady_3D=dict_id["is_init_unsteady_3D"]
                        )
                    # else gather statistics
                    else:
                        logger.info((f"{self.computation_type} simulation {dict_id}"
                                     "continued for statistics"))
                        dict_id.update({"is_stats": True})
                        # clean directory of line sensors
                        if self.computation_type == "unsteady":
                            rm_filelist([os.path.join(sim_outdir, "line*")])
                        self.execute_sim(
                            dict_id["meshfile"], dict_id["gid"], dict_id["cid"],
                            dict_id["restart"],
                            is_stats=dict_id["is_stats"],
                            ndeb_stats=dict_id["ndeb_stats"]
                        )
                break

            else:
                if dict_id["restart"] < self.restart:
                    # reduce CFL
                    self.lower_CFL /= self.config['simulator']['divide_CFL_by']
                    logger.error((f"ERROR -- simulation {dict_id} crashed with CFL={self.CFL} "
                                  f"and will be restarted with lower CFL="
                                  f"{self.lower_CFL}"))

                    temp_dir = "examples/LRN-CASCADE/cascade_musicaa_base/"
                    with open(os.path.join(sim_outdir,
                                           f"{self.solver_name}_g0_c0.out"), "r") as out:
                        text = out.readlines()
                        with open(os.path.join(temp_dir, "temp.out"), "w") as tout:
                            tout.writelines(text)
                    with open(os.path.join(sim_outdir,
                                           f"{self.solver_name}_g0_c0.err"), "r") as err:
                        text = err.readlines()
                        with open(os.path.join(temp_dir, "temp.err"), "w") as terr:
                            terr.writelines(text)

                    finished_sim.append(id)
                    shutil.rmtree(sim_outdir, ignore_errors=True)
                    self.execute_sim(
                        dict_id["meshfile"], dict_id["gid"], dict_id["cid"], dict_id["restart"] + 1
                    )
                else:
                    raise Exception(f"ERROR -- simulation {dict_id} crashed")
        # update the list of active processes
        self.sim_pro = [tup for id, tup in enumerate(self.sim_pro) if id not in finished_sim]
        return len(self.sim_pro)

    def post_process(self, dict_id: dict, sim_outdir: str) -> str:
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
                get_value: Callable = getattr(self, qty)
                value = get_value(sim_outdir)
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
        # get sim_info
        nx = self.sim_info[f"block_{bl}"]["nx"]
        ny = self.sim_info[f"block_{bl}"]["ny"]
        ngh = self.sim_info["ngh"]
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

    def pressure_loss_coeff(self, sim_outdir: str) -> float:
        """
        **Post-processes** the results of a terminated simulation.
        **Returns** the extracted results in a DataFrame.
        """
        # compute inlet mixed-out pressure
        inlet_data = self.extract_measurement_line(sim_outdir, "inlet")
        inlet_mixed_out_state = self.mixed_out(inlet_data)

        # compute oulet mixed-out pressure
        outlet_data = self.extract_measurement_line(sim_outdir, "outlet")
        outlet_mixed_out_state = self.mixed_out(outlet_data)

        return (inlet_mixed_out_state["p0_bar"] - outlet_mixed_out_state["p0_bar"]) /\
               (inlet_mixed_out_state["p0_bar"] - inlet_mixed_out_state["p_bar"])

    def kill_all(self):
        """
        **Kills** all active processes.
        """
        logger.debug("Function 'kill_all' not yet implemented")

    def read_stats_bl(self, sim_outdir: str, bl_list: list[int], var_list: list[str]) -> dict:
        """
        **Reads** statistics of MUSICAA computation block from stats*_bl*.bin.
        """
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
        data = {}
        stats = 1
        for vars in all_vars:
            for bl in bl_list:
                # get filename
                filename = os.path.join(sim_outdir, f"stats{stats}_bl{bl}.bin")

                # get block dimensions
                nx = self.sim_info[f"block_{bl}"]["nx"]
                ny = self.sim_info[f"block_{bl}"]["ny"]

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
        feos_info = self.get_feos_info(sim_outdir)
        data["gam"] = feos_info["Equivalent gamma"]
        data["R"] = feos_info["Gas constant"]

        return data

    def extract_measurement_line(self, sim_outdir: str, location: str) -> dict:
        """
        Extract data along measurement line.
        """
        if location == "inlet":
            bl_list = self.config["plot3D"]["mesh"]["inlet_bl"]
        else:
            bl_list = self.config["plot3D"]["mesh"]["outlet_bl"]

        # get simulation information
        self.get_sim_info(sim_outdir)

        # get time-averaged block data
        var_list_interp = ["uu", "vv", "ww", "rhou", "rho*uu", "rho*uv", "rho*uw", "p", "T"]
        data = self.read_stats_bl(sim_outdir, bl_list, var_list_interp)

        # get coordinates
        for bl in bl_list:
            data[f"block_{bl}"]["x"], data[f"block_{bl}"]["y"] = self.read_bl(sim_outdir, bl)

        # interpolation line definition
        x1 = self.config["simulator"]["post_process"]["measurement_lines"][f"{location}_x1"]
        x2 = self.config["simulator"]["post_process"]["measurement_lines"][f"{location}_x2"]
        closest_index = find_closest_index(data[f"block_{bl_list[0]}"]["x"][:, 0], x1)
        y1 = data[f"block_{bl_list[0]}"]["y"][closest_index, :].min()
        y2 = y1 + self.config["plot3D"]["mesh"]["pitch"]
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
        # regular expression to match lines with a name and a corresponding value
        pattern = re.compile(r"([A-Za-z\s]+)\.{2,}\s*(\S+)")

        # iterate over each line and apply the regex pattern
        feos_info = {}
        with open(os.path.join(sim_outdir, "feos_air.ini"), "r") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2)
                    feos_info[key] = float(value)

        return feos_info

    def get_time_info(self, sim_outdir: str) -> dict:
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

    def get_sim_info(self, sim_outdir: str) -> dict:
        """
        **Returns** a dictionnary containing relevant information on the mesh
        used by MUSICAA: number of blocks, block size, number of ghost points.
        Note: this element is automatically produced by MUSICAA, and need not be
        created manually.
        """
        # read relevant lines
        with open(os.path.join(sim_outdir, "info.ini"), "r") as f:
            lines = f.readlines()
        self.sim_info["nbloc"] = int(lines[0].split()[4])
        for bl in range(self.sim_info["nbloc"]):
            bl += 1
            self.sim_info[f"block_{bl}"] = {}
            self.sim_info[f"block_{bl}"]["nx"] = int(lines[bl].split()[5])
            self.sim_info[f"block_{bl}"]["ny"] = int(lines[bl].split()[6])
            self.sim_info[f"block_{bl}"]["nz"] = int(lines[bl].split()[7])
        self.sim_info["ngh"] = int(lines[bl + 8].split()[-1])
        self.sim_info["dt"] = float(lines[bl + 9].split()[-1])

    def get_sensors(self, sim_outdir: str, just_get_niter: bool = False) -> dict:
        """
        **Returns** the sensors time signals as a dictionnary, both point and line types.
        """
        # iterate over sensors
        sensors: dict = {}
        sensors["total_nb_points"] = self.block_info["total_nb_points"]
        sensors["total_nb_lines"] = self.block_info["total_nb_lines"]
        nbl = self.block_info["nbl"]
        for bl in range(nbl):
            bl += 1
            for sensor_type, nb_sensors in [("point", "nb_points"), ("line", "nb_lines")]:
                for sensor_nb in range(self.block_info[f"block_{bl}"][nb_sensors]):
                    sensor_nb += 1
                    sensors = self.read_sensor(sim_outdir, sensors, sensor_type,
                                               bl, sensor_nb, just_get_niter)
                    if just_get_niter:
                        return sensors
        return sensors

    def read_sensor(self,
                    sim_outdir: str,
                    sensors: dict,
                    sensor_type: str,
                    bl: int,
                    sensor_nb: int,
                    just_get_niter: bool = False) -> np.array:
        """
        **Returns** data from a sensor.
        """
        dtype = np.dtype("f8")
        nvars = self.block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nvars"]
        freq = self.block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["freq"]
        nz1 = self.block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nz1"]
        nz2 = self.block_info[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"]["nz2"]
        nz = 1 + nz2 - nz1

        # open file and collect data
        f = open(
            os.path.join(sim_outdir,
                         f"{sensor_type}_{str(sensor_nb).rjust(3, '0')}_bl{bl}.bin"),
            "r")
        data = np.fromfile(f, dtype=dtype, count=-1)
        niter_sensor = data.size // nvars
        niter = data.size // nvars * freq
        sensors["niter"] = niter
        if just_get_niter:
            return sensors
        data = data[:niter_sensor * nvars].reshape((niter_sensor, nvars, nz))
        sensors[f"block_{bl}"] = {}
        sensors[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"] = data.copy()
        return sensors

    def get_residuals(self, sim_outdir: str) -> dict:
        """
        **Returns** residuals from ongoing computations.
        """
        res = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': [], 'RANS_var': []}
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
                if it * self.nprint > self.ndeb_RANS:
                    # read arg
                    np.fromfile(f, dtype=f8_dtype, count=1)
                    res['RANS_var'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                it += 1
            except IndexError:
                break
        return res

    def check_residuals(self, sim_outdir: str) -> tuple[bool, int]:
        """
        **Returns** True if residual convergence criterion is met. If so,
        the iteration at which statistics are started is returned, otherwise
        the current iteration is returned.
        """
        # if residuals.bin file exists
        try:
            res = self.get_residuals(sim_outdir)
            unwanted = ["nn"]
            # if file not empty
            try:
                niter = res["nn"][-1]
                if niter <= self.ndeb_RANS:
                    nvars = 4
                    unwanted.append("RANS_var")
                else:
                    nvars = 5
            except IndexError:
                return False, 0

            # compute current order of convergence for each variable
            nvars_converged = 0
            res_max = []
            for var in set(res) - set(unwanted):
                res_max.append(max(res[var]) - min(res[var]))
                if max(res[var]) - min(res[var]) > self.residual_convergence_order:
                    nvars_converged += 1
            if niter > self.ndeb_RANS:
                print((f"it: {niter}; "
                       f"lowest convergence order = {round_number(min(res_max), 'down', 2)}"))
                if nvars_converged == nvars:
                    return True, int(round_number(niter, "up", -3))
                else:
                    return False, int(round_number(niter, "up", -3))
            else:
                print((f"it: {niter}; RANS starts at it: {self.ndeb_RANS}"))
                return False, int(round_number(niter, "up", -3))

        except FileNotFoundError:
            return False, 0

    def check_transient(self, sim_outdir: str, dict_id: dict) -> tuple[bool, int]:
        """
        **Returns** True if transient ending criterion is met.
        If criterion is met, the iteration at which statistics are started is returned, otherwise
        the current iteration is returned.
        """
        try:
            # read the first sensor to get current iteration
            sensors = self.get_sensors(sim_outdir, just_get_niter=True)
            niter = sensors["niter"]
        except FileNotFoundError:
            # computation not started
            return False, 0

        # the checking frequency is set to one f.t.t = L_ref/u_ref
        feos_info = self.get_feos_info(sim_outdir)
        Mach_ref = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                                "Reference Mach"))
        T_ref = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                             "Reference temperature"))
        c_ref = np.sqrt(feos_info["Equivalent gamma"] * feos_info["Gas constant"] * T_ref)
        u_ref = c_ref * Mach_ref
        Lgrid = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                             "Scaling value for the grid Lgrid"))
        L_ref = self.config["plot3D"]["mesh"]["chord_length"]
        if Lgrid != 0:
            L_ref = L_ref * Lgrid
        ftt = L_ref / u_ref
        self.get_sim_info(sim_outdir)
        niter_ftt = int(ftt / self.sim_info["dt"])

        # proceed if this criterion has not already been checked
        # and if at least 4 periods have passed (see J. Boudet (2018))
        try:
            if (
                niter // niter_ftt > dict_id["n_convergence_check"]
                and niter // niter_ftt >= self.nb_ftt_before_criterion
            ):
                dict_id["n_convergence_check"] += 1
                return self.Boudet_crit(sim_outdir, niter_ftt)
            # already checked
            else:
                return False, 0
        except KeyError:
            dict_id.update({"n_convergence_check": self.nb_ftt_before_criterion - 1})
            return False, 0

    def Boudet_crit(self, sim_outdir: str, niter_ftt: int) -> tuple[bool, int]:
        """
        **Returns** (True, iteration to start statistics) if transient ending criterion is met:
        see J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
        """
        # get sensors
        sensors = self.get_sensors(sim_outdir)
        converged: list[int] = []
        global_crit: list[float] = []
        for block in (key for key in sensors if key.startswith("block")):
            bl = int(block.split("_")[-1])
            for sensor_type, nb_sensors in [("point", "nb_points"), ("line", "nb_lines")]:
                for sensor_nb in range(self.block_info[f"block_{bl}"][nb_sensors]):
                    sensor_nb += 1
                    sensor = sensors[f"block_{bl}"][f"{sensor_type}_{sensor_nb}"].copy()
                    # moving average to avoid temporary convergence
                    mov_avg_crit: list[float] = []
                    for sample in range(self.nb_ftt_before_criterion):
                        sample = -sample * niter_ftt - 1
                        sensor = sensor[:sample]
                        crit, niter = self.compute_Boudet_crit(sensor)
                        mov_avg_crit.append(crit)

                    # if threshold w.r.t mean only or rms too
                    if self.only_compute_mean_crit:
                        transient_convergence_percent = self.transient_convergence_percent_mean
                    else:
                        transient_convergence_percent = self.transient_convergence_percent_rms

                    # check if converged
                    print(mov_avg_crit)
                    mov_avg = sum(mov_avg_crit) / len(mov_avg_crit)
                    if sum(mov_avg_crit) / len(mov_avg_crit) < transient_convergence_percent:
                        is_converged = True
                    else:
                        is_converged = False
                    converged.append(is_converged)
                    global_crit.append(mov_avg)

                    # # if threshold w.r.t mean only or rms too
                    # if self.only_compute_mean_crit:
                    #     transient_convergence_percent = self.transient_convergence_percent_mean
                    # else:
                    #     transient_convergence_percent = self.transient_convergence_percent_rms
                    # crit, niter = self.compute_Boudet_crit(sensor)
                    # if crit < transient_convergence_percent:
                    #     is_converged = True
                    # else:
                    #     is_converged = False
                    # converged.append(is_converged)
                    # global_crit.append(crit)

        print((f"it: {sensors['niter']}; "
               f"max transient variation = {round_number(max(global_crit), 'closest', 2)}%"))
        if sum(converged) == sensors["total_nb_points"] + sensors["total_nb_lines"]:
            return True, round_number(niter, "up", -3)
        else:
            return False, 0

    def compute_Boudet_crit(self, sensor: np.ndarray) -> dict:
        """
        **Returns** (True, iteration to start statistics) if transient ending criterion is met:
        see J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
        This criterion is here modified in an attempt to reduce the unused simulation
        time from 3 additional transients to 1 (in practice ~2) by considering thirds
        of the second half rather than quarters of the whole.
        The longer the transient, the more efficient this approach.
        In the end, a geometric mean of both criteria is retained.
        """
        # shape of array: (niter, nvars, nz)
        niter = sensor.shape[0]
        nvars = sensor.shape[1]
        nz = sensor.shape[2]

        # get mean, rms and quarters
        quarter = niter // 4
        mean = np.mean(sensor, axis=(0, 2))
        sensor_squared = (sensor - mean[np.newaxis, :, np.newaxis])**2
        if niter % 4 == 0:
            mean_quarters = sensor.reshape((4, quarter, nvars, nz))
            rms_quarters = sensor_squared.reshape((4, quarter, nvars, nz))
        else:
            mean_quarters = np.zeros((4, quarter, nvars, nz))
            rms_quarters = np.zeros((4, quarter, nvars, nz))
            mean_quarters[0] = sensor[:quarter]
            mean_quarters[1] = sensor[quarter:2 * quarter]
            mean_quarters[2] = sensor[2 * quarter:3 * quarter]
            mean_quarters[3] = sensor[3 * quarter:4 * quarter]
            rms_quarters[0] = sensor_squared[:quarter]
            rms_quarters[1] = sensor_squared[quarter:2 * quarter]
            rms_quarters[2] = sensor_squared[2 * quarter:3 * quarter]
            rms_quarters[3] = sensor_squared[3 * quarter:4 * quarter]
        mean_quarters = np.mean(mean_quarters, axis=(1, 3))
        rms_quarters = np.mean(np.sqrt(np.mean(rms_quarters, axis=1)), axis=-1)

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
            if self.only_compute_mean_crit:
                crit.append(max(mean_crit) * 100)
            else:
                crit.append(max(max(mean_crit), max(rms_crit)) * 100)
        # return if original Boudet criterion requested
        if self.Boudet_criterion_type == "original":
            return max(crit), niter

        # get mean, rms and quarters
        half = niter // 2
        sixth = half // 3
        mean = np.mean(sensor, axis=(0, 2))
        sensor_squared = (sensor - mean[np.newaxis, :, np.newaxis])**2
        if niter % 6 == 0:
            mean_sixths = sensor[half:].reshape((3, sixth, nvars, nz))
            rms_sixths = sensor_squared[half:].reshape((3, sixth, nvars, nz))
        else:
            mean_sixths = np.zeros((3, sixth, nvars, nz))
            rms_sixths = np.zeros((3, sixth, nvars, nz))
            mean_sixths[0] = sensor[:sixth]
            mean_sixths[1] = sensor[sixth:2 * sixth]
            mean_sixths[2] = sensor[2 * sixth:3 * sixth]
            rms_sixths[0] = sensor_squared[:sixth]
            rms_sixths[1] = sensor_squared[sixth:2 * sixth]
            rms_sixths[2] = sensor_squared[2 * sixth:3 * sixth]
        mean_sixths = np.mean(mean_sixths, axis=(1, 3))
        rms_sixths = np.mean(np.sqrt(np.mean(rms_sixths, axis=1)), axis=-1)

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
            if self.only_compute_mean_crit:
                crit_modif.append(max(mean_crit) * 100)
            else:
                crit_modif.append(max(max(mean_crit), max(rms_crit)) * 100)
        # return if modified Boudet criterion requested
        if self.Boudet_criterion_type == "modified":
            return max(crit_modif), niter

        # compute geometric mean if requested
        if self.Boudet_criterion_type == "mean":
            crit_mean = [abs(crit_Boudet * crit_modif[i]) / (crit_Boudet * crit_modif[i])
                         * np.sqrt(abs(crit_Boudet * crit_modif[i])) for
                         i, crit_Boudet in enumerate(crit)]
            return max(crit_mean), niter

    def check_stats(self, sim_outdir: str) -> bool:
        """
        **Returns** True if MUSICAA statistics convergence criterion is met:
        see J. Boudet (2018): https://doi.org/10.1007/s11630-015-0752-8
        """
        # compute stats convergence

    def check_convergence(self, sim_outdir: str, dict_id: dict) -> tuple[bool, int]:
        """
        **Returns** True if MUSICAA computation convergence criterion is met. If so,
        returns current iteration number to start statistics.
        """
        if self.computation_type == "steady":
            if not dict_id["is_stats"]:
                return self.check_residuals(sim_outdir)
            else:
                return False, 0
        else:
            if not dict_id["is_stats"]:
                return self.check_transient(sim_outdir, dict_id)
            else:
                return "Unsteady convergence criterion to be implemented."

    def pre_process_stats(self, sim_outdir: str, ndeb_stats: int):
        """
        **Pre-processes** computation for statistics.
        """
        # modify param.ini file
        args = {}
        args.update({"from_field": "2"})
        args.update({"Iteration number to start statistics": f"{ndeb_stats + 1}"})
        if self.computation_type == "steady":
            args.update({"Max number of temporal iterations": "2 3000.0"})
        elif self.computation_type == "unsteady":
            args.update({"Max number of temporal iterations": "9999999 3000.0"})
        custom_input(os.path.join(sim_outdir,
                                  self.config["simulator"]["ref_input"].split("/")[-1]
                                  ), args)

    def pre_process_init_unsteady(self, sim_outdir: str, dimension: str):
        """
        **Pre-processes** unsteady computation to initialize with 2D or 3D simulation.
        """
        args = {}
        if dimension == "2D":
            # modify param.ini file
            args.update({"Implicit Residual Smoothing": "2"})
            args.update({"Residual smoothing parameter": "0.42 0.1 0.005 0.00025 0.0000125"})
            is_SF = read_next_line_in_file(self.config["simulator"]["ref_input"],
                                           ("Selective Filtering: is_SF"
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
            custom_input(os.path.join(sim_outdir,
                                      self.config["simulator"]["ref_input"].split("/")[-1]
                                      ), args)
            # modify param_blocks.ini file: 3D->2D
            self.change_dimensions_param_blocks(sim_outdir)
        else:
            # copy original param.ini and param_blocks.ini files to directory
            cp_filelist([self.config["simulator"]["ref_input"],
                         self.config["simulator"]["ref_input_mesh"]], [sim_outdir] * 2)

    def change_dimensions_param_blocks(self, sim_outdir: str):
        """
        **Modifies** the param_blocks.ini file of a given simulation to 2D.
        """
        # file and block info
        filename = os.path.join(sim_outdir, "param_blocks.ini")
        mesh = CustomMesh({"dat_dir": sim_outdir}, just_get_block_info=True)
        self.block_info = mesh.block_info
        nbl = self.block_info["nbl"]

        with open(filename, "r") as f:
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
                    nb_points = self.block_info[f"block_{bl}"]["nb_points"]
                    if nb_points > 0:
                        for point_nb in range(nb_points):
                            point_nb += 1
                            if self.block_info[f"block_{bl}"][f"point_{point_nb}"]["nz1"] != 1:
                                position \
                                    = self.block_info[f"block_{bl}"][(f"point_"
                                                                      f"{point_nb}")]["position"]
                                var_list \
                                    = self.block_info[f"block_{bl}"][(f"point_"
                                                                      f"{point_nb}")]["var_list"]
                                info_snap = [int(dim) for dim in
                                             re.findall(r"\d+", filedata[i + 21 + position])]
                                info_snap[4], info_snap[5] = 1, 1
                                replacement = "   " + "    ".join([str(j) for j in info_snap]) \
                                    + "    " + " ".join(var_list) + "\n"
                                filedata[i + 21 + position] = replacement
                    # if line sensors, change to points
                    nb_lines = self.block_info[f"block_{bl}"]["nb_lines"]
                    if nb_lines > 0:
                        for line_nb in range(nb_lines):
                            line_nb += 1
                            position \
                                = self.block_info[f"block_{bl}"][f"line_{line_nb}"]["position"]
                            var_list \
                                = self.block_info[f"block_{bl}"][f"line_{line_nb}"]["var_list"]
                            info_snap = [int(dim) for dim in
                                         re.findall(r"\d+", filedata[i + 21 + position])]
                            info_snap[4], info_snap[5] = 1, 1
                            replacement = "   " + "    ".join([str(j) for j in info_snap]) \
                                + "    " + " ".join(var_list) + "\n"
                            filedata[i + 21 + position] = replacement
        with open(filename, "w") as f:
            f.writelines(filedata)

    def get_niter_ftt(self, sim_outdir: str) -> int:
        """
        **Returns** the number of iterations per flow-through time (ftt)
        """
        # f.t.t = L_ref/u_ref
        feos_info = self.get_feos_info(sim_outdir)
        Mach_ref = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                                "Reference Mach"))
        T_ref = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                             "Reference temperature"))
        c_ref = np.sqrt(feos_info["Equivalent gamma"] * feos_info["Gas constant"] * T_ref)
        u_ref = c_ref * Mach_ref
        Lgrid = float(read_next_line_in_file(self.config["simulator"]["ref_input"],
                                             "Scaling value for the grid Lgrid"))
        L_ref = self.config["plot3D"]["mesh"]["chord_length"]
        if Lgrid != 0:
            L_ref = L_ref * Lgrid
        ftt = L_ref / u_ref
        self.get_sim_info(sim_outdir)

        return int(ftt / self.sim_info["dt"])

    def stop_MUSICAA(self, sim_outdir: str):
        """
        **Stops** MUSICAA during execution.
        """
        # send signal to MUSICAA if convergence reached
        with open(f"{sim_outdir}/stop", "w") as stop:
            stop.write("stop")
