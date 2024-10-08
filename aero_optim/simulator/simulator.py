import logging
import os
import pandas as pd
import shutil
import subprocess

from abc import ABC, abstractmethod
from aero_optim.utils import check_dir

logger = logging.getLogger(__name__)


class Simulator(ABC):
    """
    This class implements an abstract simulation class.
    """
    def __init__(self, config: dict):
        """
        Instantiates the Simulator object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - cwd (str): the working directory.
        - solver_name (str): the solver name.
        - outdir (str): the output directory where the simulation results folder will be stored.
        - exec_cmd (list[str]): solver execution command.
        - ref_input (str): a simulation input file template.
        - sim_args (dict): arguments to modify to customize ref_input.
        - files_to_cp (list[str]): list of files to be copied to the output directory.
        - post_process_args (dict): quantities to extract from result files.
        - df_dict (dict): dictionary of dataframes containing all simulations extracted quantities.
        """
        self.cwd: str = os.getcwd()
        self.config = config
        self.process_config()
        self.set_solver_name()
        # study params
        self.outdir: str = config["study"]["outdir"]
        # simulator params
        self.exec_cmd: list[str] = config["simulator"]["exec_cmd"].split(" ")
        self.ref_input: str = config["simulator"]["ref_input"]
        self.sim_args: dict = config["simulator"].get("sim_args", {})
        self.files_to_cp: list[str] = config["simulator"].get("files_to_cp", [])
        self.post_process_args: dict = config["simulator"].get("post_process", {})
        # simulation results
        self.df_dict: dict[int, dict[int, pd.DataFrame]] = {}

    def custom_input(self, fname: str):
        """
        **Writes** a customized input file.
        """
        ref_output = open(self.ref_input, "r").read().splitlines()
        for key, value in self.sim_args.items():
            idx = ref_output.index(key)
            # in place substitution
            # {"keyword": {{"inplace": true}, {'param': [param]}}}
            if value["inplace"]:
                logger.info(f"{key}: replace {ref_output[idx]} by {value['param'][0]}")
                ref_output[idx] = value['param'][0]
            # multiline substitution
            # {"keyword": {{"inplace": false}, {'param': [param0, param1, param..]}}}
            else:
                for ii, param in enumerate(value['param']):
                    logger.info(f"{key}: replace {ref_output[idx + 1 + ii]} by {param}")
                    ref_output[idx + 1 + ii] = param

        with open(fname, 'w') as ftw:
            ftw.write("\n".join(ref_output))
            logger.info(f"input file saved to {fname}")

    def get_sim_outdir(self, gid: int = 0, cid: int = 0) -> str:
        """
        **Returns** the path to the folder containing the simulation results.
        """
        return os.path.join(
            self.outdir, f"{self.solver_name.upper()}",
            f"{self.solver_name}_g{gid}_c{cid}"
        )

    def kill_all(self, *args, **kwargs):
        """
        **Kills** all active simulations.
        """
        logger.debug("kill_all not implemented")

    @abstractmethod
    def set_solver_name(self):
        """
        Sets the solver_name attribute.
        """
        self.solver_name = "solver"

    @abstractmethod
    def process_config(self):
        """
        Makes sure the config file contains the required information.
        """

    @abstractmethod
    def execute_sim(self, *args, **kwargs):
        """
        Runs a single simulation.
        """


class WolfSimulator(Simulator):
    """
    This class implements a simulator for the CFD code WOLF.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfSimulator object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - exec_cmd (list[str]): solver execution command.

        Note:
            with wolf, the exec_cmd is expected to contain a @.mesh argument that is
            automatically replaced with the simulation input mesh file name.

        - sim_pro (list[tuple[dict, subprocess.Popen[str]]]): list to track simulations
          and their associated subprocess.

            It has the following form:
            ({'gid': gid, 'cid': cid, 'meshfile': meshfile, 'restart': restart}, subprocess).

        - restart (int): how many times a simulation is allowed to be restarted in case of failure.
        """
        super().__init__(config)
        self.sim_pro: list[tuple[dict, subprocess.Popen[str]]] = []
        self.restart: int = config["simulator"].get("restart", 0)

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
        **Sets** the solver name to wolf.
        """
        self.solver_name = "wolf"

    def execute_sim(self, meshfile: str = "", gid: int = 0, cid: int = 0, restart: int = 0):
        """
        **Pre-processes** and **executes** a Wolf simulation.
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
            sim_outdir, exec_cmd = self.pre_process(meshfile, gid, cid)
            # Execution
            self.execute(sim_outdir, exec_cmd, gid, cid, meshfile, restart)

    def pre_process(self, meshfile: str, gid: int, cid: int) -> tuple[str, list[str]]:
        """
        **Pre-processes** the simulation execution
        and **returns** the execution command and directory.
        """
        # get the simulation meshfile
        full_meshfile = meshfile if meshfile else self.config["simulator"]["file"]
        path_to_meshfile: str = "/".join(full_meshfile.split("/")[:-1])
        meshfile = full_meshfile.split("/")[-1]
        # generate custom input by altering the wolf template
        sim_outdir = self.get_sim_outdir(gid=gid, cid=cid)
        check_dir(sim_outdir)
        self.custom_input(os.path.join(sim_outdir, f"{meshfile.split('.')[0]}.wolf"))
        # copy meshfile to the output directory
        shutil.copy(os.path.join(path_to_meshfile, meshfile), sim_outdir)
        logger.info(f"{os.path.join(path_to_meshfile, meshfile)} copied to {sim_outdir}")
        # copy any other solver expected files
        suffix_list = [file.split(".")[-1] for file in self.files_to_cp]
        [shutil.copy(file, os.path.join(sim_outdir, f"{meshfile.split('.')[0]}.{suffix}"))
         for file, suffix in zip(self.files_to_cp, suffix_list)]
        logger.info(f"{self.files_to_cp} copied to {sim_outdir}")
        # update the execution command with the right mesh file
        exec_cmd = self.exec_cmd.copy()
        idx = self.exec_cmd.index("@.mesh")
        exec_cmd[idx] = os.path.join(meshfile)
        return sim_outdir, exec_cmd

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
        # move to the output directory, execute wolf and move back to the main directory
        os.chdir(sim_outdir)
        with open(f"{self.solver_name}_g{gid}_c{cid}.out", "wb") as out:
            with open(f"{self.solver_name}_g{gid}_c{cid}.err", "wb") as err:
                logger.info(f"execute simulation g{gid}, c{cid} with {self.solver_name}")
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

    def monitor_sim_progress(self) -> int:
        """
        **Updates** the list of simulations under execution and **returns** its length.
        """
        finished_sim = []
        # loop over the list of simulation processes
        for id, (dict_id, p_id) in enumerate(self.sim_pro):
            returncode = p_id.poll()
            if returncode is None:
                pass  # simulation still running
            elif returncode == 0:
                logger.info(f"simulation {dict_id} finished")
                finished_sim.append(id)
                sim_outdir = self.get_sim_outdir(dict_id["gid"], dict_id["cid"])
                self.df_dict[dict_id["gid"]][dict_id["cid"]] = self.post_process(
                    dict_id, sim_outdir
                )
                break
            else:
                if dict_id["restart"] < self.restart:
                    logger.error(f"ERROR -- simulation {dict_id} crashed and will be restarted")
                    finished_sim.append(id)
                    sim_out_dir = self.get_sim_outdir(dict_id["gid"], dict_id["cid"])
                    shutil.rmtree(sim_out_dir, ignore_errors=True)
                    self.execute_sim(
                        dict_id["meshfile"], dict_id["gid"], dict_id["cid"], dict_id["restart"] + 1
                    )
                else:
                    raise Exception(f"ERROR -- simulation {dict_id} crashed")
        # update the list of active processes
        self.sim_pro = [tup for id, tup in enumerate(self.sim_pro) if id not in finished_sim]
        return len(self.sim_pro)

    def post_process(self, dict_id: dict, sim_out_dir: str) -> pd.DataFrame:
        """
        **Post-processes** the results of a terminated simulation.</br>
        **Returns** the extracted results in a DataFrame.
        """
        qty_list: list[list[float]] = []
        head_list: list[str] = []
        # loop over the post-processing arguments to extract from the results
        for key, value in self.post_process_args.items():
            # filter removes possible blank lines avoiding index out of range errors
            file = list(filter(None, open(os.path.join(sim_out_dir, key), "r").read().splitlines()))
            headers = file[0][2:].split()  # ignore "# " before first item in headers
            for qty in value:
                try:
                    idx = headers.index(qty)
                    qty_list.append([float(line.split()[idx]) for line in file[1:]])
                    head_list.append(qty)
                except Exception as e:
                    logger.warning(f"could not read {qty} in {headers}")
                    logger.warning(f"exception {e} was raised")
        # pd.Series allows columns of different lengths
        df = pd.DataFrame({head_list[i]: pd.Series(qty_list[i]) for i in range(len(qty_list))})
        logger.info(
            f"g{dict_id['gid']}, c{dict_id['cid']} converged in {len(df)} it."
        )
        logger.info(f"last values:\n{df.tail(n=1).to_string(index=False)}")
        return df

    def kill_all(self):
        """
        **Kills** all active processes.
        """
        logger.info(f"{len(self.sim_pro)} remaining simulation(s) will be killed")
        _ = [subpro.terminate() for _, subpro in self.sim_pro]


class DebugSimulator(Simulator):
    """
    This class implements a basic simulator for debugging purposes.
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def process_config(self):
        """
        Dummy process_config.
        """
        logger.debug("process_config not implemented")

    def set_solver_name(self):
        """
        Dummy set_solver_name.
        """
        self.solver_name = "debug"

    def execute_sim(self, candidate: list[float], gid: int = 0, cid: int = 0):
        """
        Dummy execute_sim.
        """
        logger.debug(f"problem dim: {len(candidate)}")
        if gid not in self.df_dict:
            self.df_dict[gid] = {}
        sim_res = self.compute_sol(candidate)
        self.post_process(sim_res, gid, cid)

    def compute_sol(self, candidate) -> float:
        """
        Dummy compute_sol based on the Ackley function.
        """
        import math
        dim = len(candidate)
        return (
            -20 * math.exp(-0.2 * math.sqrt(1.0 / dim * sum([x**2 for x in candidate])))
            - math.exp(1.0 / dim * sum([math.cos(2 * math.pi * x) for x in candidate]))
            + 20 + math.e
        )

    def post_process(self, sim_res: float, gid: int, cid: int):
        """
        Dummy post_process.
        """
        # pd.Series allows columns of different lengths
        df = pd.DataFrame({"result": pd.Series(sim_res)})
        logger.info(
            f"g{gid}, c{cid} converged in {len(df)} it."
        )
        logger.info(f"last values:\n{df.tail().to_string(index=False)}")
        self.df_dict[gid][cid] = df
