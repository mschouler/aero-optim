import os
import pandas as pd
import shutil
import subprocess

from abc import ABC, abstractmethod
from .utils import check_dir


class Simulator(ABC):
    """
    This class implements a basic Simulator class.
    """
    def __init__(self, config: dict):
        """
        Instantiates the Simulator object.

        Input
            >> config: the config file dictionary.
        Inner
            >> cwd: the working directory.
            >> outdir: the output directory where the simulation results folder will be stored.
            >> ref_input: a simulation input file template.
            >> sim_args: arguments to modify to customize ref_input.
            >> post_process_args: quantities to extract from result files.
            >> def_list: list of dataframe containinf the extracted quantities of all simulation.
        """
        self.cwd: str = os.getcwd()
        self.config = config
        self.process_config()
        # study params
        self.outdir: str = config["study"]["outdir"]
        # simulator params
        self.exec_cmd: list[str] = config["simulator"]["exec_cmd"].split(" ")
        self.ref_input: str = config["simulator"]["ref_input"]
        self.sim_args: dict = config["simulator"].get("sim_args", {})
        self.post_process_args: dict = config["simulator"].get("post_process", {})
        # simulation results
        self.df_list: list[pd.DataFrame] = []

    def custom_input(self, fname: str):
        """
        Modifies the reference input to generate a new customized one.
        """
        ref_output = open(self.ref_input, "r").read().splitlines()
        for key, value in self.sim_args.items():
            idx = ref_output.index(key)
            # in place substitution
            # {"keyword": {{"inplace": true}, {'param': [param]}}}
            if value["inplace"]:
                print(f">> {key}: replace {ref_output[idx]} by {value['param'][0]}")
                ref_output[idx] = value['param'][0]
            # multiline substitution
            # {"keyword": {{"inplace": false}, {'param': [param0, param1, param..]}}}
            else:
                for ii, param in enumerate(value['param']):
                    print(f">> {key}: replace {ref_output[idx + 1 + ii]} by {param}")
                    ref_output[idx + 1 + ii] = param

        with open(fname, 'w') as ftw:
            ftw.write("\n".join(ref_output))
            print(f">> input file saved to {fname}")

    @abstractmethod
    def process_config(self):
        """
        Makes sure the config file contains the required information.
        """

    @abstractmethod
    def execute_sim(self, meshfile: str = "", gid: int = 0, cid: int = 0):
        """
        Runs a single simulation.
        """


class WolfSimulator(Simulator):
    """
    This class implements a WolfSimulator class.
    """
    def __init__(self, config: dict):
        """
        Instantiates the WolfSimulator object.

        Inner
            >> wolf_pro: list to track simulations and their associated subprocess.
               It has the following form ({'generation': gid, 'candidate': cid}, subprocess).
        """
        super().__init__(config)
        self.wolf_pro: list[tuple[dict, subprocess.Popen[str]]] = []

    def process_config(self):
        """
        Makes sure the config file contains the required information and extract it.
        """
        if "exec_cmd" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <exec_cmd> entry in {self.config['simulator']}")
        if "ref_input" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <ref_input> entry in {self.config['simulator']}")
        if "sim_args" not in self.config["simulator"]:
            print(f"WARNING -- no <sim_args> entry in {self.config['simulator']}")
        if "post_process" not in self.config["simulator"]:
            print(f"WARNING -- no <post_process> entry in {self.config['simulator']}")

    def get_sim_outdir(self, gid: int = 0, cid: int = 0) -> str:
        """
        Returns the path to the folder containing the simulation results.
        """
        return os.path.join(self.outdir, "WOLF", f"wolf_g{gid}_c{cid}")

    def execute_sim(self, meshfile: str = "", gid: int = 0, cid: int = 0):
        """
        Pre-processes and executes a Wolf simulation.
        It also updates wolf_pro.
        """
        # Pre-process
        # get the simulation meshfile
        full_meshfile = meshfile if meshfile else self.config["simulator"]["file"]
        path_to_meshfile: str = "/".join(full_meshfile.split("/")[:-1])
        meshfile = full_meshfile.split("/")[-1]
        # generate custom input by altering the wolf template
        sim_outdir = self.get_sim_outdir(gid=gid, cid=cid)
        check_dir(sim_outdir)
        self.custom_input(os.path.join(sim_outdir, f"{meshfile[:-5]}.wolf"))
        # copy meshfile to the output directory
        shutil.copy(os.path.join(path_to_meshfile, meshfile), sim_outdir)
        print(f">> {os.path.join(path_to_meshfile, meshfile)} copied to {sim_outdir}")
        # update the execution command with the right mesh file
        exec_cmd = self.exec_cmd.copy()
        idx = self.exec_cmd.index("@.mesh")
        exec_cmd[idx] = os.path.join(meshfile)
        # Execution
        # move to the output directory, execute wolf and move back to the main directory
        os.chdir(sim_outdir)
        with open(f"wolf_g{gid}_c{cid}.out", "wb") as out:
            with open(f"wolf_g{gid}_c{cid}.err", "wb") as err:
                print(f">> execute simulation g{gid} c{cid} with Wolf")
                proc = subprocess.Popen(exec_cmd,
                                        env=os.environ,
                                        stdin=subprocess.DEVNULL,
                                        stdout=out,
                                        stderr=err,
                                        universal_newlines=True)
        os.chdir(self.cwd)
        # append simulation to the list of active processes
        self.wolf_pro.append(({"generation": gid, "candidate": cid}, proc))

    def monitor_sim_progress(self) -> int:
        """
        Updates and returns the list of simulations under execution.
        """
        finished_sim = []
        for id, (dict_id, p_id) in enumerate(self.wolf_pro):
            returncode = p_id.poll()
            if returncode is None:
                pass  # simulation still running
            elif returncode == 0:
                print(f">> simulation {dict_id} finished")
                finished_sim.append(id)
                self.df_list.append(self.post_process(dict_id))
                break
            else:
                raise Exception(f"ERROR -- simulation {p_id} crashed")
        self.wolf_pro = [tup for id, tup in enumerate(self.wolf_pro) if id not in finished_sim]
        return len(self.wolf_pro)

    def post_process(self, dict_id: dict) -> pd.DataFrame:
        """
        Post-processes the results of a terminated simulation.
        Returns the extracted results in a DataFrame.
        """
        sim_out_dir = self.get_sim_outdir(dict_id["generation"], dict_id["candidate"])
        qty_list: list[list[float]] = []
        head_list: list[str] = []
        for key, value in self.post_process_args.items():
            file = open(os.path.join(sim_out_dir, key), "r").read().splitlines()
            headers = file[0][2:].split()  # ignore "# " before "Iter" in headers
            for qty in value:
                try:
                    idx = headers.index(qty)
                    qty_list.append([float(line.split()[idx]) for line in file[1:]])
                    head_list.append(qty)
                except Exception as e:
                    print(f"WARNING -- could not read {qty} in {headers}")
                    print(f"WARNING -- exception {e} was raised")
        df = pd.DataFrame({head_list[i]: qty_list[i] for i in range(len(qty_list))})
        print(f">> g{dict_id['generation']}, c{dict_id['candidate']} converged in {len(df)} it.")
        print(f">> last five values: {df.tail(n=5).to_string(index=False)}")
        return df

    def kill_all(self):
        """
        Kills all active processes.
        """
        print(f">> {len(self.wolf_pro)} remaining simulations will be killed")
        _ = [subpro.terminate() for _, subpro in self.wolf_pro]
