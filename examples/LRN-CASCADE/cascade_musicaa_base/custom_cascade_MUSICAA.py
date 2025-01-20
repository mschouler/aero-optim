import subprocess
import os
import shutil

from aero_optim.mesh.mesh import MeshMusicaa
from aero_optim.simulator.simulator import MusicaaSimulator
from aero_optim.utils import modify_next_line_in_file, from_dat, check_dir

"""
This script contains various customizations of the aero_optim module
to run with the solver MUSICAA.
"""


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
        # ! The following lines will be modified to account for the
        # new MUSICAA version (coming soon), so leave the commented lines
        # change MUSICAA restart mode to 5
        modify_next_line_in_file(f'{self.config["simulator"]["ref_input_num"]}',
                                 "Restart indicator", str(5))
        # # change mesh files directory
        # modify_next_line_in_file(f'{self.config["simulator"]["ref_input_num"]}',
        #                          "dirGRID", mesh_dir)

        # copy baseline mesh in Grid directory
        path_to_Grid = os.path.join(mesh_dir, "Grid")
        check_dir(path_to_Grid)
        for bl in range(self.mesh_info["nbloc"]):
            in_file = f"{self.dat_dir}/{self.mesh_name}_bl{bl+1}.x"
            shutil.copy(in_file, path_to_Grid)

        # execute MUSICAA to deform mesh
        os.chdir(self.dat_dir)
        deform_cmd = self.config["simulator"]["deform_cmd"]
        subprocess.Popen(deform_cmd, env=os.environ)
        os.chdir(self.cwd)


# class MusicaaOptimizer(Optimizer, ABC):
#     """
#     This class implements a Wolf based Optimizer.
#     """
#     def __init__(self, config: dict):
#         """
#         Instantiates the WolfOptimizer object.

#         **Input**

#         - config (dict): the config file dictionary.
#         """
#         super().__init__(config)

#     def set_simulator_class(self):
#         """
#         **Sets** the simulator class as custom if found, as WolfSimulator otherwise.
#         """
#         super().set_simulator_class()
#         if not self.SimulatorClass:
#             self.SimulatorClass = MusicaaSimulator


class CustomSimulator(MusicaaSimulator):
    def __init__(self, config: dict):
        super().__init__(config)
