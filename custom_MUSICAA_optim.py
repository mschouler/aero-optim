import numpy as np

from aero_optim.mesh.mesh import MeshMusicaa
from aero_optim.simulator.simulator import MusicaaSimulator

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

        **Inner**

        - pitch (float): blade-to-blade pitch
        - periodic_bl (list[int]): list containing the blocks that must be translated DOWN to reform
                        the blade geometry fully. Not needed if the original mesh already
                        surrounds the blade

        """
        super().__init__(config)
        self.pitch: int = config["musicaa_mesh"].get('pitch', 1)
        self.periodic_bl: list[int] = config["musicaa_mesh"].get("periodic_bl", [0])

    def deform_mesh(self):
        """
        **Executes** the MUSICAA mesh deformation routine.
        """

        


class CustomSimulator(MusicaaSimulator):
    def __init__(self, config: dict):
        super().__init__(config)