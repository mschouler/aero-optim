import gmsh
import logging

from aero_optim.mesh.mesh import Mesh

logger = logging.getLogger(__name__)


class FlatPlateMesh(Mesh):
    """
    This class implements a meshing routine for a flat plate.

    The computational domain is a rectangle domain as described in
    B. S. Venkatachari (2023): https://doi.org/10.2514/1.C037445
    """
    def __init__(self, config: dict, datfile: str = ""):
        """
        Instantiates the FlatPlateMesh object.

        **Input**

        - config (dict): the config file dictionary.
        - dat_file (str): path to input_geometry.dat.

        **Inner**

        - dinlet (float): the inlet distance to the flat plate.
        - doutlet (float): the distance between the flat plate leading edge and the outlet.
        - dtop (float): the inlet/outlet length.
        - nodes_inlet (int): the number of nodes to mesh the inlet/outlet.
        - c_npro (float): the progression coefficient in the plate normal direction.
        - c_spro (float): the progression coefficient in the plate streamwise direction.
        - snodes (tuple(int, int)): the number of nodes to mesh upstream and the flat plate.
        """
        super().__init__(config, datfile)
        self.dinlet: float = self.config["gmsh"]["domain"].get("inlet", 0.25)
        self.doutlet: float = self.config["gmsh"]["domain"].get("outlet", 20)
        self.dtop: int = self.config["gmsh"]["domain"].get("top", 5)
        self.nodes_inlet: int = self.config["gmsh"]["mesh"].get("nodes_inlet", 25)
        self.c_npro: float = self.config["gmsh"]["mesh"].get("normal_progression", 1.3)
        self.c_spro: float = self.config["gmsh"]["mesh"].get("streamwise_progression", 1.05)
        self.snodes: tuple[int, int] = self.config["gmsh"]["mesh"].get("side_nodes", (13, 45))

    def process_config(self):
        logger.debug("processing config..")
        if "domain" not in self.config["gmsh"]:
            logger.debug(f"no <domain> entry in {self.config['gmsh']}, empty entry added")
            self.config["gmsh"]["domain"] = {}
        if "inlet" not in self.config["gmsh"]["domain"]:
            logger.debug(f"no <inlet> entry in {self.config['gmsh']['domain']}")
        if "top" not in self.config["gmsh"]["domain"]:
            logger.debug(f"no <top> entry in {self.config['gmsh']['domain']}")

    def build_2dmesh(self):
        """
        **Builds** the surface mesh of the computational domain.
        """

        # add points and lines for the domain definition
        x_le, y_le = [0, 0]
        x_in, y_in = [x_le - self.dinlet, y_le]
        x_out, y_out = [x_le + self.doutlet, y_le]

        # construction points
        bottom_left = gmsh.model.geo.addPoint(x_in, y_in, 0.)
        bottom_le = gmsh.model.geo.addPoint(x_le, y_le, 0.)
        top_le = gmsh.model.geo.addPoint(x_le, y_le + self.dtop, 0.)
        bottom_right = gmsh.model.geo.addPoint(x_out, y_out, 0.)
        top_left = gmsh.model.geo.addPoint(x_in, y_in + self.dtop, 0.)
        top_right = gmsh.model.geo.addPoint(x_out, y_out + self.dtop, 0.)

        # boundary lines
        bottom_upstream = gmsh.model.geo.addLine(bottom_left, bottom_le)
        bottom_plate = gmsh.model.geo.addLine(bottom_le, bottom_right)
        outlet = gmsh.model.geo.addLine(bottom_right, top_right)
        top_plate = gmsh.model.geo.addLine(top_le, top_right)
        top_upstream = gmsh.model.geo.addLine(top_left, top_le)
        inlet = gmsh.model.geo.addLine(bottom_left, top_left)

        # inner line
        inner_line = gmsh.model.geo.addLine(bottom_le, top_le)

        # transfinite curves on non-blade boundaries
        gmsh.model.geo.mesh.setTransfiniteCurve(
            bottom_upstream, self.snodes[0], "Progression", 1 / self.c_spro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            top_upstream, self.snodes[0], "Progression", 1 / self.c_spro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            bottom_plate, self.snodes[1], "Progression", self.c_spro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            top_plate, self.snodes[1], "Progression", self.c_spro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            inlet, self.nodes_inlet, "Progression", self.c_npro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            outlet, self.nodes_inlet, "Progression", self.c_npro
        )
        gmsh.model.geo.mesh.setTransfiniteCurve(
            inner_line, self.nodes_inlet, "Progression", self.c_npro
        )

        # closed curve loop and computational domain surface definition
        cloop = [bottom_upstream, inner_line, -top_upstream, -inlet]
        left_loop = gmsh.model.geo.addCurveLoop(cloop)
        surf_1 = gmsh.model.geo.addPlaneSurface([left_loop], tag=1001)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_1)
        cloop = [bottom_plate, outlet, -top_plate, -inner_line]
        right_loop = gmsh.model.geo.addCurveLoop(cloop)
        surf_2 = gmsh.model.geo.addPlaneSurface([right_loop], tag=1002)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_2)
        self.surf_tag = [surf_1, surf_2]

        # define physical groups for boundary conditions
        gmsh.model.geo.addPhysicalGroup(1, [bottom_upstream], tag=100)
        logger.debug(f"BC: Upstream tags are {[bottom_upstream]}")
        gmsh.model.geo.addPhysicalGroup(1, [bottom_plate], tag=200)
        logger.debug(f"BC: Wall tags are {[bottom_plate]}")
        gmsh.model.geo.addPhysicalGroup(1, [inlet], tag=300)
        logger.debug(f"BC: Inlet tags are {[inlet]}")
        gmsh.model.geo.addPhysicalGroup(1, [outlet], tag=400)
        logger.debug(f"BC: Outlet tags are {[outlet]}")
        gmsh.model.geo.addPhysicalGroup(1, [top_plate, top_upstream], tag=500)
        logger.debug(f"BC: Top tags are {[top_plate, top_upstream]}")
        gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=600)
