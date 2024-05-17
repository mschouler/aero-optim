import gmsh
import logging
import numpy as np

from src.mesh.mesh import Mesh

logger = logging.getLogger(__name__)


class CascadeMesh(Mesh):
    """
    This class implements a mesh routine for a compressor cascade geometry.
    """
    def __init__(self, config: dict, datfile: str = ""):
        """
        Instantiates the CascadeMesh object.

        **Input**

        - config (dict): the config file dictionary.
        - dat_file (str): path to input_geometry.dat.

        **Inner**

        - bl_sizefar (float): boundary layer mesh size far from the curves.
        """
        super().__init__(config, datfile)
        self.bl_sizefar: float = config["gmsh"]["mesh"].get("bl_sizefar", 1e-5)
        self.nodes_inlet: int = self.config["gmsh"]["mesh"].get("nodes_inlet", 25)
        self.nodes_outlet: int = self.config["gmsh"]["mesh"].get("nodes_outlet", 17)
        self.snodes: int = self.config["gmsh"]["mesh"].get("side_nodes", 31)
        self.c_snodes: int = self.config["gmsh"]["mesh"].get("curved_side_nodes", 7)

    def process_config(self):
        logger.info("processing config..")
        if "inlet" not in self.config["gmsh"]["domain"]:
            logger.warning(f"no <inlet> entry in {self.config['gmsh']['domain']}")
        if "outlet" not in self.config["gmsh"]["domain"]:
            logger.warning(f"no <outlet> entry in {self.config['gmsh']['domain']}")

    def reorder_blade(self) -> list[list[float]]:
        """
        **Returns** the blade profile after reordering.
        """
        d = np.sqrt([abs(x**2 + y**2) for x, y, _ in self.pts])
        start = np.argmin(d)
        if self.pts[start + 1][1] > self.pts[start][1]:
            return [[p[0], p[1], p[2]] for p in self.pts[start:] + self.pts[:start]]
        else:
            return [[p[0], p[1], p[2]] for p in self.pts[:start] + self.pts[start:]]

    def build_bl(self, blade_tag: list[int]):
        """
        **Builds** the boundary layer around the blade.
        """
        f_bl = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.setNumbers(f_bl, 'CurvesList', blade_tag)
        gmsh.model.mesh.field.setNumber(f_bl, 'Size', self.bl_size)
        gmsh.model.mesh.field.setNumber(f_bl, 'Ratio', self.bl_ratio)
        gmsh.model.mesh.field.setNumber(f_bl, 'Quads', int(self.structured))
        gmsh.model.mesh.field.setNumber(f_bl, 'Thickness', self.bl_thickness)
        gmsh.model.mesh.field.setNumber(f_bl, 'SizeFar', self.bl_sizefar)
        gmsh.model.mesh.field.setAsBoundaryLayer(f_bl)

    def build_cylinder_field(
            self, radius: float, VIn: float, VOut: float,
            XAxis: float, XCenter: float,
            YAxis: float, YCenter: float,
            ZAxis: float = 0.
    ) -> int:
        """
        **Builds** a cylinder field in the computational domain.
        """
        f_cyl = gmsh.model.mesh.field.add('Cylinder')
        gmsh.model.mesh.field.setNumber(f_cyl, 'Radius', radius)
        gmsh.model.mesh.field.setNumber(f_cyl, 'VIn', VIn)
        gmsh.model.mesh.field.setNumber(f_cyl, 'VOut', VOut)
        gmsh.model.mesh.field.setNumber(f_cyl, 'XAxis', XAxis)
        gmsh.model.mesh.field.setNumber(f_cyl, 'XCenter', XCenter)
        gmsh.model.mesh.field.setNumber(f_cyl, 'YAxis', YAxis)
        gmsh.model.mesh.field.setNumber(f_cyl, 'YCenter', YCenter)
        gmsh.model.mesh.field.setNumber(f_cyl, 'ZAxis', ZAxis)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_cyl)
        return f_cyl

    def build_minaniso_field(self, tag: list[int]):
        """
        **Builds** a MinAniso field in the computational domain.
        """
        f_minaniso = gmsh.model.mesh.field.add('MinAniso')
        gmsh.model.mesh.field.setNumbers(f_minaniso, 'FieldsList', tag)
        gmsh.model.mesh.field.setAsBackgroundMesh(f_minaniso)

    def build_2dmesh(self):
        """
        **Builds** the surface mesh of the computational domain.
        """
        wall = self.reorder_blade()
        pt_wall = [gmsh.model.geo.addPoint(p[0], p[1], p[2]) for p in wall]

        # blade splines and transfinite curves
        spl_1 = gmsh.model.geo.addSpline(pt_wall[:35])
        spl_2 = gmsh.model.geo.addSpline(pt_wall[35 - 1:88])
        spl_3 = gmsh.model.geo.addSpline(pt_wall[88 - 1:129])
        spl_4 = gmsh.model.geo.addSpline(pt_wall[129 - 1:157])
        spl_5 = gmsh.model.geo.addSpline(pt_wall[157 - 1:168])
        spl_6 = gmsh.model.geo.addSpline(pt_wall[168 - 1:179])
        spl_7 = gmsh.model.geo.addSpline(pt_wall[179 - 1:245])
        spl_8 = gmsh.model.geo.addSpline(pt_wall[245 - 1:287])
        spl_9 = gmsh.model.geo.addSpline(pt_wall[287 - 1:322] + [pt_wall[0]])
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_1, 8, "Progression", 1.02)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_2, 42, "Progression", 1.03)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_3, 42, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_4, 14, "Progression", 0.94)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_5, 8, "Progression", 0.97)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_6, 8, "Progression", 1.025)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_7, 57, "Progression", 1.015)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_8, 32, "Progression", 0.955)
        gmsh.model.geo.mesh.setTransfiniteCurve(spl_9, 8, "Progression", 0.9)
        spl_list = [spl_1, spl_2, spl_3, spl_4, spl_5, spl_6, spl_7, spl_8, spl_9]
        blade_loop = gmsh.model.geo.addCurveLoop(spl_list)

        # domain construction points
        pt_323 = gmsh.model.geo.addPoint(-6e-2, -5e-2, 0.)
        pt_324 = gmsh.model.geo.addPoint(-1.5e-2, -2.5e-2, 0.)
        pt_325 = gmsh.model.geo.addPoint(0., -1.7e-2, 0.)
        pt_326 = gmsh.model.geo.addPoint(1.270973e-02, -1.164466e-02, 0.)
        pt_327 = gmsh.model.geo.addPoint(2.585445e-02, -7.360298e-03, 0.)
        pt_328 = gmsh.model.geo.addPoint(3.934429e-02, -4.053609e-03, 0.)
        pt_329 = gmsh.model.geo.addPoint(5.308943e-02, -1.631280e-03, 0.)
        pt_330 = gmsh.model.geo.addPoint(6.7e-2, 0., 0.)
        pt_331 = gmsh.model.geo.addPoint(1.3e-1, 0., 0.)
        pt_332 = gmsh.model.geo.addPoint(1.3e-1, 4.039e-2, 0.)
        pt_333 = gmsh.model.geo.addPoint(6.7e-2, 4.039e-2, 0.)
        pt_334 = gmsh.model.geo.addPoint(5.308943e-02, 3.875872e-02, 0.)
        pt_335 = gmsh.model.geo.addPoint(3.934429e-02, 3.633639e-02, 0.)
        pt_336 = gmsh.model.geo.addPoint(2.585445e-02, 3.302970e-02, 0.)
        pt_337 = gmsh.model.geo.addPoint(1.270973e-02, 2.874534e-02, 0.)
        pt_338 = gmsh.model.geo.addPoint(0., 2.339e-2, 0.)
        pt_339 = gmsh.model.geo.addPoint(-1.5e-2, 1.539e-2, 0.)
        pt_340 = gmsh.model.geo.addPoint(-6e-2, -9.61e-3, 0.)

        # domain construction lines
        l_10 = gmsh.model.geo.addLine(pt_340, pt_323)
        l_11 = gmsh.model.geo.addLine(pt_323, pt_324)
        l_12 = gmsh.model.geo.addLine(pt_339, pt_340)
        l_13 = gmsh.model.geo.addLine(pt_324, pt_339)
        l_14 = gmsh.model.geo.addLine(pt_324, pt_325)
        l_15 = gmsh.model.geo.addLine(pt_325, pt_326)
        l_16 = gmsh.model.geo.addLine(pt_326, pt_327)
        l_17 = gmsh.model.geo.addLine(pt_327, pt_328)
        l_18 = gmsh.model.geo.addLine(pt_328, pt_329)
        l_19 = gmsh.model.geo.addLine(pt_329, pt_330)
        l_20 = gmsh.model.geo.addLine(pt_330, pt_331)
        l_21 = gmsh.model.geo.addLine(pt_331, pt_332)
        l_22 = gmsh.model.geo.addLine(pt_332, pt_333)
        l_23 = gmsh.model.geo.addLine(pt_333, pt_334)
        l_24 = gmsh.model.geo.addLine(pt_334, pt_335)
        l_25 = gmsh.model.geo.addLine(pt_335, pt_336)
        l_26 = gmsh.model.geo.addLine(pt_336, pt_337)
        l_27 = gmsh.model.geo.addLine(pt_337, pt_338)
        l_28 = gmsh.model.geo.addLine(pt_338, pt_339)

        # transfinite curves on non-blade boundaries
        gmsh.model.geo.mesh.setTransfiniteCurve(l_10, self.nodes_inlet, "Progression", 1.)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_13, self.nodes_inlet, "Progression", 1.)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_21, self.nodes_outlet, "Progression", 1.)
        # bottom / top periodicity
        bottom_tags = [l_11, l_14, l_15, l_16, l_17, l_18, l_19, l_20]
        top_tags = [l_12, l_28, l_27, l_26, l_25, l_24, l_23, l_22]
        # bottom non-curved side nodes
        _ = [gmsh.model.geo.mesh.setTransfiniteCurve(l_i, self.snodes, "Progression", 1.)
             for l_i in [l_11, l_20]]
        # bottom curved side nodes
        _ = [gmsh.model.geo.mesh.setTransfiniteCurve(l_i, self.c_snodes, "Progression", 1.)
             for l_i in bottom_tags[1:-1]]
        # periodic boundaries /y direction
        gmsh.model.geo.synchronize()
        translation = [1, 0, 0, 0, 0, 1, 0, 0.04039, 0, 0, 1, 0, 0, 0, 0, 1]
        for tid, bid in zip(top_tags, bottom_tags):
            gmsh.model.mesh.setPeriodic(1, [tid], [bid], translation)

        # closed curve loop and computational domain surface definition
        cloop_2 = gmsh.model.geo.addCurveLoop([l_10, l_11, l_12, l_13])
        cloop_3 = gmsh.model.geo.addCurveLoop(
            [-l_13, l_14, l_15, l_16, l_17, l_18, l_19, l_20,
             l_21, l_22, l_23, l_24, l_25, l_26, l_27, l_28]
        )
        surf_1 = gmsh.model.geo.addPlaneSurface([cloop_2], tag=1002)
        surf_2 = gmsh.model.geo.addPlaneSurface([cloop_3, blade_loop], tag=1003)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_1)

        # Fields definition
        # Boundary Layer
        self.build_bl(spl_list) if self.bl else 0
        # Cylinder #1
        f_cyl1 = self.build_cylinder_field(
            9e-3, 8e-4, 5e-3, 1.675e-2, 8.364e-2, -1.171e-3, 1.9754e-2
        )
        # Cylinder #2
        f_cyl2 = self.build_cylinder_field(
            1.62e-2, 1.6e-3, 5e-3, 2.01e-2, 8.699e-2, -1.406e-3, 1.9519e-2
        )
        # MinAniso
        self.build_minaniso_field([f_cyl1, f_cyl2])

        # define physical groups for boundary conditions
        self.surf_tag = [surf_1, surf_2]
        gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=100)
        gmsh.model.geo.addPhysicalGroup(1, [l_10], tag=10)
        logger.info(f"BC: Inlet tags are {[l_10]}")
        gmsh.model.geo.addPhysicalGroup(1, [l_21], tag=20)
        logger.info(f"BC: Outlet tags are {[l_21]}")
        gmsh.model.geo.addPhysicalGroup(1, spl_list, tag=30)
        logger.info(f"BC: Wall tags are {spl_list}")
        gmsh.model.geo.addPhysicalGroup(1, top_tags, tag=40)
        logger.info(f"BC: Top tags are {top_tags}")
        gmsh.model.geo.addPhysicalGroup(1, bottom_tags, tag=50)
        logger.info(f"BC: Bottom tags are {bottom_tags}")

        # non-corner points defined as flow-field and inner block line nodes
        self.non_corner_tag.extend([abs(s_tag) for s_tag in self.surf_tag])
        self.non_corner_tag.append(abs(l_13))
