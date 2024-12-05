import gmsh
import logging
import numpy as np

from aero_optim.mesh.mesh import Mesh

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

        - doutlet (float): outlet distance to the blade trailing edge.
        - dlr_mesh (bool): builds the DLR provided mesh (True) or a simpler for adaptation (False).
        - bl_sizefar (float): boundary layer mesh size far from the curves.
        - nodes_inlet (int): the number of nodes to mesh the inlet.
        - nodes_outlet (int): the number of nodes to mesh the outlet.
        - snodes_inlet (int): the number of nodes to mesh the inlet top and bottom sides.
        - snodes_outlet (int): the number of nodes to mesh the outlet top and bottom sides.
        - c_snodes (int): the number of nodes to mesh the inner sides.
        - le (int): the number of nodes to mesh the blade leading edge portion.
        - te (int): the number of nodes to mesh the blade trailing edge lower portion.
        - nodes_sp2 (int): the number of nodes to mesh the 1st section of the blade suction side.
        - nodes_sp3 (int): the number of nodes to mesh the 2nd section of the blade suction side.
        - nodes_sp4 (int): the number of nodes to mesh the 3rd section of the blade suction side.
        - nodes_sp7 (int): the number of nodes to mesh the 1st section of the blade pressure side.
        - nodes_sp8 (int): the number of nodes to mesh the 2nd section of the blade pressure side.
        - nodes_ss (int): the number of nodes to mesh the suction side (dlr_mesh set to False).
        - nodes_ps (int): the number of nodes to mesh the pressure side (dlr_mesh set to False).
        - cyl_vin (float): cylinder field parameter Vin.
        - cyl_vout (float): cylinder field parameter Vout.
        - cyl_xaxis (float): cylinder field parameter Xaxis.
        - cyl_xcenter (float): cylinder field parameter Xcenter.

        Note:
            for the DLR configuration, the blade is split into 9 splines (clockwise from the tip):

            * 2 splines (1 and 9) for the leading edge parameterized with **le**
              i.e. each spline has **le**/2 nodes,
            * 2 splines (5 and 6) for the trailing edge parameterized with **te**
              i.e. each spline has **te**/2 nodes,
            * 3 splines for the suction side (2, 3, 4) of lengths 0.027, 0.038 and 0.061 m,
              and parameterized with **nodes_sp2**, **nodes_sp3** and **nodes_sp4**,
            * 2 splines for the pressure side (7, 8) of lengths 0.0526 and 0.0167 m,
              and parameterized with **nodes_sp7**, **nodes_sp8**

        """
        super().__init__(config, datfile)
        self.doutlet: float = self.config["gmsh"]["domain"].get("outlet", 6.3e-2)
        self.dlr_mesh: bool = config["gmsh"]["mesh"].get("DLR_mesh", False)
        self.bl_sizefar: float = config["gmsh"]["mesh"].get("bl_sizefar", 1e-5)
        self.nodes_inlet: int = self.config["gmsh"]["mesh"].get("nodes_inlet", 25)
        self.nodes_outlet: int = self.config["gmsh"]["mesh"].get("nodes_outlet", 17)
        self.snodes_inlet: int = self.config["gmsh"]["mesh"].get("side_nodes_inlet", 31)
        self.snodes_outlet: int = self.config["gmsh"]["mesh"].get("side_nodes_outlet", 31)
        self.c_snodes: int = self.config["gmsh"]["mesh"].get("curved_side_nodes", 7)
        self.le: int = self.config["gmsh"]["mesh"].get("le", 16)
        self.te: int = self.config["gmsh"]["mesh"].get("te", 16)
        self.nodes_sp2: int = self.config["gmsh"]["mesh"].get("nodes_sp2", 42)
        self.nodes_sp3: int = self.config["gmsh"]["mesh"].get("nodes_sp3", 42)
        self.nodes_sp4: int = self.config["gmsh"]["mesh"].get("nodes_sp4", 14)
        self.nodes_sp7: int = self.config["gmsh"]["mesh"].get("nodes_sp7", 57)
        self.nodes_sp8: int = self.config["gmsh"]["mesh"].get("nodes_sp8", 32)
        self.nodes_ss: int = self.config["gmsh"]["mesh"].get("nodes_ss", 400)
        self.nodes_ps: int = self.config["gmsh"]["mesh"].get("nodes_ps", 400)
        self.cyl_vin: float = self.config["gmsh"]["mesh"].get("cyl_vin", 8e-4)
        self.cyl_vout: float = self.config["gmsh"]["mesh"].get("cyl_vout", 5e-3)
        self.cyl_xaxis: float = self.config["gmsh"]["mesh"].get("cyl_xaxis", 1.675e-2)
        self.cyl_xcenter: float = self.config["gmsh"]["mesh"].get("cyl_xcenter", 8.364e-2)

    def process_config(self):
        logger.debug("processing config..")
        if "domain" not in self.config["gmsh"]:
            logger.debug(f"no <domain> entry in {self.config['gmsh']}, empty entry added")
            self.config["gmsh"]["domain"] = {}

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
        if self.dlr_mesh:
            spl_1 = gmsh.model.geo.addSpline(pt_wall[:35])
            spl_2 = gmsh.model.geo.addSpline(pt_wall[35 - 1:88])
            spl_3 = gmsh.model.geo.addSpline(pt_wall[88 - 1:129])
            spl_4 = gmsh.model.geo.addSpline(pt_wall[129 - 1:157])
            spl_5 = gmsh.model.geo.addSpline(pt_wall[157 - 1:168])
            spl_6 = gmsh.model.geo.addSpline(pt_wall[168 - 1:179])
            spl_7 = gmsh.model.geo.addSpline(pt_wall[179 - 1:245])
            spl_8 = gmsh.model.geo.addSpline(pt_wall[245 - 1:287])
            spl_9 = gmsh.model.geo.addSpline(pt_wall[287 - 1:322] + [pt_wall[0]])
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_1, self.le // 2, "Progression", 1.02)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_2, self.nodes_sp2, "Progression", 1.03)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_3, self.nodes_sp3, "Progression", 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_4, self.nodes_sp4, "Progression", 0.94)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_5, self.te // 2, "Progression", 0.97)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_6, self.te // 2, "Progression", 1.025)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_7, self.nodes_sp7, "Progression", 1.015)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_8, self.nodes_sp8, "Progression", 0.955)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_9, self.le // 2, "Progression", 0.9)
            spl_list = [spl_1, spl_2, spl_3, spl_4, spl_5, spl_6, spl_7, spl_8, spl_9]
        else:
            spl_le = gmsh.model.geo.addSpline(pt_wall[287 - 1:322] + [pt_wall[0]] + pt_wall[:35])
            spl_ss = gmsh.model.geo.addSpline(
                pt_wall[35 - 1:88] + pt_wall[88 - 1:129] + pt_wall[129 - 1:157]
            )
            spl_te = gmsh.model.geo.addSpline(pt_wall[157 - 1:168] + pt_wall[168 - 1:179])
            spl_ps = gmsh.model.geo.addSpline(pt_wall[179 - 1:245] + pt_wall[245 - 1:287])
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_le, self.le, "Progression", 1.0)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_ss, self.nodes_ss, "Progression", 1.0)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_te, self.te, "Progression", 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(spl_ps, self.nodes_ps, "Progression", 1.0)
            spl_list = [spl_le, spl_ss, spl_te, spl_ps]
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
        pt_331 = gmsh.model.geo.addPoint(6.7e-2 + self.doutlet, 0., 0.)
        pt_332 = gmsh.model.geo.addPoint(6.7e-2 + self.doutlet, 4.039e-2, 0.)
        pt_333 = gmsh.model.geo.addPoint(6.7e-2, 4.039e-2, 0.)
        pt_334 = gmsh.model.geo.addPoint(5.308943e-02, 3.875872e-02, 0.)
        pt_335 = gmsh.model.geo.addPoint(3.934429e-02, 3.633639e-02, 0.)
        pt_336 = gmsh.model.geo.addPoint(2.585445e-02, 3.302970e-02, 0.)
        pt_337 = gmsh.model.geo.addPoint(1.270973e-02, 2.874534e-02, 0.)
        pt_338 = gmsh.model.geo.addPoint(0., 2.339e-2, 0.)
        pt_339 = gmsh.model.geo.addPoint(-1.5e-2, 1.539e-2, 0.)
        pt_340 = gmsh.model.geo.addPoint(-6e-2, -9.61e-3, 0.)

        # domain construction lines
        l_10 = gmsh.model.geo.addLine(pt_340, pt_323, tag=10)
        l_11 = gmsh.model.geo.addLine(pt_323, pt_324, tag=11)
        l_12 = gmsh.model.geo.addLine(pt_339, pt_340, tag=12)
        if self.dlr_mesh:
            l_13 = gmsh.model.geo.addLine(pt_324, pt_339, tag=13)
        l_14 = gmsh.model.geo.addLine(pt_324, pt_325, tag=14)
        l_15 = gmsh.model.geo.addLine(pt_325, pt_326, tag=15)
        l_16 = gmsh.model.geo.addLine(pt_326, pt_327, tag=16)
        l_17 = gmsh.model.geo.addLine(pt_327, pt_328, tag=17)
        l_18 = gmsh.model.geo.addLine(pt_328, pt_329, tag=18)
        l_19 = gmsh.model.geo.addLine(pt_329, pt_330, tag=19)
        l_20 = gmsh.model.geo.addLine(pt_330, pt_331, tag=20)
        l_21 = gmsh.model.geo.addLine(pt_331, pt_332, tag=21)
        l_22 = gmsh.model.geo.addLine(pt_332, pt_333, tag=22)
        l_23 = gmsh.model.geo.addLine(pt_333, pt_334, tag=23)
        l_24 = gmsh.model.geo.addLine(pt_334, pt_335, tag=24)
        l_25 = gmsh.model.geo.addLine(pt_335, pt_336, tag=25)
        l_26 = gmsh.model.geo.addLine(pt_336, pt_337, tag=26)
        l_27 = gmsh.model.geo.addLine(pt_337, pt_338, tag=27)
        l_28 = gmsh.model.geo.addLine(pt_338, pt_339, tag=28)

        # transfinite curves on non-blade boundaries
        gmsh.model.geo.mesh.setTransfiniteCurve(l_10, self.nodes_inlet, "Progression", 1.)
        if self.dlr_mesh:
            gmsh.model.geo.mesh.setTransfiniteCurve(l_13, self.nodes_inlet, "Progression", 1.)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_21, self.nodes_outlet, "Progression", 1.)
        # bottom / top periodicity
        self.bottom_tags = [l_11, l_14, l_15, l_16, l_17, l_18, l_19, l_20]
        self.top_tags = [l_12, l_28, l_27, l_26, l_25, l_24, l_23, l_22]
        # bottom non-curved side nodes
        gmsh.model.geo.mesh.setTransfiniteCurve(l_11, self.snodes_inlet, "Progression", 1.)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_20, self.snodes_outlet, "Progression", 1.)
        # bottom curved side nodes
        _ = [gmsh.model.geo.mesh.setTransfiniteCurve(l_i, self.c_snodes, "Progression", 1.)
             for l_i in self.bottom_tags[1:-1]]
        # periodic boundaries /y direction
        gmsh.model.geo.synchronize()
        translation = [1, 0, 0, 0, 0, 1, 0, 0.04039, 0, 0, 1, 0, 0, 0, 0, 1]
        for tid, bid in zip(self.top_tags, self.bottom_tags):
            gmsh.model.mesh.setPeriodic(1, [tid], [bid], translation)

        # closed curve loop and computational domain surface definition
        if self.dlr_mesh:
            cloop_2 = gmsh.model.geo.addCurveLoop([l_10, l_11, l_12, l_13])
            cloop_3 = gmsh.model.geo.addCurveLoop(
                [-l_13, l_14, l_15, l_16, l_17, l_18, l_19, l_20,
                 l_21, l_22, l_23, l_24, l_25, l_26, l_27, l_28,]
            )
            surf_1 = gmsh.model.geo.addPlaneSurface([cloop_2], tag=1002)
            surf_2 = gmsh.model.geo.addPlaneSurface([cloop_3, blade_loop], tag=1003)
            gmsh.model.geo.mesh.setTransfiniteSurface(surf_1)
        else:
            cloop_3 = gmsh.model.geo.addCurveLoop(
                [l_10, l_11, l_14, l_15, l_16, l_17, l_18, l_19, l_20,
                 l_21, l_22, l_23, l_24, l_25, l_26, l_27, l_28, l_12]
            )
            surf_2 = gmsh.model.geo.addPlaneSurface([cloop_3, blade_loop], tag=1003)

        # Fields definition
        # Boundary Layer
        self.build_bl(spl_list) if self.bl else 0
        # Cylinder #1
        f_cyl1 = self.build_cylinder_field(
            9e-3,
            self.cyl_vin,
            self.cyl_vout,
            self.cyl_xaxis,
            self.cyl_xcenter,
            -1.171e-3,
            1.9754e-2
        )
        # Cylinder #2
        f_cyl2 = self.build_cylinder_field(
            1.62e-2, 1.6e-3, 5e-3, 2.01e-2, 8.699e-2, -1.406e-3, 1.9519e-2
        )
        # MinAniso
        self.build_minaniso_field([f_cyl1, f_cyl2])

        # define physical groups for boundary conditions
        self.surf_tag = [surf_1, surf_2] if self.dlr_mesh else [surf_2]
        if self.extrusion_layers == 0:
            gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=100, name="fluid")
            gmsh.model.geo.addPhysicalGroup(1, [l_10], tag=10, name="inlet")
            logger.debug(f"2D BC: Inlet tags are {[l_10]}")
            gmsh.model.geo.addPhysicalGroup(1, [l_21], tag=20, name="outlet")
            logger.debug(f"2D BC: Outlet tags are {[l_21]}")
            gmsh.model.geo.addPhysicalGroup(1, spl_list, tag=30, name="wall")
            logger.debug(f"2D BC: Wall tags are {spl_list}")
            gmsh.model.geo.addPhysicalGroup(1, self.top_tags, tag=40, name="periodic_vert_l")
            logger.debug(f"2D BC: Top tags are {self.top_tags}")
            gmsh.model.geo.addPhysicalGroup(1, self.bottom_tags, tag=50, name="periodic_vert_r")
            logger.debug(f"2D BC: Bottom tags are {self.bottom_tags}")
        else:
            gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=100, name="periodic_span_l")

        # non-corner points defined as flow-field, inner block line and wall nodes
        self.non_corner_tags.extend([abs(s_tag) for s_tag in self.surf_tag])
        self.non_corner_tags.extend([abs(s_tag) for s_tag in spl_list])
        if self.dlr_mesh:
            self.non_corner_tags.append(abs(l_13))

    def build_3dmesh(self):
        """
        **Performs** an extrusion along the z axis.
        - h_size (float): the total extruded depth.
        """
        h_size = self.extrusion_size
        self.ext_tag = [gmsh.model.geo.extrude(
            [(2, s)], 0, 0, h_size, [self.extrusion_layers], [1], True) for s in self.surf_tag]
        # retrieve extruded surfaces and volumes
        vol = [tu[-1] for tu in [self.ext_tag[0][1], self.ext_tag[1][1]]]
        top = [tu[-1] for tu in [self.ext_tag[0][0], self.ext_tag[1][0]]]
        # 1st block
        inlet = [self.ext_tag[0][2][-1]]
        perlo = [self.ext_tag[0][3][-1]]
        perup = [self.ext_tag[0][5][-1]]
        # 2nd block
        perlo += [tu[-1] for tu in self.ext_tag[1][3:10]]
        outlet = [self.ext_tag[1][10][-1]]
        perup += [tu[-1] for tu in self.ext_tag[1][11:18]]
        wall = [tu[-1] for tu in self.ext_tag[1][18:]]
        # create physical groups
        gmsh.model.geo.addPhysicalGroup(3, vol, tag=2000, name="fluid")
        logger.debug("3D BC: vol tag is 2000")
        gmsh.model.geo.addPhysicalGroup(2, top, tag=2001, name="periodic_span_r")
        logger.debug("3D BC: periodic_span_l tag is 100")
        logger.debug("3D BC: periodic_span_r tag is 2001")
        gmsh.model.geo.addPhysicalGroup(2, perlo, tag=2003, name="periodic_vert_l")
        logger.debug("3D BC: periodic_vert_l tag is 2003")
        gmsh.model.geo.addPhysicalGroup(2, perup, tag=2004, name="periodic_vert_r")
        logger.debug("3D BC: periodic_vert_r tag is 2004")
        gmsh.model.geo.addPhysicalGroup(2, inlet, tag=2005, name="inlet")
        logger.debug("3D BC: inlet tag is 2005")
        gmsh.model.geo.addPhysicalGroup(2, outlet, tag=2006, name="outlet")
        logger.debug("3D BC: outlet tag is 2006")
        gmsh.model.geo.addPhysicalGroup(2, wall, tag=2007, name="wall")
        logger.debug("3D BC: wall tag is 2007")
        # set 2 tags to none to prevent reformatting
        self.non_corner_tags = None
        self.bottom_tags = None
        self.top_tags = None
