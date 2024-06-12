import gmsh
import logging

from src.mesh.naca_base_mesh import NACABaseMesh

logger = logging.getLogger(__name__)


class NACABlockMesh(NACABaseMesh):
    """
    This class implements a blocking mesh routine for a naca profile based on:</br>
    https://github.com/ComputationalDomain/CMesh_rae69ck-il
    """
    def __init__(self, config: dict, datfile: str = ""):
        """
        Instantiates the BlockMesh object.

        **Input**

        - config (dict): the config file dictionary.
        - dat_file (str): path to input_geometry.dat.
        """
        super().__init__(config, datfile)

    def build_2dmesh(self):
        """
        **Builds** the surface mesh of the computational domain.

        **Inner**

        - R (float): radius of the outer circle.
        - d_out (float): distance to the outlet.
        - offset (int): offset from leading edge.
        - b_width (float): block_width.
        - n_inlet (int): nbr of lead edge & inlet nodes.
        - n_vertical (int) : nbr of out & verti nodes.
        - r_vertical (float): out & vert growth.
        - n_airfoil (int): nbr of nodes on each sides.
        - r_airfoil (float): airfoil sides growth.
        - n_wake (int): nbr of nodes in the wake dir.
        - r_wake (float): wake growth.
        """
        R = self.dinlet
        d_out = self.doutlet
        offset = self.config["gmsh"]["domain"].get("le_offset", 10)
        b_width = self.config["gmsh"]["domain"].get("block_width", 10)
        n_inlet = self.config["gmsh"]["mesh"].get("n_inlet", 60)
        n_vertical = self.config["gmsh"]["mesh"].get("n_vertical", 90)
        r_vertical = self.config["gmsh"]["mesh"].get("r_vertical", 1 / 0.95)
        n_airfoil = self.config["gmsh"]["mesh"].get("n_airfoil", 50)
        r_airfoil = self.config["gmsh"]["mesh"].get("r_airfoil", 1)
        n_wake = self.config["gmsh"]["mesh"].get("n_wake", 100)
        r_wake = self.config["gmsh"]["mesh"].get("r_wake", 1 / 0.95)

        _, self.idx_le = min((p[0], idx) for (idx, p) in enumerate(self.pts))
        _, self.idx_te = max((p[0], idx) for (idx, p) in enumerate(self.pts))
        u_side, l_side = self.split_naca()

        # add points and lines for the naca lower and upper parts
        x_le, y_le = self.pts[self.idx_le][:2]
        x_te, y_te = self.pts[self.idx_te][:2]
        te_tag = gmsh.model.geo.addPoint(x_te, y_te, 0.)
        le_tag = gmsh.model.geo.addPoint(x_le, y_le, 0.)
        pt_u = [te_tag] + [gmsh.model.geo.addPoint(p[0], p[1], p[2]) for p in u_side]
        pt_l = [le_tag] +\
               [gmsh.model.geo.addPoint(p[0], p[1], p[2]) for p in l_side] + [te_tag]

        # airfoil boundary
        spline_low = gmsh.model.geo.addSpline(pt_l[offset:], tag=1)
        spline_le = gmsh.model.geo.addSpline(pt_u[-offset:] + pt_l[:offset + 1], tag=2)
        spline_up = gmsh.model.geo.addSpline(pt_u[:-offset + 1], tag=3)

        # domain and block construction points
        pt_229 = gmsh.model.geo.addPoint(x_te - b_width, R, 0., tag=229)
        pt_230 = gmsh.model.geo.addPoint(x_te - b_width, -R, 0., tag=230)
        pt_231 = gmsh.model.geo.addPoint(x_te, R, 0., tag=231)
        pt_232 = gmsh.model.geo.addPoint(x_te, -R, 0., tag=232)
        pt_233 = gmsh.model.geo.addPoint(d_out, R, 0., tag=233)
        pt_234 = gmsh.model.geo.addPoint(d_out, -R, 0., tag=234)
        pt_235 = gmsh.model.geo.addPoint(d_out, 0, 0., tag=235)

        # domain and block lines
        circle_4 = gmsh.model.geo.addCircleArc(pt_230, te_tag, pt_229)
        line_5 = gmsh.model.geo.addLine(pt_u[-offset], pt_229)
        line_6 = gmsh.model.geo.addLine(pt_l[offset], pt_230)
        line_7 = gmsh.model.geo.addLine(pt_229, pt_231)
        line_8 = gmsh.model.geo.addLine(pt_230, pt_232)
        line_9 = gmsh.model.geo.addLine(pt_231, pt_233)
        line_10 = gmsh.model.geo.addLine(pt_232, pt_234)
        line_11 = gmsh.model.geo.addLine(pt_235, pt_234)
        line_12 = gmsh.model.geo.addLine(pt_235, pt_233)
        line_13 = gmsh.model.geo.addLine(te_tag, pt_231)
        line_14 = gmsh.model.geo.addLine(te_tag, pt_232)
        line_15 = gmsh.model.geo.addLine(te_tag, pt_235)

        # meshing parameters
        gmsh.model.geo.mesh.setTransfiniteCurve(circle_4, n_inlet, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_le, n_inlet, "Progression", 1)
        _ = [gmsh.model.geo.mesh.setTransfiniteCurve(lid, n_vertical, "Progression", r_vertical)
             for lid in [line_5, line_6, line_11, line_12, line_13, line_14]]
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_low, n_airfoil, "Bump", r_airfoil)
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_up, n_airfoil, "Bump", r_airfoil)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_7, n_airfoil, "Bump", 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_8, n_airfoil, "Bump", 2)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_15, n_wake, "Progression", r_wake)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_9, n_wake, "Bump", 0.2)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_10, n_wake, "Bump", 0.2)

        # domain and block surfaces
        cloop_1 = gmsh.model.geo.addCurveLoop([circle_4, -line_5, spline_le, line_6])
        surf_1 = gmsh.model.geo.addPlaneSurface([cloop_1], tag=1001)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_1)

        cloop_2 = gmsh.model.geo.addCurveLoop([line_5, line_7, -line_13, spline_up])
        surf_2 = gmsh.model.geo.addPlaneSurface([cloop_2], tag=1002)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_2)

        cloop_3 = gmsh.model.geo.addCurveLoop([line_13, line_9, -line_12, -line_15])
        surf_3 = gmsh.model.geo.addPlaneSurface([cloop_3], tag=1003)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_3)

        cloop_4 = gmsh.model.geo.addCurveLoop([line_6, line_8, -line_14, -spline_low])
        surf_4 = gmsh.model.geo.addPlaneSurface([cloop_4], tag=1004)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_4)

        cloop_5 = gmsh.model.geo.addCurveLoop([line_14, line_10, -line_11, -line_15])
        surf_5 = gmsh.model.geo.addPlaneSurface([cloop_5], tag=1005)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_5)

        # physical groups
        self.surf_tag = [surf_1, surf_2, surf_3, -surf_5, -surf_4]
        gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=100)
        gmsh.model.geo.addPhysicalGroup(1, [circle_4, line_7, line_8], tag=10)
        logger.info(f"BC: Inlet tags are {[circle_4, line_7, line_8]}")
        gmsh.model.geo.addPhysicalGroup(1, [line_11, line_12], tag=20)
        logger.info(f"BC: Outlet tags are {[line_11, line_12]}")
        gmsh.model.geo.addPhysicalGroup(1, [line_9, line_10], tag=40)
        logger.info(f"BC: Side tags are {[line_9, line_10]}")
        gmsh.model.geo.addPhysicalGroup(1, [spline_up, spline_le, spline_low], tag=30)
        logger.info(f"BC: Wall tags are {[spline_up, spline_le, spline_low]}")

        # non-corner points defined as flow-field and inner block line nodes
        self.non_corner_tag.extend([abs(s_tag) for s_tag in self.surf_tag])
        self.non_corner_tag.extend([abs(li) for li in [line_5, line_6, line_13, line_14, line_15]])
