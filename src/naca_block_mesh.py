import gmsh

from .naca_base_mesh import NACABaseMesh


class NACABlockMesh(NACABaseMesh):
    """
    This class implements a blocking mesh routine for a naca profile based on:
    https://github.com/ComputationalDomain/CMesh_rae69ck-il
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def build_2dmesh(self):
        """
        Build the surface mesh of the computational domain.
        """
        R = self.dinlet  # radius of the outer circle
        d_out = self.doutlet  # distance to the outlet
        offset = self.config["domain"].get("le_offset", 10)  # offset from leading edge
        b_width = self.config["domain"].get("block_width", 10)  # block_width
        n_inlet = self.config["mesh"].get("n_inlet", 60)  # nbr of leading edge & inlet nodes
        n_vertical = self.config["mesh"].get("n_vertical", 90)  # nbr of outlet & vertical nodes
        r_vertical = self.config["mesh"].get("r_vertical", 1 / 0.95)  # outlet & vertical growth
        n_airfoil = self.config["mesh"].get("n_airfoil", 50)  # nbr of airfoil nodes on each sides
        r_airfoil = self.config["mesh"].get("r_airfoil", 1)  # airfoil sides growth
        n_wake = self.config["mesh"].get("n_wake", 100)  # nbr of nodes in the wake direction
        r_wake = self.config["mesh"].get("r_wake", 1 / 0.95)  # wake growth

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
        surf_1 = gmsh.model.geo.addPlaneSurface([cloop_1])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_1)

        cloop_2 = gmsh.model.geo.addCurveLoop([line_5, line_7, -line_13, spline_up])
        surf_2 = gmsh.model.geo.addPlaneSurface([cloop_2])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_2)

        cloop_3 = gmsh.model.geo.addCurveLoop([line_13, line_9, -line_12, -line_15])
        surf_3 = gmsh.model.geo.addPlaneSurface([cloop_3])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_3)

        cloop_4 = gmsh.model.geo.addCurveLoop([line_6, line_8, -line_14, -spline_low])
        surf_4 = gmsh.model.geo.addPlaneSurface([cloop_4])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_4)

        cloop_5 = gmsh.model.geo.addCurveLoop([line_14, line_10, -line_11, -line_15])
        surf_5 = gmsh.model.geo.addPlaneSurface([cloop_5])
        gmsh.model.geo.mesh.setTransfiniteSurface(surf_5)

        # physical groups
        self.surf_tag = [surf_1, surf_2, surf_3, -surf_5, -surf_4]
        gmsh.model.geo.addPhysicalGroup(2, self.surf_tag, tag=100)
        gmsh.model.geo.addPhysicalGroup(1, [circle_4, line_7, line_8], tag=10)
        print(f">> BC: Inlet tags are {[circle_4, line_7, line_8]}")
        gmsh.model.geo.addPhysicalGroup(1, [line_11, line_12], tag=20)
        print(f">> BC: Outlet tags are {[line_11, line_12]}")
        gmsh.model.geo.addPhysicalGroup(1, [line_9, line_10], tag=40)
        print(f">> BC: Side tags are {[line_9, line_10]}")
        gmsh.model.geo.addPhysicalGroup(1, [spline_up, spline_le, spline_low], tag=30)
        print(f">> BC: Wall tags are {[spline_up, spline_le, spline_low]}")
