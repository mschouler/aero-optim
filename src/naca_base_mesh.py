import gmsh

from .mesh import Mesh, plot_quality, set_display, split_view


class NACABaseMesh(Mesh):
    """
    This class implements a meshing routine for a naca profile.
    >> The computational domain has the following structure:

                    pt_hi_inlet
                       !   top_side
                       !      !
                    *  * ------------- * pt_hi_outlet
              *  *  *                  |
           *  *                        |
           *                           |
        *  *                           |
        *                              |
        *   <- arc_inlet               | <- outlet
        *                              |
        *  *                           |
           *                           |
           *  *                        |
              *  *  *                  |
                    *  * ------------- * pt_low_outlet
                       !        !
                       !   bottom_side
                    pt_low_inlet

    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.dinlet: float = self.config["domain"].get("inlet", 2)
        self.doutlet: float = self.config["domain"].get("outlet", 10)
        self.nodes_inlet: int = self.config["mesh"].get("nodes_inlet", 100)
        self.nodes_outlet: int = self.config["mesh"].get("nodes_outlet", 100)
        self.snodes: int = self.config["mesh"].get("side_nodes", 100)

    def process_config(self):
        if "inlet" not in self.config["domain"]:
            print(f"WARNING -- no <inlet> entry in {self.config}[domain]")
        if "outlet" not in self.config["domain"]:
            print(f"WARNING -- no <outlet> entry in {self.config}[domain]")

    def build_mesh(self):
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.logger.start()
        gmsh.model.add("model")

        self.build_2dmesh()

        if self.structured:
            [gmsh.model.geo.mesh.setRecombine(2, abs(id)) for id in self.surf_tag]
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # visualization
        if self.quality:
            plot_quality()
        elt_type = "Mesh.Triangles" if not self.structured else "Mesh.Quadrangles"
        color = [
            ("General.BackgroundGradient", 255, 255, 255),
            (elt_type, 255, 0, 0)
        ]
        number = [
            ("Geometry.Points", 0),
            ("Geometry.Curves", 0),
            ("Mesh.ColorCarousel", 0),
        ]
        if not self.quality:
            number.append(("Mesh.SurfaceFaces", 1))
        set_display(color, number)
        split_view(self.nview) if self.nview > 1 else 0

        # output
        gmsh.fltk.run() if self.GUI else 0
        self.write_outputs()
        gmsh.logger.stop()
        gmsh.finalize()

    def split_naca(self) -> tuple[list[list[float]], list[list[float]]]:
        """
        Return the upper and lower parts of the airfoil as ordered lists (wrt the x axis).
        Note:
            The trailing and leading edges are voluntarily excluded from both parts
            since the geometry is closed and these points must each have a unique tag.
        """
        start: int = min(self.idx_le, self.idx_te)
        end: int = max(self.idx_le, self.idx_te)
        if (
            max([p[1] for p in self.pts[start:end + 1]])
            > max([p[1] for p in self.pts[:start + 1] + self.pts[end:]])
        ):
            upper = [[p[0], p[1], p[2]] for p in self.pts[start + 1:end]]
            lower = [[p[0], p[1], p[2]] for p in self.pts[:start] + self.pts[end + 1:]]
        else:
            lower = [[p[0], p[1], p[2]] for p in self.pts[start + 1:end]]
            upper = [[p[0], p[1], p[2]] for p in self.pts[:start] + self.pts[end + 1:]]
        upper = sorted(upper, key=lambda x: (x[0]), reverse=True)
        lower = sorted(lower, key=lambda x: (x[0]))
        return upper, lower

    def build_bl(self, naca_tag: list[int], te_tag: int):
        """
        Build the boundary layer around the blade part.
        """
        self.f = gmsh.model.mesh.field.add('BoundaryLayer')
        gmsh.model.mesh.field.setNumbers(self.f, 'CurvesList', naca_tag)
        gmsh.model.mesh.field.setNumber(self.f, 'Size', self.bl_size)
        gmsh.model.mesh.field.setNumber(self.f, 'Ratio', self.bl_ratio)
        gmsh.model.mesh.field.setNumber(self.f, 'Quads', int(self.structured))
        gmsh.model.mesh.field.setNumber(self.f, 'Thickness', self.bl_thickness)
        gmsh.model.mesh.field.setNumber(self.f, 'IntersectMetrics', 1)
        gmsh.option.setNumber('Mesh.BoundaryLayerFanElements', 10)
        gmsh.model.mesh.field.setNumbers(self.f, 'FanPointsList', [te_tag])
        gmsh.model.mesh.field.setAsBoundaryLayer(self.f)

    def build_2dmesh(self):
        """
        Build the surface mesh of the computational domain.
        """
        _, self.idx_le = min((p[0], idx) for (idx, p) in enumerate(self.pts))
        _, self.idx_te = max((p[0], idx) for (idx, p) in enumerate(self.pts))
        u_side, l_side = self.split_naca()

        # add points and lines for the naca lower and upper parts
        x_le, y_le = self.pts[self.idx_le][:2]
        x_te, y_te = self.pts[self.idx_te][:2]
        te_tag = gmsh.model.geo.addPoint(x_te, y_te, 0.)
        le_tag = gmsh.model.geo.addPoint(x_le, y_le, 0., self.elt_size)
        pt_u = [te_tag] + [gmsh.model.geo.addPoint(p[0], p[1], p[2], self.elt_size) for p in u_side]
        pt_l = [le_tag] +\
               [gmsh.model.geo.addPoint(p[0], p[1], p[2], self.elt_size) for p in l_side] + [te_tag]

        # airfoil boundary
        offset = 10
        low = 70
        up = 70
        le = 20
        spline_low = gmsh.model.geo.addSpline(pt_l[offset:], tag=1)
        spline_le = gmsh.model.geo.addSpline(pt_u[-offset:] + pt_l[:offset + 1], tag=2)
        spline_up = gmsh.model.geo.addSpline(pt_u[:-offset + 1], tag=3)
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_low, low, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_le, le, "Bump", 2.)
        gmsh.model.geo.mesh.setTransfiniteCurve(spline_up, up, "Progression", 1)
        naca_loop = gmsh.model.geo.addCurveLoop([spline_low, spline_le, spline_up])

        # boundary layer
        self.build_bl([spline_low, spline_le, spline_up], te_tag) if self.bl else 0

        # construction points
        pt_hi_inlet = gmsh.model.geo.addPoint(x_te, y_te + self.dinlet, 0.)
        pt_low_inlet = gmsh.model.geo.addPoint(x_te, y_te - self.dinlet, 0.)
        pt_hi_outlet = gmsh.model.geo.addPoint(x_te + self.doutlet, y_te + self.dinlet, 0.)
        pt_low_outlet = gmsh.model.geo.addPoint(x_te + self.doutlet, y_te - self.dinlet, 0.)

        # non-blade boundary lines
        arc_inlet = gmsh.model.geo.addCircleArc(pt_hi_inlet, te_tag, pt_low_inlet)
        top_side = gmsh.model.geo.addLine(pt_hi_inlet, pt_hi_outlet)
        bottom_side = gmsh.model.geo.addLine(pt_low_outlet, pt_low_inlet)
        outlet = gmsh.model.geo.addLine(pt_hi_outlet, pt_low_outlet)

        # transfinite curves on non-blade boundaries
        gmsh.model.geo.mesh.setTransfiniteCurve(arc_inlet, self.nodes_inlet, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(top_side, self.snodes, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(bottom_side, self.snodes, "Progression", 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(outlet, self.nodes_outlet, "Progression", 1)

        # closed curve loop and computational domain surface definition
        cloop = [-arc_inlet, top_side, outlet, bottom_side]
        boundary_loop = gmsh.model.geo.addCurveLoop(cloop)
        self.surf_tag = gmsh.model.geo.addPlaneSurface([boundary_loop, naca_loop])

        # define physical groups for boundary conditions
        gmsh.model.geo.addPhysicalGroup(1, [spline_low, spline_le, spline_up], tag=100)
        print(f">> BC: Wall tags are {[spline_low, spline_le, spline_up]}")
        gmsh.model.geo.addPhysicalGroup(1, [arc_inlet], tag=200)
        print(f">> BC: Inlet tags are {[arc_inlet]}")
        gmsh.model.geo.addPhysicalGroup(1, [outlet], tag=300)
        print(f">> BC: Outlet tags are {[outlet]}")
        gmsh.model.geo.addPhysicalGroup(1, [top_side], tag=400)
        print(f">> BC: Top tags are {[top_side]}")
        gmsh.model.geo.addPhysicalGroup(1, [bottom_side], tag=500)
        print(f">> BC: Bottom tags are {[bottom_side]}")
        gmsh.model.geo.addPhysicalGroup(2, [self.surf_tag], tag=600)
