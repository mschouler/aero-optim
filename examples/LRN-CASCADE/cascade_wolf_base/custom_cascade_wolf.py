import gmsh
import logging
import numpy as np
import os
import sys

from pymoo.core.problem import Problem
from scipy.spatial.distance import cdist

from aero_optim.geom import (get_area, get_camber_th, get_chords, get_circle,
                             get_cog, split_profile, plot_profile, plot_sides)
from aero_optim.mesh.cascade_mesh import CascadeMesh
from aero_optim.optim.optimizer import WolfOptimizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cascade_adap.custom_cascade import CustomEvolution as WolfCustomEvolution # noqa
from cascade_adap.custom_cascade import CustomOptimizer as WolfCustomOptimizer # noqa
from cascade_adap.custom_cascade import CustomSimulator as WolfCustomSimulator # noqa

logger = logging.getLogger()


def get_valid_center(
        x: np.ndarray, y: np.ndarray, dmin: float, dmax: float,
        le: bool = True, percent: float = 10, resolution: int = 50
) -> np.ndarray | None:
    """
    **Computes** and **returns** the center of a valid circle in regards of the
    leading/trailing edge constraints.

    In particular, it checks if circles of radius dmin can fit in both the leading
    and trailing edges, and if such circles have their centers located at a distance
    to the leading/trailing edge that is smaller than dmax.
    **Returns** None if the constraint is not respected.
    """
    # sort coordinates
    count = int(len(x) * percent / 100)
    indices = np.argsort(x) if le else np.argsort(x)[::-1]
    x_sorted = x[indices][:count]
    y_sorted = y[indices][:count]

    # find bounding box for circle center location search
    profile = np.column_stack([x_sorted, y_sorted])
    min_x, min_y = profile.min(axis=0)
    max_x, max_y = profile.max(axis=0)

    # generate grid of candidate points, starting near leading edge
    x_vals = np.linspace(min_x, max_x, resolution)
    y_vals = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    candidate_points = np.vstack([X.ravel(), Y.ravel()]).T

    # sort candidates by smallest x and on the right side of the le/te edge
    candidate_points = candidate_points[np.argsort(candidate_points[:, 0])]
    candidate_points = (
        candidate_points[candidate_points[:, 0] > profile[0, 0]] if le
        else candidate_points[candidate_points[:, 0] < profile[0, 0]]
    )

    # keep only the candidates at a distance from the le/te edge
    # comprised between dmin and dmax
    dists = cdist(candidate_points, np.array([[x_sorted[0], y_sorted[0]]]))
    idx, _ = np.where((dists > dmin) & (dists < dmax))

    # check if there is at least one pt that gives a valid circle center
    for pt in candidate_points[idx]:
        if np.min(cdist([pt], profile)) > dmin:
            return pt
    return None


class CustomMesh(CascadeMesh):
    def build_2dmesh(self):
        """
        **Builds** the surface mesh of the computational domain.

        The meshing strategy consists of an updated version of the original cascade mesh
        that takes uses the LES mesh based profile as the core geometry.
        """
        self.custom_mesh = self.config["gmsh"]["mesh"].get("custom", True)
        if not self.custom_mesh:
            return super().build_2dmesh()

        wall = self.reorder_blade()
        pt_wall = [gmsh.model.geo.addPoint(p[0], p[1], p[2]) for p in wall]

        # blade splines and transfinite curves
        # 35 -> 5, 287 -> 272 and 322 -> 275
        spl_le = gmsh.model.geo.addSpline(pt_wall[272 - 1:275] + [pt_wall[0]] + pt_wall[:5])
        # 88 -> 69, 129 -> 117 and 157 -> 147
        spl_ss = gmsh.model.geo.addSpline(
            pt_wall[5 - 1:69] + pt_wall[69 - 1:117] + pt_wall[117 - 1:147]
        )
        # 157 -> 147, 168 -> 153 and 179 -> 163
        spl_te = gmsh.model.geo.addSpline(pt_wall[147 - 1:153] + pt_wall[153 - 1:163])
        # 179 -> 163, 245 -> 232 and 287 -> 272
        spl_ps = gmsh.model.geo.addSpline(pt_wall[163 - 1:232] + pt_wall[232 - 1:272])
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


class CustomSimulator(WolfCustomSimulator):
    """
    Same custom class as the one defined in cascade_adap.
    """


class CustomOptimizer(WolfCustomOptimizer):
    """
    Same custom class as the one defined in cascade_adap.

    The optimization strategy consists of an updated version of the original cascade optimizer
    with a more complete set of constraints:
    - the leading/trailing edge radius constraints are enhanced and adapted to
      the LES mesh based profile,
    - constraints related to the outflow angle are also introduced.
    """
    def __init__(self, config: dict):
        """
         **Inner**

        - feasible_cid (dict[int, list[int]]): dictionary containing feasible cid of each gid.
        """
        WolfOptimizer.__init__(self, config)
        Problem.__init__(
            self, n_var=self.n_design, n_obj=2, n_ieq_constr=9, xl=self.bound[0], xu=self.bound[1]
        )
        self.feasible_cid: dict[int, list[int]] = {}

    def set_inner(self):
        """
        **Sets** some baseline quantities required to compute the relative constraints:

        - angle_ADP (float): the outflow angle accepted deviation at ADP
        - angle_OP1 (float): the outflow angle accepted deviation at OP1
        - angle_OP2 (float): the outflow angle accepted deviation at OP2
        """
        super().set_inner()
        self.CoI = self.config["optim"].get("CoI", "OutflowAngle")
        self.angle_ADP = self.config["optim"].get("angle_ADP", 1.6)
        self.angle_OP1 = self.config["optim"].get("angle_OP1", 4)
        self.angle_OP2 = self.config["optim"].get("angle_OP2", 4)

    def _evaluate(self, X: np.ndarray, out: np.ndarray, *args, **kwargs):
        """
        **Computes** the objective function and constraints for each candidate in the generation.

        Note:
            for this use-case, some of the constraints can be computed before simulations.
            Unfeasible candidates are not simulated.
        """
        gid = self.gen_ctr
        self.feasible_cid[gid] = []

        # compute candidates geometric constraints and execute feasible candidates only
        geom_constraints = self.execute_constrained_candidates(X, gid)

        # update candidates fitness
        # Note: this time only the first value in the dataframe should be read
        for cid in range(len(X)):
            if cid in self.feasible_cid[gid]:
                loss_ADP = self.simulator.df_dict[gid][cid]["ADP"][self.QoI].dropna().iloc[-1]
                loss_OP1 = self.simulator.df_dict[gid][cid]["OP1"][self.QoI].dropna().iloc[-1]
                loss_OP2 = self.simulator.df_dict[gid][cid]["OP2"][self.QoI].dropna().iloc[-1]
                logger.info(f"g{gid}, c{cid}: "
                            f"w_ADP = {loss_ADP}, w_OP = {0.5 * (loss_OP1 + loss_OP2)}")
                self.J.append([loss_ADP, 0.5 * (loss_OP1 + loss_OP2)])
            else:
                self.J.append([float("nan"), float("nan")])

        # compute candidates angle constraints
        if not self.constraint:
            angle_constraints = [[-1.] * 3 for _ in range(len(X))]
        else:
            angle_constraints = []
            CoI = self.CoI
            for cid in range(len(X)):
                outflow_angle_ADP = self.simulator.df_dict[gid][cid]["ADP"][CoI].dropna().iloc[-1]
                outflow_angle_OP1 = self.simulator.df_dict[gid][cid]["OP1"][CoI].dropna().iloc[-1]
                outflow_angle_OP2 = self.simulator.df_dict[gid][cid]["OP2"][CoI].dropna().iloc[-1]
                angle_constraints.append(
                    [abs(outflow_angle_ADP) - self.angle_ADP,
                     abs(outflow_angle_OP1) - self.angle_OP1,
                     abs(outflow_angle_OP2) - self.angle_OP2]
                )
                logger.debug(f"g{gid}, c{cid} ADP outflow angle: ({abs(outflow_angle_ADP)})")
                if abs(outflow_angle_ADP) - self.angle_ADP > 0:
                    logger.info(f"g{gid}, c{cid} ADP outflow angle: constraint violation")
                logger.debug(f"g{gid}, c{cid} OP1 outflow angle: ({abs(outflow_angle_OP1)})")
                if abs(outflow_angle_OP1) - self.angle_OP1 > 0:
                    logger.info(f"g{gid}, c{cid} OP1 outflow angle: constraint violation")
                logger.debug(f"g{gid}, c{cid} OP2 outflow angle: ({abs(outflow_angle_OP2)})")
                if abs(outflow_angle_OP2) - self.angle_OP2 > 0:
                    logger.info(f"g{gid}, c{cid} OP2 outflow angle: constraint violation")

        out["F"] = np.vstack(self.J[-self.doe_size:])
        self._observe(out["F"])
        out["G"] = np.column_stack([geom_constraints, np.vstack(angle_constraints)])
        self.gen_ctr += 1

    def apply_candidate_constraints(self, profile: np.ndarray, gid: int, cid: int) -> list[float]:
        """
        **Computes** various relative and absolute constraints of a given candidate
        and **returns** their values as a list of floats.

        Note:
            when some constraint is violated, a graph is also generated.
        """
        if not self.constraint:
            return [-1.] * 6
        # relative constraints
        # thmax / c:        +/- 30%
        # Xthmax / c_ax:    +/- 20%
        upper, lower = split_profile(profile)
        c, c_ax = get_chords(profile)
        camber_line, thmax, Xthmax, th_vec = get_camber_th(upper, lower, interpolate=True)
        th_over_c = thmax / c
        Xth_over_cax = Xthmax / c_ax
        logger.debug(f"th_max = {thmax} m, Xth_max {Xthmax} m")
        logger.debug(f"th_max / c = {th_over_c}, Xth_max / c_ax = {Xth_over_cax}")
        th_cond = abs(th_over_c - self.bsl_th_over_c) / self.bsl_th_over_c - 0.3
        logger.debug(f"th_max / c: {'violated' if th_cond > 0 else 'not violated'} ({th_cond})")
        Xth_cond = abs(Xth_over_cax - self.bsl_Xth_over_cax) / self.bsl_Xth_over_cax - 0.2
        logger.debug(f"Xth_max / c_ax: {'violated' if Xth_cond > 0 else 'not violated'} "
                     f"({Xth_cond})")
        # area / (c * c):   +/- 20%
        area = get_area(profile)
        area_over_c2 = area / c**2
        area_cond = abs(area_over_c2 - self.bsl_area_over_c2) / self.bsl_area_over_c2 - 0.2
        logger.debug(f"area / (c * c): {'violated' if area_cond > 0 else 'not violated'} "
                     f"({area_cond})")
        # X_cg / c_ax:      +/- 20%
        cog = get_cog(profile)
        Xcg_over_cax = cog[0] / c_ax
        cog_cond = abs(Xcg_over_cax - self.bsl_Xcg_over_cax) / self.bsl_Xcg_over_cax - 0.2
        logger.debug(f"X_cg / c_ax: {'violated' if cog_cond > 0 else 'not violated'} ({cog_cond})")
        # absolute constraints
        # leading/trailing edge radii and principal axis condition
        O_le = get_valid_center(
            profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=True
        )
        O_te = get_valid_center(
            profile[:, 0], profile[:, 1], dmin=0.005 * c, dmax=1.4 * 0.005 * c, le=False
        )
        le_circle = get_circle(O_le, 0.005 * c) if O_le is not None else np.array([])
        te_circle = get_circle(O_te, 0.005 * c) if O_te is not None else np.array([])
        # leading edge radius: r_le > 0.5% * c
        logger.debug(f"le radius: {'violated' if O_le is None else 'not violated'}")
        le_cond = 1 if O_le is None else -1
        # trailing edge radius: r_te > 0.5% * c
        logger.debug(f"te radius: {'violated' if O_te is None else 'not violated'}")
        te_cond = 1 if O_te is None else -1
        if cog_cond > 0:
            fig_name = os.path.join(self.figdir, f"profile_g{gid}_c{cid}.png")
            plot_profile(profile, cog, fig_name)
        if (th_cond > 0 or Xth_cond > 0 or area_cond > 0 or le_cond > 0 or te_cond > 0):
            fig_name = os.path.join(self.figdir, f"sides_g{gid}_c{cid}.png")
            plot_sides(upper, lower, camber_line, le_circle, te_circle, th_vec, fig_name)
        return [th_cond, Xth_cond, area_cond, cog_cond, le_cond, te_cond]


class CustomEvolution(WolfCustomEvolution):
    """
    Same custom class as the one defined in cascade_adap.
    """
