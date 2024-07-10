import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

EPS = 1e-6


def get_area(pts: np.ndarray) -> float:
    """
    **Returns** the geometry signed area computed with the shoelace formula.</br>
    see https://rosettacode.org/wiki/Shoelace_formula_for_polygonal_area#Python
    """
    return 0.5 * np.sum([pts[i - 1, 0] * pts[i, 1] - pts[i, 0] * pts[i - 1, 1]
                         for i in range(len(pts))])


def get_camber_th(
        upper: np.ndarray,
        lower: np.ndarray,
        interpolate: bool = False
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """
    **Returns** an approximation of the camber line, of the geometry absolute (th)
    and axial (th_x) thicknesses, and the coordinates of the points maximizing the thickness.

    If `interpolate` is set to `True`, the upper and lower profiles are interpolated with scipy
    which gives a more precise estimation of the thicknesses.
    """
    min_dx = []
    min_dvec = []
    if interpolate:
        c, _ = get_chords(np.vstack((upper, lower)))
        int_upper_f = scipy.interpolate.interp1d(upper[:, 0], upper[:, 1])
        int_lower_f = scipy.interpolate.interp1d(lower[:, 0], lower[:, 1])
        xnew_lower = np.arange(min(lower[:, 0]) + EPS, max(lower[:, 0]), 0.0025 * c)
        xnew_upper = np.arange(min(upper[:, 0]) + EPS, max(upper[:, 0]), 0.0025 * c)
        int_upper = int_upper_f(xnew_upper)
        int_lower = int_lower_f(xnew_lower)
        upper = np.column_stack((xnew_upper, int_upper))
        lower = np.column_stack((xnew_lower, int_lower))
    for x in upper:
        d_vec = np.sqrt(np.einsum("ij,ij->i", lower - x, lower - x))
        idx_min = np.argmin(d_vec)
        min_dx.append(idx_min)
        min_dvec.append(d_vec[idx_min])
    # lower_n is the vector of points minimizing the distance wrt the upper points
    # such that (upper[i], lower_n[i]) forms the pair of closest points between both sides
    lower_n = lower[min_dx]
    camber_line = (upper + lower_n) / 2.
    th_idx = np.argmax(min_dvec)
    le_x = np.min(np.vstack((upper, lower)), axis=0)[0]
    # the pair of points with the maximal thickness
    th_vec = np.array([upper[th_idx], lower_n[th_idx]])
    return camber_line, min_dvec[th_idx], camber_line[th_idx][0] - le_x, th_vec


def get_chords(pts: np.ndarray) -> tuple[float, float]:
    """
    **Returns** chord (c) and axial chord (c_ax).
    """
    idx_le, idx_te = get_phys_edges_idx(pts)
    return float(np.linalg.norm(pts[idx_le] - pts[idx_te])), float((pts[idx_te] - pts[idx_le])[0])


def get_circle(origin: np.ndarray, r: float) -> np.ndarray:
    """
    **Returns** the coordinates of the points on the circle centered on `origin` with radius `r`.
    """
    theta = np.linspace(0., 2 * np.pi, 100)
    x_c = r * np.cos(theta) + origin[0]
    y_c = r * np.sin(theta) + origin[1]
    return np.column_stack((x_c, y_c))


def get_cog(pts: np.ndarray) -> np.ndarray:
    """
    **Returns** the coordinates of the geometry's center of gravity.
    """
    area = get_area(pts)
    x_cg = np.sum(
        [(pts[i - 1, 0] + pts[i, 0]) * (pts[i - 1, 0] * pts[i, 1] - pts[i, 0] * pts[i - 1, 1])
         for i in range(len(pts))]
    ) / (6 * area)
    y_cg = np.sum(
        [(pts[i - 1, 1] + pts[i, 1]) * (pts[i - 1, 0] * pts[i, 1] - pts[i, 0] * pts[i - 1, 1])
         for i in range(len(pts))]
    ) / (6 * area)
    return np.array([x_cg, y_cg])


def get_curv_abs(pts: np.ndarray) -> np.ndarray:
    """
    **Returns** the curvilinear abscissa vector of the given points.
    """
    s: list[float] = [0.] * len(pts)
    for pt_id in range(1, len(pts)):
        d = np.linalg.norm(pts[pt_id] - pts[pt_id - 1])
        s[pt_id] = s[pt_id - 1] + float(d)
    return np.array(s)


def get_geom_edges_idx(pts: np.ndarray) -> tuple[int, int]:
    """
    **Returns** the indices of the leading/trailing geometric edges
    i.e. computed from the minimal curvature.
    """
    # physical leading/trailing edge
    phy_idx_le, phy_idx_te = get_phys_edges_idx(pts)

    # geometrical leading/trailing edge bounds
    assert abs(phy_idx_le - phy_idx_te) > 60
    le_bnds = (max(0, phy_idx_le - 30), min(len(pts), phy_idx_le + 30))
    te_bnds = (max(0, phy_idx_te - 30), min(len(pts), phy_idx_te + 30))

    # leading edge
    le_pts = pts[le_bnds[0]:le_bnds[-1]]
    le_s_np = get_curv_abs(le_pts)
    lap_s = np.gradient(np.gradient(le_s_np))
    le_idx_lap = le_bnds[0] + np.argmin(abs(lap_s))

    # trailing edge
    te_pts = pts[te_bnds[0]:te_bnds[-1]]
    te_s_np = get_curv_abs(te_pts)
    lap_s = np.gradient(np.gradient(te_s_np))
    te_idx_lap = te_bnds[0] + np.argmin(abs(lap_s))
    return int(le_idx_lap), int(te_idx_lap)


def get_phys_edges_idx(pts: np.ndarray) -> tuple[int, int]:
    """
    **Returns** the indices of the leading/trailing physical edges
    i.e. computed from the left/right-most points.
    """
    return np.argmin(pts, axis=0)[0], np.argmax(pts, axis=0)[0]


def get_orth_proj(
        pts: np.ndarray,
        le_idx: int,
        te_idx: int,
        d: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    **Returns** the orthogonal projection of the leading/trailing edge wrt its tangent
    and at distance `d` inside the geometry.
    """
    # leading edge
    lle = pts[le_idx + 1] - pts[le_idx]
    le_x = pts[le_idx, 0] + (d**2 - d**2 / (1 + lle[1]**2 / lle[0]**2))**0.5
    le_y = pts[le_idx, 1] + d / (1 + lle[1]**2 / lle[0]**2)**0.5
    # trailing edge
    tte = pts[te_idx + 1] - pts[te_idx]
    te_x = pts[te_idx, 0] - (d**2 - d**2 / (1 + tte[1]**2 / tte[0]**2))**0.5
    te_y = pts[te_idx, 1] - d / (1 + tte[1]**2 / tte[0]**2)**0.5
    return np.array([le_x, le_y]), np.array([te_x, te_y])


def get_radius_violation(pts: np.ndarray, origin: np.ndarray, d: float) -> bool:
    """
    **Returns** `True` if a circle of radius `d` centered on `origin` can fit inside `pts`,
    `False` otherwise.
    """
    pts_dist = np.sqrt(np.einsum("ij,ij->i", pts[:, :2] - origin, pts[:, :2] - origin))
    return np.where(pts_dist < d)[0].size > 0


def split_profile(
        pts: np.ndarray,
        idx_le: int = -1,
        idx_te: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    **Returns** the upper and lower parts wrt the leading/trailing edges.

    Note:
        by default, the leading/trailing edges are computed as the left/right-most points
        of the geometry.
    """
    idx_le, idx_te = get_phys_edges_idx(pts) if idx_le < 0 and idx_te < 0 else (idx_le, idx_te)
    start: int = min(idx_le, idx_te)
    end: int = max(idx_le, idx_te)
    if (
        max([p[1] for p in pts[start:end + 1]])
        > max([p[1] for p in np.vstack((pts[:start + 1], pts[end:]))])
    ):
        upper = pts[start:end + 1]
        lower = np.vstack((pts[end:], pts[:start + 1]))
    else:
        lower = pts[start:end + 1]
        upper = np.vstack((pts[:start + 1], pts[end:]))
    return upper, lower


def plot_profile(pts: np.ndarray, cog: np.ndarray = np.array([])):
    """
    **Plots** the complete profile and other optional attributes
    such as the center of gravity and the leading/trailing edges.
    """
    idx_le, idx_te = get_phys_edges_idx(pts)
    idx_le_geom, idx_te_geom = get_geom_edges_idx(pts)
    # Figure
    fsize = (12, 4)
    _, ax = plt.subplots(figsize=fsize)
    ax.plot(pts[:, 0], pts[:, 1], label="baseline profile")
    # physical leading/trailing edges
    ax.scatter(pts[idx_le][0], pts[idx_le][1],
               c="red", s=40, marker="+", zorder=40, label="phys. leading edge")
    ax.scatter(pts[idx_te][0], pts[idx_te][1],
               c="black", s=40, marker="+", zorder=40, label="phys. trailing edge")
    # geometric leading/trailing edges
    ax.scatter(pts[idx_le_geom][0], pts[idx_le_geom][1],
               c="black", s=40, marker="+", zorder=10, label="geo. leading edge")
    ax.scatter(pts[idx_te_geom][0], pts[idx_te_geom][1],
               c="red", s=40, marker="+", zorder=10, label="geo. trailing edge")
    # CoG
    if cog.size > 0:
        ax.scatter(cog[0], cog[1], c="green", s=12, label="CoG")
    # legend and display
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def plot_sides(
        upper: np.ndarray,
        lower: np.ndarray,
        camber: np.ndarray = np.array([]),
        le_circle: np.ndarray = np.array([]),
        te_circle: np.ndarray = np.array([]),
        th_vec: np.ndarray = np.array([])
):
    """
    **Plots** the upper and lower sides of the profile and other optional attributes
    such as the camber line, the leading/trailing edge circles and the maximal thickness.
    """
    # Figure
    fsize = (12, 4)
    _, ax = plt.subplots(figsize=fsize)
    ax.plot(upper[:, 0], upper[:, 1], label="baseline upper side")
    ax.plot(lower[:, 0], lower[:, 1], label="baseline lower side")
    if camber.size > 0:
        ax.plot(camber[:, 0], camber[:, 1], label="camber line")
    if le_circle.size > 0:
        ax.plot(le_circle[:, 0], le_circle[:, 1], label="le circle")
        axins = zoomed_inset_axes(ax, 3.5, loc="upper left")
        axins.plot(upper[:, 0], upper[:, 1])
        axins.plot(lower[:, 0], lower[:, 1])
        axins.plot(camber[:, 0], camber[:, 1])
        axins.plot(le_circle[:, 0], le_circle[:, 1], linestyle="dotted")
        axins.scatter(
            np.sum(le_circle[:, 0]) / len(le_circle),
            np.sum(le_circle[:, 1]) / len(le_circle),
            s=20, marker="+"
        )
        axins.set_xlim(-0.001, 0.003)
        axins.set_ylim(-0.001, 0.003)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if te_circle.size > 0:
        ax.plot(te_circle[:, 0], te_circle[:, 1], label="te circle")
        axins = zoomed_inset_axes(ax, 3.5, loc="lower right")
        axins.plot(upper[:, 0], upper[:, 1])
        axins.plot(lower[:, 0], lower[:, 1])
        axins.plot(camber[:, 0], camber[:, 1])
        axins.plot(te_circle[:, 0], te_circle[:, 1], linestyle="dotted")
        axins.scatter(
            np.sum(te_circle[:, 0]) / len(te_circle),
            np.sum(te_circle[:, 1]) / len(te_circle),
            s=20, marker="+"
        )
        axins.set_xlim(0.065, 0.068)
        axins.set_ylim(0.018, 0.021)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if th_vec.size > 0:
        ax.plot(th_vec[:, 0], th_vec[:, 1], label="th_max")
    # legend and display
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
