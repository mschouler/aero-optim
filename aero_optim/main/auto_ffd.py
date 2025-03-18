import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

from scipy.stats import qmc
from aero_optim.ffd.ffd import FFD_2D
from aero_optim.utils import check_file

# set pillow and matplotlib loggers to WARNING mode
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.set_loglevel(level='warning')

# get framework logger
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def plot_profile(ffd: FFD_2D, profiles: list[np.ndarray], Delta: np.ndarray,
                 outdir: str, in_lat: bool = False):
    """
    Plots various generated elements in the lattice or original referential.
    """
    # change of referential if necessary
    pts = ffd.lat_pts if in_lat else ffd.pts
    profiles = [ffd.to_lat(pro) for pro in profiles] if in_lat else profiles
    x1 = ffd.to_lat(ffd.x1) if in_lat else ffd.x1
    width = ffd.max_x - ffd.min_x if not in_lat else 1
    height = ffd.max_y - ffd.min_y if not in_lat else 1
    # Figure
    fsize = (6, 6) if in_lat else (12, 4)
    _, ax = plt.subplots(figsize=fsize)
    ax.plot(pts[:, 0], pts[:, 1], c="k", label="baseline profile")
    # colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # plot profiles
    for pid, pro in enumerate(profiles):
        ax.plot(pro[:, 0], pro[:, 1], linestyle='dashed', c=colors[pid], label=f"Pid-{pid}")
    # plot lattice grid
    ax.add_patch(patches.Rectangle((x1[0], x1[1]), width, height, angle=0.0,
                 linewidth=2, edgecolor="r", facecolor="none", label="lattice"))
    # plot deformed points
    for did, delta in enumerate(Delta):
        delta = ffd.pad_Delta(delta)
        if in_lat:
            [ax.scatter(([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                        ([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1], c=colors[did])
                for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
            [ax.annotate(f"$P_{{{i}{j}}} + D_{{{i}{j}}}$",
                         (([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                         ([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1]))
                for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
            # black edges
            [ax.scatter(*np.array([i, j]), c="k") for i in range(2) for j in range(2)]
            ax.set(xlabel="$x$ [-]", ylabel="$y$ [-]", title="FFD illustrated in the lattice ref.")
        else:
            [ax.scatter(ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                        ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1],
                        c=colors[did]) for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
            [ax.annotate(f"$P_{{{i}{j}}} + D_{{{i}{j}}}$",
                         (ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                         ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1]))
                for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
            # black edges
            ax.set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="FFD illustrated in the original ref.")
            [ax.scatter(*ffd.from_lat(np.array([i, j])), c="k") for i in range(2) for j in range(2)]
    # legend and display
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ffd.png"))
    plt.show()


def main():
    """
    This program orchestrates one or multiple simple 2D FFD iteration(s).
    Newly generated profiles are also plotted.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f", "--file", type=str, required=True, help="baseline geometry: --file=/path/to/file.dat")
    parser.add_argument(
        "-c", "--config", type=str, default="", help="config: --config=/path/to/config.json")
    parser.add_argument(
        "-o", "--outdir", type=str, default="output", help="output directory")
    parser.add_argument(
        "-nc", "--ncontrol", type=int, default=3,
        help="number of control points on each side of the lattice")
    parser.add_argument(
        "-np", "--nprofile", type=int, default=3, help="number of profiles to generate")
    parser.add_argument(
        "-d", "--delta", type=str, default=None, help="Delta: 'D10 D20 .. D2nc'")
    args = parser.parse_args()

    # check input arguments
    check_file(args.file)

    # build ffd config with header and padding options
    ffd_config = {} if not args.config else args.config.get("ffd", {})
    # Note: the config can also be used to pass arguments used in this script
    # delta (list[float]): a deformation list [D10 D20 .. D2nc] (default=None)
    delta = ffd_config.get("delta", None) if not args.delta else args.delta
    # nprofile (int): number of profiles to generate (default=3, ignored if a delta is given)
    nprofile = ffd_config.get("nprofile", args.nprofile)
    # referential (bool): to plot the profiles in the lattice referential (default=False)
    referential = ffd_config.get("referential", False)

    # FFD object and displacements
    ffd = FFD_2D(args.file, args.ncontrol, **ffd_config)
    if not delta:
        sampler = qmc.LatinHypercube(d=args.ncontrol * 2, seed=1234)
        sample = sampler.random(n=nprofile)
        scaled_sample = qmc.scale(sample, -0.5, 0.5)
    else:
        scaled_sample = [np.array([float(d) for d in args.delta.split()])]
    # FFD profiles
    profiles = []
    for Delta in scaled_sample:
        profiles.append(ffd.apply_ffd(Delta))
    for pid, profile in enumerate(profiles):
        _ = ffd.write_ffd(profile, scaled_sample[pid], args.outdir, cid=pid)
    # FFD figure
    plot_profile(ffd, profiles, scaled_sample, args.outdir, referential)


if __name__ == "__main__":
    main()
