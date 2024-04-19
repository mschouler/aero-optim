import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys

from scipy.stats import qmc
from src.ffd import FFD_2D
from src.utils import check_file

plt.set_loglevel(level='warning')
logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def plot_profile(ffd: FFD_2D, profiles: list[np.ndarray], delta: np.ndarray, in_lat: bool = False):
    """
    Plots various generated elements in the lattice or original referential.
    """
    # change of referential if necessary
    delta = ffd.pad_Delta(delta)
    pts = ffd.lat_pts if in_lat else ffd.pts
    profiles = [ffd.to_lat(pro) for pro in profiles] if in_lat else profiles
    x1 = ffd.to_lat(ffd.x1) if in_lat else ffd.x1
    width = ffd.max_x - ffd.min_x if not in_lat else 1
    height = ffd.max_y - ffd.min_y if not in_lat else 1
    # Figure
    fsize = (6, 6) if in_lat else (12, 4)
    _, ax = plt.subplots(figsize=fsize)
    ax.plot(pts[:, 0], pts[:, 1], label="naca profile")
    # plot profiles
    for pid, pro in enumerate(profiles):
        ax.plot(pro[:, 0], pro[:, 1], linestyle='dashed', label=f"Pid-{pid}")
    # plot lattice grid
    ax.add_patch(patches.Rectangle((x1[0], x1[1]), width, height, angle=0.0,
                 linewidth=2, edgecolor="r", facecolor="none", label="lattice"))
    # plot deformed points
    if in_lat:
        [ax.scatter(([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                    ([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1], c="k")
            for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
        [ax.annotate(f"$P_{{{i}{j}}} + D_{{{i}{j}}}$",
                     (([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                      ([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1]))
            for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
        ax.set(xlabel="X", ylabel="Y", title="FFD representation in lattice ref.")
    else:
        [ax.scatter(ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                    ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1],
                    c="k") for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
        [ax.annotate(f"$P_{{{i}{j}}} + D_{{{i}{j}}}$",
                     (ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[0],
                      ffd.from_lat([i / ffd.L, j / ffd.M] + ffd.dPij(i, j, delta))[1]))
            for i in range(ffd.L + 1) for j in range(ffd.M + 1)]
        ax.set(xlabel="X", ylabel="Y", title="FFD representation in original ref.")
    # legend and display
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def get_sampler(sampler: str, ncontrol: int, seed: int = 123):
    """
    Builds scipy qmc sampler.
    """
    if sampler not in ["lhs", "sobol", "halton"]:
        raise Exception(f"Unrecognized sampler {sampler}")
    else:
        return (
            qmc.LatinHypercube(d=2 * ncontrol, seed=seed) if sampler == "lhs"
            else qmc.Halton(d=2 * ncontrol, seed=seed) if sampler == "halton"
            else qmc.Sobol(d=2 * ncontrol, seed=seed)
        )


if __name__ == "__main__":
    """
    This program orchestrates one or multiple simple 2D FFD iteration(s).
    Newly generated profiles are also plotted.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-f", "--file", type=str, help="baseline geometry: --datfile=/path/to/file.dat")
    parser.add_argument(
        "-o", "--outdir", type=str, help="output directory", default="output")
    parser.add_argument(
        "-nc", "--ncontrol", type=int, help="number of control points on each side of the lattice",
        default=3)
    parser.add_argument(
        "-np", "--nprofile", type=int, help="number of profiles to generate", default=3)
    parser.add_argument(
        "-r", "--referential", action="store_true", help="plot new profiles in the lattice ref.")
    parser.add_argument(
        "-s", "--sampler", type=str, help="sampling technique [lhs, halton, sobol]", default="lhs")
    parser.add_argument(
        "-d", "--delta", nargs="*", type=float, default=None, help="Delta: D10 D20 .. D2nc (> 0)")
    args = parser.parse_args()

    check_file(args.file)

    # FFD routine
    # (i) Instantiate an FFD object
    #  |  -> create the lattice box
    #  |  -> project the baseline points in the lattice referential
    # (ii) sample a random deformation
    # (iii) Generate a new geometry applying the deformation to the baseline points
    #  |  -> compute the displacements from the deformation and the
    #  |     tensor product of the Bernstein basis polynomials
    #  |  -> project the new profile back into the original referential
    seed = 1234
    ncontrol = args.ncontrol
    ffd = FFD_2D(args.file, ncontrol)
    if not args.delta:
        sampler = get_sampler(args.sampler, ncontrol, seed)
        sample = sampler.random(n=args.nprofile)
        scaled_sample = qmc.scale(sample, -0.5, 0.5)
    else:
        scaled_sample = [np.array(args.delta)]
    profiles = []
    for Delta in scaled_sample:
        profiles.append(ffd.apply_ffd(Delta))
    plot_profile(ffd, profiles, Delta, args.referential)
    for pid, profile in enumerate(profiles):
        file_name = ffd.write_ffd(profile, scaled_sample[pid], "output")
