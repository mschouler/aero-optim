import logging
import math
import numpy as np
import os

from abc import ABC, abstractmethod
from scipy.stats import qmc
from typing import Any

from aero_optim.utils import from_dat, check_dir

logger = logging.getLogger(__name__)


class Deform(ABC):
    """
    This class implements an abstract Deform class.
    """
    def __init__(self, dat_file: str, ncontrol: int, header: int = 2):
        """
        Instantiates the abstract Deform object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - ncontrol (int): the number of control points.
        - header (int): the number of header lines in dat_file.

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.
        """
        self.dat_file: str = dat_file
        self.pts: np.ndarray = np.array(from_dat(self.dat_file, header))
        self.ncontrol = ncontrol

    def write_ffd(
            self,
            profile: np.ndarray,
            Delta: np.ndarray,
            outdir: str,
            gid: int = 0, cid: int = 0
    ) -> str:
        """
        **Writes** the deformed geometry to file and **returns** /path/to/outdir/outfile.

        - profile (np.ndarray): the deformed geometry coordinates to be written to outfile.
        - Delta (np.ndarray): the deformation vector.
        - outdir (str): the output directory (it is to be combined with outfile).
        """
        outfile = f"{self.dat_file.split('/')[-1][:-4]}_g{gid}_c{cid}.dat"
        check_dir(outdir)
        logger.info(f"write profile g{gid} c{cid} as {outfile} to {outdir}")
        np.savetxt(os.path.join(outdir, outfile), profile,
                   header=f"Deformed profile {outfile}\nDelta={[d for d in Delta]}")
        return os.path.join(outdir, outfile)

    @abstractmethod
    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        """
        Returns a deformed profile.
        """


class FFD_2D(Deform):
    """
    This class implements a simple 2D FFD algorithm with deformation /y only.

    For ncontrol = 2 i.e. 2 control points per side, the unperturbed lattice is:

            P01 ----- P11 ---- P21 ---- P31
              |                         |
              |     ***************     |
              |    **** profile ****    |
              |     ***************     |
              |                         |
            P00 ----- P10 ---- P20 ---- P30

    with (P00, P30, P01, P31) fixed if pad = (1, 1).
    """
    def __init__(
            self, dat_file: str, ncontrol: int, pad: tuple[int, int] = (1, 1), header: int = 2
    ):
        """
        Instantiates the FFD_2D object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - ncontrol (int): the number of control points on each side of the lattice.
        - pad (tuple[int, int]): padding around the displacement vector.
        - header (int): the number of header lines in dat_file.

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.

        - L (int): the number of control points in the x direction of each side of the lattice.
        - M (int): the number of control points in the y direction of each side of the lattice.
        - lat_pts (np.ndarray): the geometry coordinates in the lattice referential.
        """
        super().__init__(dat_file, ncontrol, header)
        assert pad in [(0, 0), (1, 1), (0, 1), (1, 0)], f"wrong padding: {pad}"
        self.pad: tuple[int, int] = pad
        self.L: int = ncontrol - 1 + sum(pad)
        self.M: int = 1
        self.build_lattice()
        self.lat_pts: np.ndarray = self.to_lat(self.pts)

    def build_lattice(self):
        """
        **Builds** a rectangle lattice with x1 as its origin.
        """
        epsilon = 0.
        self.min_x = np.min(self.pts, axis=0)[0] - epsilon
        self.max_x = np.max(self.pts, axis=0)[0] + epsilon
        self.min_y = np.min(self.pts, axis=0)[1] - epsilon
        self.max_y = np.max(self.pts, axis=0)[1] + epsilon
        self.x1 = np.array([self.min_x, self.min_y])

    def to_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        **Returns** the coordinates projected in the lattices referential.

        - pts (np.ndarray): the geometry coordinates in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([(pts[0] - self.min_x) / (self.max_x - self.min_x),
                             (pts[1] - self.min_y) / (self.max_y - self.min_y)])
        return np.column_stack(((pts[:, 0] - self.min_x) / (self.max_x - self.min_x),
                                (pts[:, 1] - self.min_y) / (self.max_y - self.min_y)))

    def from_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        **Returns** lattice coordinates back in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([pts[0] * (self.max_x - self.min_x) + self.min_x,
                             pts[1] * (self.max_y - self.min_y) + self.min_y])
        return np.column_stack((pts[:, 0] * (self.max_x - self.min_x) + self.min_x,
                                pts[:, 1] * (self.max_y - self.min_y) + self.min_y))

    def dPij(self, i: int, j: int, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** y-oriented displacement coordinates dPij from a 1D array Delta.
        """
        return np.array([0., Delta[i + j * (self.L + 1)]])

    def pad_Delta(self, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** padded Delta = [0, dP10, dP20, ..., dP{nc}0, 0, 0, dP11, dP21, ..., dP{nc}1, 0]
        with nc = ncontrol.

        - Delta (np.ndarray): the non-padded deformation vector.
        """
        return np.concatenate((np.pad(Delta[:self.ncontrol], self.pad),
                               np.pad(Delta[self.ncontrol:], self.pad)))

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        """
        **Returns** a new profile resulting from a perturbation Delta in the original referential.

        - Delta (np.ndarray): the deformation vector.</br>
          Delta = [dP10, dP20, ..., dP{nc}0, dP11, dP21, ..., dP{nc}1] with nc = ncontrol.
        """
        assert len(Delta) == 2 * self.ncontrol, f"len(Delta) {len(Delta)} != {2 * self.ncontrol}"
        Delta = self.pad_Delta(Delta)
        new_profile = []
        for x in self.lat_pts:
            x_new = x.copy()
            for ll in range(self.L + 1):
                for m in range(self.M + 1):
                    x_new += (math.comb(self.L, ll) * (1 - x[0])**(self.L - ll)
                              * math.comb(self.M, m) * (1 - x[1])**(self.M - m)
                              * x[0]**ll * x[1]**m * self.dPij(ll, m, Delta))
            new_profile.append([x_new])
        return self.from_lat(np.reshape(new_profile, (-1, 2)))


class FFD_POD_2D(Deform):
    """
    This class implements a 2D FFD-POD coupled class.
    """
    def __init__(
            self,
            dat_file: str,
            pod_ncontrol: int,
            ffd_ncontrol: int,
            ffd_dataset_size: int,
            ffd_bound: tuple[Any],
            seed: int = 123,
            **kwargs
    ):
        """
        Instantiates the FFD_POD_2D object.

        **Input**

        - dat_file (str): path to input_geometry.dat.
        - pod_ncontrol (int): the number of POD control points.
        - ffd_ncontrol (int): the number of FFD control points.
        - ffd_dataset_size (int): the number of ffd profiles in the POD dataset.
        - ffd_bound (tuple[Any]): the ffd dataset deformation boundaries.
        - seed (int): seed for the POD dataset sampling.
        - kwargs (dict): additional options to be passed to the FFD_2D inner object.

        **Inner**

        - pts (np.ndarray): the geometry coordinates in the original referential.

            pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
            where N is the number of points describing the geometry and (z0, ..., zN)
            are null or identical.

        - ffd (FFD_2D): the ffd object used to build the POD dataset.
        """
        super().__init__(dat_file, ffd_ncontrol, **kwargs)
        self.pod_ncontrol = pod_ncontrol
        self.ffd_ncontrol = ffd_ncontrol
        self.ffd_dataset_size = ffd_dataset_size
        self.ffd = FFD_2D(dat_file, ffd_ncontrol // 2, **kwargs)
        self.ffd_bound = ffd_bound
        self.seed = seed
        self.build_pod_dataset()

    def build_pod_dataset(self):
        sampler = qmc.LatinHypercube(d=self.ffd_ncontrol, seed=self.seed)
        sample = sampler.random(n=self.ffd_dataset_size)
        scaled_sample = qmc.scale(sample, *self.ffd_bound)

        profiles = []
        for Delta in scaled_sample:
            profiles.append(self.ffd.apply_ffd(Delta))

        self.S = np.stack([p[:, -1] for p in profiles] , axis=1)
        self.S_mean = 1 / len(profiles) * np.sum(self.S, axis=1)
        self.F = self.S[:, :] - self.S_mean[:, None]
        self.C = np.matmul(np.transpose(self.F), self.F)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.C)
        self.phi = np.matmul(self.F, self.eigenvectors)

        nmode = self.pod_ncontrol
        self.phi_tilde = self.phi[:, -nmode:]
        self.V_tilde_inv = np.linalg.inv(self.eigenvectors)[-nmode:, :]
        self.D_tilde = self.S_mean[:, None] + np.matmul(self.phi_tilde, self.V_tilde_inv)

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (self.ffd.pts[:, 0], self.S_mean + np.sum(self.phi_tilde * Delta, axis=1))
        )

    def get_bound(self) -> tuple[list[float], list[float]]:
        l_bound = [min(v) for v in self.V_tilde_inv]
        u_bound = [max(v) for v in self.V_tilde_inv]
        return l_bound, u_bound
