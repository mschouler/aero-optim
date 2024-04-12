import math
import numpy as np
import os

from .utils import from_dat, check_dir


class FFD_2D:
    """
    This class implements a simple 2D FFD algorithm with deformation /y only.
    >> For ncontrol = 2 i.e. 2 control points per side, the unperturbed lattice is:

            P01 ----- P11 ---- P21 ---- P31
              |                         |
              |     ***************     |
              |    **** profile ****    |
              |     ***************     |
              |                         |
            P00 ----- P10 ---- P20 ---- P30

       with (P00, P30, P01, P31) fixed.
    """
    def __init__(self, file: str, ncontrol: int, header_len: int = 2):
        """
        Instantiates the FFD_2D object.

        Input
            >> file: path to input_geometry.dat.
            >> ncontrol: the number of control points on each side of the lattice.
            >> header_len: the number of header lines in file.
        Inner
            >> pts: the geometry coordinates in the original referential.
               pts = [[x0, y0, z0], [x1, y1, z1], ..., [xN, yN, zN]]
               where N is the number of points describing the geometry
               and (z0, ..., zN) are null or identical.
            >> L: the number of control points in the x direction of each side of the lattice.
            >> M: the number of control points in the y direction of each side of the lattice.
            >> lat_pts: the geometry coordinates in the lattice referential.
        """
        self.dat_file: str = file
        self.pts: np.ndarray = np.array(from_dat(self.dat_file, header_len))
        self.ncontrol = ncontrol
        self.L: int = ncontrol + 1
        self.M: int = 1
        self.build_lattice()
        self.lat_pts: np.ndarray = self.to_lat(self.pts)

    def build_lattice(self):
        """
        Builds a rectangle lattice with x1 as its origin.
        """
        epsilon = 0.
        self.min_x = np.min(self.pts, axis=0)[0] - epsilon
        self.max_x = np.max(self.pts, axis=0)[0] + epsilon
        self.min_y = np.min(self.pts, axis=0)[1] - epsilon
        self.max_y = np.max(self.pts, axis=0)[1] + epsilon
        self.x1 = np.array([self.min_x, self.min_y])

    def to_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        Returns the coordinates projected in the lattices referential
        >> pts: the geometry coordinates in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([(pts[0] - self.min_x) / (self.max_x - self.min_x),
                             (pts[1] - self.min_y) / (self.max_y - self.min_y)])
        return np.column_stack(((pts[:, 0] - self.min_x) / (self.max_x - self.min_x),
                                (pts[:, 1] - self.min_y) / (self.max_y - self.min_y)))

    def from_lat(self, pts: np.ndarray) -> np.ndarray:
        """
        Projects lattice coordinates back in the original referential.
        """
        if len(pts.shape) == 1:
            return np.array([pts[0] * (self.max_x - self.min_x) + self.min_x,
                             pts[1] * (self.max_y - self.min_y) + self.min_y])
        return np.column_stack((pts[:, 0] * (self.max_x - self.min_x) + self.min_x,
                                pts[:, 1] * (self.max_y - self.min_y) + self.min_y))

    def dPij(self, i: int, j: int, Delta: np.ndarray) -> np.ndarray:
        """
        Returns y-oriented displacement coordinates dPij from a 1D array Delta.
        """
        return np.array([0., Delta[i + j * (self.L + 1)]])

    def pad_Delta(self, Delta: np.ndarray) -> np.ndarray:
        """
        Returns padded Delta = [0, dP10, dP20, ..., dP{nc}0, 0, 0, dP11, dP21, ..., dP{nc}1, 0]
        with nc = ncontrol.
        >> Delta: the non-padded deformation vector.
        """
        return np.concatenate((np.pad(Delta[:self.ncontrol], (1, 1)),
                               np.pad(Delta[self.ncontrol:], (1, 1))))

    def apply_ffd(self, Delta: np.ndarray) -> np.ndarray:
        """
        Generates and returns a new naca profile resulting from a perturbation Delta
        in the original referential.
        >> Delta the deformation vector
           i.e. Delta = [dP10, dP20, ..., dP{nc}0, dP11, dP21, ..., dP{nc}1]
           with nc = ncontrol.
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

    def write_ffd(self, profile: np.ndarray, Delta: np.ndarray, outdir: str,
                  gid: int = 0, cid: int = 0) -> str:
        """
        Writes the deformed geometry to file and returns /path/to/outdir/outfile/outfile.
        >> profile: the deformed geometry coordinates to be written to outfile.
        >> outdir: the output directory (it is to be combined with outfile).
        >> outfile: the name of the outputed geometry (<geom>.dat).
        """
        outfile = f"{self.dat_file.split('/')[-1][:-4]}_g{gid}_c{cid}.dat"
        check_dir(outdir)
        print(f">> write profile g{gid} c{cid} as {outfile} to {outdir}")
        np.savetxt(os.path.join(outdir, outfile), profile,
                   header=f"Deformed profile {outfile}\nDelta={[d for d in Delta]}")
        return os.path.join(outdir, outfile)
