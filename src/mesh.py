import gmsh
import math
import os

from abc import ABC, abstractmethod
from .utils import from_dat, check_dir


def mesh_format2d(mesh_file: str):
    """
    Make gmsh default .mesh formatting consistent with WOLF's.
    """
    mesh = open(mesh_file, "r").read().splitlines()

    # fix dimension if 2d
    try:
        idx = next(idx for idx, el in enumerate(mesh) if "Dimension" in el)
    except StopIteration:
        raise Exception("ERROR -- no 'Dimension' entry in mesh file")
    mesh[idx] = " Dimension 2"
    del mesh[idx + 1]

    # remove third coordinate
    try:
        vert_idx = next(idx for idx, el in enumerate(mesh) if "Vertices" in el)
        n_vert = int(mesh[vert_idx + 1])
    except StopIteration:
        raise Exception("ERROR -- no 'Vertices' entry in mesh file")
    for id in range(vert_idx + 2, vert_idx + 2 + n_vert):
        mesh[id] = mesh[id][:-5]

    # overwrite mesh file
    with open(mesh_file, 'w') as ftw:
        ftw.write("\n".join(mesh))


def split_view(nview: int):
    gmsh.fltk.initialize()
    if nview == 2:
        gmsh.fltk.splitCurrentWindow('h', 0.5)
    if nview == 3:
        gmsh.fltk.splitCurrentWindow('v', 0.5)
    if nview == 4:
        gmsh.fltk.setCurrentWindow(0)
        gmsh.fltk.splitCurrentWindow('v', 0.5)


def set_display(color: list[tuple], number: list[tuple]):
    for args in color:
        gmsh.option.setColor(*args)
    for args in number:
        gmsh.option.setNumber(*args)


def plot_quality():
    gmsh.plugin.setNumber("AnalyseMeshQuality", "ICNMeasure", 1.)
    gmsh.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1.)
    t = gmsh.plugin.run("AnalyseMeshQuality")
    dataType, tags, data, time, numComp = gmsh.view.getModelData(t, 0)


class Mesh(ABC):
    def __init__(self, config: dict, datfile: str = ""):
        self.config = config
        self.process_config()
        # study params
        self.dat_file: str = config["study"]["file"] if not datfile else datfile
        self.outdir: str = config["study"]["outdir"]
        self.outfile = self.config["study"].get("outfile", self.dat_file.split("/")[-1][:-4])
        self.scale: int = config["study"].get("scale", 1)
        self.header: int = config["study"].get("header", 2)
        # mesh params (boundary layer)
        self.bl: bool = config["gmsh"]["mesh"].get("bl", False)
        self.bl_thickness: float = config["gmsh"]["mesh"].get("bl_thickness", 1e-3)
        self.bl_ratio: float = config["gmsh"]["mesh"].get("bl_ratio", 1.1)
        self.bl_size: float = config["gmsh"]["mesh"].get("bl_size", 1e-5)
        # mesh params (3d extrusion)
        self.elt_size: float = config["gmsh"]["mesh"].get("elt_size", 5e-2)
        self.structured: bool = config["gmsh"]["mesh"].get("structured", False)
        self.extrusion_layers: int = config["gmsh"]["mesh"].get("extrusion_layers", 0)
        self.extrusion_size: int = config["gmsh"]["mesh"].get("extrusion_size", 0.001)
        # gui options
        self.GUI: bool = config["gmsh"]["view"].get("GUI", True)
        self.nview: int = config["gmsh"]["view"].get("nview", 1)
        self.quality: bool = config["gmsh"]["view"].get("quality", False)
        # geometry coordinates loading
        self.pts: list[list[float]] = from_dat(self.dat_file, self.header, self.scale)

    def get_nlayer(self) -> int:
        """
        Return the number of layers required to reach bl_thickness given the growth bl_ratio
        and the first element size bl_size.
        """
        return math.ceil(
            math.log(1 - self.bl_thickness * (1 - self.bl_ratio) / self.bl_size)
            / math.log(self.bl_ratio) - 1
        )

    def write_outputs(self):
        """
        Write output files: <file>.geo_unrolled, <file>.log, <file>.mesh.
        """
        check_dir(self.outdir)
        print(f">> writing {self.outfile}.geo_unrolled to {self.outdir}")
        gmsh.write(os.path.join(self.outdir, self.outfile + ".geo_unrolled"))
        print(f">> writing {self.outfile}.mesh to {self.outdir}")
        gmsh.write(os.path.join(self.outdir, self.outfile + ".mesh"))
        if self.extrusion_layers == 0:
            print(f">> 2d formatting of {self.outfile}.mesh")
            mesh_format2d(os.path.join(self.outdir, self.outfile + ".mesh"))
        log = gmsh.logger.get()
        log_file = open(os.path.join(self.outdir, self.outfile + ".log"), "w")
        print(f">> writing {self.outfile}.log to {self.outdir}")
        log_file.write("\n".join(log))
        summary = [line for line in log if "nodes" in line and "elements" in line][-1][6:]
        print(f">> GMSH summary: {summary}")
        gmsh.logger.stop()

    @abstractmethod
    def process_config(self):
        """
        Make sure the config file contains the required information and extract it.
        """

    @abstractmethod
    def build_mesh(self):
        """
        Define the gmsh routine.
        """
