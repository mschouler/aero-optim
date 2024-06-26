import gmsh
import logging
import math
import os

from abc import ABC, abstractmethod
from aero_optim.utils import from_dat, check_dir

logger = logging.getLogger(__name__)


def mesh_format(mesh_file: str, non_corner_tags: list[int]):
    """
    Makes gmsh default .mesh formatting consistent with WOLF's and defines 'Corners'.
    """
    mesh = open(mesh_file, "r").read().splitlines()
    c_vertices: list[int] = []
    logger.info(f"non-corner tags: {non_corner_tags}")

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
    for v_id, id in enumerate(range(vert_idx + 2, vert_idx + 2 + n_vert)):
        line_data = list(map(float, mesh[id].split()))
        if int(line_data[-1]) not in non_corner_tags:
            c_vertices.append(v_id + 1)
        mesh[id] = " " * 4 + f"{line_data[0]:>20}" + \
                   " " * 4 + f"{line_data[1]:>20}" + \
                   " " * 4 + f"{int(line_data[-1]):>20}"

    # append corners
    mesh = mesh[:-1] + ["Corners", str(len(c_vertices))] + [str(v) for v in c_vertices] + ["End"]

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
    """
    This class implements an abstract meshing class.
    """
    def __init__(self, config: dict, datfile: str = ""):
        """
        Instantiates the abstract Mesh object.

        **Input**

        - config (dict): the config file dictionary.
        - dat_file (str): path to input_geometry.dat.

        **Inner**

        - outdir (str): path/to/outputdirectory
        - outfile (str): the core name of all outputed files e.g. outfile.log, outfile.mesh, etc.
        - scale (float): geometry scaling factor.
        - header (int): the number of header lines in dat_file.
        - mesh_format (str): the mesh format (mesh or cgns).
        - bl (bool): whether to mesh the boundary layer (True) or not (False).
        - bl_thickness (float): the BL meshing cumulated thickness.
        - bl_ratio (float): the BL meshing growth ratio.
        - bl_size (float): the BL first element size.
        - structured (bool): whether to recombine triangles (True) or not (False).
        - extrusion_layers (int): the number of extrusion layers when generating a 3D mesh.
        - GUI (bool): whether to launch gmsh GUI (True) or not (False).
        - nview (int): the number of sub-windows in gmsh GUI.
        - quality (bool): whether to display quality metrics in gmsh GUI (True) or not (False).
        - pts (list[list[float]]): the geometry coordinates.
        - surf_tag (list[int]): flow-field elements tags used to recombine the mesh if structured.
        - non_corner_tag (list[int]): non-corner physical entity tags used to define 'Corners'.
        """
        self.config = config
        self.process_config()
        # study params
        self.dat_file: str = datfile if datfile else config["study"]["file"]
        self.outdir: str = config["study"]["outdir"]
        self.outfile = self.config["study"].get("outfile", self.dat_file.split("/")[-1][:-4])
        self.scale: int = config["study"].get("scale", 1)
        self.header: int = config["study"].get("header", 2)
        # mesh params (format & boundary layer)
        self.mesh_format: str = config["gmsh"]["mesh"].get("mesh_format", "mesh").lower()
        self.bl: bool = config["gmsh"]["mesh"].get("bl", False)
        self.bl_thickness: float = config["gmsh"]["mesh"].get("bl_thickness", 1e-3)
        self.bl_ratio: float = config["gmsh"]["mesh"].get("bl_ratio", 1.1)
        self.bl_size: float = config["gmsh"]["mesh"].get("bl_size", 1e-5)
        # mesh params (3d extrusion)
        self.structured: bool = config["gmsh"]["mesh"].get("structured", False)
        self.extrusion_layers: int = config["gmsh"]["mesh"].get("extrusion_layers", 0)
        self.extrusion_size: int = config["gmsh"]["mesh"].get("extrusion_size", 0.001)
        # gui options
        self.GUI: bool = config["gmsh"]["view"].get("GUI", True)
        self.nview: int = config["gmsh"]["view"].get("nview", 1)
        self.quality: bool = config["gmsh"]["view"].get("quality", False)
        # geometry coordinates loading
        self.pts: list[list[float]] = from_dat(self.dat_file, self.header, self.scale)
        # flow-field and non-corner tags (for recombination and corners definition)
        self.surf_tag: list[int] = []
        self.non_corner_tag: list[int] = []

    def get_nlayer(self) -> int:
        """
        **Returns** the number of layers required to reach bl_thickness given the growth bl_ratio
        and the first element size bl_size.
        """
        return math.ceil(
            math.log(1 - self.bl_thickness * (1 - self.bl_ratio) / self.bl_size)
            / math.log(self.bl_ratio) - 1
        )

    def build_mesh(self):
        """
        **Defines** the gmsh routine.
        """
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
        if self.GUI:
            gmsh.fltk.run()

    def write_mesh(self, mesh_dir: str = "", format: bool = True) -> str:
        """
        **Writes** all output files: <file>.geo_unrolled, <file>.log, <file>.mesh and
        returns the mesh filename.

        - mesh_dir: the name of the directory where all gmsh generated files are saved.
        - format: whether to perform medit formatting (True) or not (False) of the mesh.
        - self.outfile: the core name of the outputed files e.g. outfile.log,
           outfile.mesh, etc.
        """
        mesh_dir = self.outdir if not mesh_dir else mesh_dir
        check_dir(mesh_dir)
        # .geo
        logger.info(f"writing {self.outfile}.geo_unrolled to {mesh_dir}")
        gmsh.write(os.path.join(mesh_dir, self.outfile + ".geo_unrolled"))
        # .mesh
        logger.info(f"writing {self.outfile}.{self.mesh_format} to {mesh_dir}")
        gmsh.write(os.path.join(mesh_dir, self.outfile + f".{self.mesh_format}"))
        # medit formatting
        if self.mesh_format == "mesh" and format:
            logger.info(f"medit formatting of {self.outfile}.mesh")
            mesh_format(os.path.join(mesh_dir, self.outfile + ".mesh"), self.non_corner_tag)
        # .log
        log = gmsh.logger.get()
        log_file = open(os.path.join(mesh_dir, self.outfile + ".log"), "w")
        logger.info(f"writing {self.outfile}.log to {mesh_dir}")
        log_file.write("\n".join(log))
        # print summary
        summary = [line for line in log if "nodes" in line and "elements" in line][-1][6:]
        logger.info(f"GMSH summary: {summary}")
        # close gmsh
        gmsh.logger.stop()
        gmsh.finalize()
        return os.path.join(mesh_dir, self.outfile + f".{self.mesh_format}")

    @abstractmethod
    def process_config(self):
        """
        Makes sure the config file contains the required information.
        """

    @abstractmethod
    def build_2dmesh(self):
        """
        Builds the surface mesh of the computational domain.
        """
