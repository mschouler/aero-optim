import gmsh
import logging
import math
import os
import numpy as np
import pandas
import re

from abc import ABC, abstractmethod
from aero_optim.utils import from_dat, check_dir, read_next_line_in_file

logger = logging.getLogger(__name__)


def get_mesh_kwd(mesh: list[str], kwd: str) -> int:
    try:
        idx = next(idx for idx, el in enumerate(mesh) if kwd in el)
    except StopIteration:
        raise Exception(f"ERROR -- no '{kwd}' entry in mesh file")
    return idx


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
        - bl_fan_elements (int): the number of BL fan elements.
        - mesh_order (int): the order of the mesh.
        - structured (bool): whether to recombine triangles (True) or not (False).
        - extrusion_layers (int): the number of extrusion layers when generating a 3D mesh.
        - extrusion_size (float): the total size of the extruded layers.
        - GUI (bool): whether to launch gmsh GUI (True) or not (False).
        - nview (int): the number of sub-windows in gmsh GUI.
        - quality (bool): whether to display quality metrics in gmsh GUI (True) or not (False).
        - pts (list[list[float]]): the geometry coordinates.
        - surf_tag (list[int]): flow-field elements tags used to recombine the mesh if structured.
        - non_corner_tags (list[int]): non-corner physical entity tags used to define 'Corners'.
        - lower_tag (list[int]): lower periodic tags to be identified as one.
        - lower_tag (list[int]): upper periodic tags to be identified as one.
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
        self.bl_fan_elements: int = config["gmsh"]["mesh"].get("bl_fan_elements", 10)
        self.mesh_order: int = config["gmsh"]["mesh"].get("order", 0)
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
        self.non_corner_tags: list[int] = []
        self.bottom_tags: list[int] = []
        self.top_tags: list[int] = []

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
        self.build_3dmesh() if self.extrusion_layers > 0 else 0

        if self.structured:
            [gmsh.model.geo.mesh.setRecombine(2, abs(id)) for id in self.surf_tag]
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3) if self.extrusion_layers > 0 else gmsh.model.mesh.generate(2)
        if self.mesh_order:
            gmsh.model.mesh.setOrder(self.mesh_order)

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

    def write_mesh(self, mesh_dir: str = "") -> str:
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
        if self.mesh_format == "mesh":
            logger.info(f"medit formatting of {self.outfile}.mesh")
            mesh_file = os.path.join(mesh_dir, self.outfile + ".mesh")
            mesh = open(mesh_file, "r").read().splitlines()
            if self.extrusion_layers == 0:
                mesh = self.reformat_2d(mesh)
            if self.non_corner_tags:
                mesh = self.add_corners(mesh)
            if self.bottom_tags and self.top_tags:
                mesh = self.merge_refs(mesh)
            with open(mesh_file, 'w') as ftw:
                ftw.write("\n".join(mesh))
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
        return self.get_meshfile(mesh_dir)

    def reformat_2d(self, mesh: list[str]) -> list[str]:
        """
        **Fix** gmsh default .mesh format in 2D.
        """
        idx = get_mesh_kwd(mesh, "Dimension")
        mesh[idx] = " Dimension 2"
        del mesh[idx + 1]

        vert_idx = get_mesh_kwd(mesh, "Vertices")
        n_vert = int(mesh[vert_idx + 1])
        for id in range(vert_idx + 2, vert_idx + 2 + n_vert):
            line_data = list(map(float, mesh[id].split()))
            mesh[id] = " " * 4 + f"{line_data[0]:>20}" + \
                       " " * 4 + f"{line_data[1]:>20}" + \
                       " " * 4 + f"{int(line_data[-1]):>20}"
        return mesh

    def add_corners(self, mesh: list[str]) -> list[str]:
        """
        **Adds** Corners at the end of the mesh file.
        """
        c_vert: list[int] = []
        logger.debug(f"non-corner tags: {self.non_corner_tags}")

        vert_idx = get_mesh_kwd(mesh, "Vertices")
        n_vert = int(mesh[vert_idx + 1])
        for v_id, id in enumerate(range(vert_idx + 2, vert_idx + 2 + n_vert)):
            line_data = list(map(float, mesh[id].split()))
            if int(line_data[-1]) not in self.non_corner_tags:
                c_vert.append(v_id + 1)

        mesh = mesh[:-1] + ["Corners", str(len(c_vert))] + [str(v) for v in c_vert] + ["End"]
        return mesh

    def merge_refs(self, mesh: list[str]) -> list[str]:
        """
        **Merges** the periodic boundaries references on each side of the domain.
        """
        logger.debug(f"top tags: {self.top_tags} merged in ref: {max(self.top_tags)}")
        logger.debug(f"bottom tags: {self.bottom_tags} merged in ref: {min(self.bottom_tags)}")

        vert_idx = get_mesh_kwd(mesh, "Vertices")
        n_vert = int(mesh[vert_idx + 1])
        for id in range(vert_idx + 2, vert_idx + 2 + n_vert):
            line_data = list(map(float, mesh[id].split()))
            if int(line_data[-1]) in self.bottom_tags:
                line_data[-1] = min(self.bottom_tags)
            elif int(line_data[-1]) in self.top_tags:
                line_data[-1] = max(self.top_tags)
            mesh[id] = " " * 4 + f"{line_data[0]:>20}" + \
                       " " * 4 + f"{line_data[1]:>20}" + \
                       " " * 4 + f"{int(line_data[-1]):>20}"

        edge_idx = get_mesh_kwd(mesh, "Edges")
        n_edges = int(mesh[edge_idx + 1])
        for id in range(edge_idx + 2, edge_idx + 2 + n_edges):
            line_data = list(map(int, mesh[id].split()))
            if line_data[2] in self.bottom_tags:
                line_data[2] = min(self.bottom_tags)
            elif line_data[2] in self.top_tags:
                line_data[2] = max(self.top_tags)
            mesh[id] = " " + f"{line_data[0]}" + " " + f"{line_data[1]}" + " " + f"{line_data[2]}"
        return mesh

    def get_meshfile(self, mesh_dir: str) -> str:
        """
        **Returns** the path to the generated mesh.
        """
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

    def build_3dmesh(self):
        """
        Builds a 3D mesh by extrusion
        """
        raise Exception("build_3dmesh method not implemented")


class MeshMusicaa(ABC):
    """
    This class implements an abstract meshing class for the solver MUSICAA.
    """
    def __init__(self, config: dict, datfile: str = ""):
        """
        Instantiates the abstract MeshMusicaa object.

        **Input**

        - config (dict): the config file dictionary.

        **Inner**

        - dat_file (str): input_geometry.dat file including path.
        - dat_dir (str): path to input_geometry.dat.
        - outdir (str): path/to/outputdirectory
        - outfile (str): the core name of all outputed files e.g. outfile.log, outfile.mesh, etc.
        - header (int): the number of header lines in dat_file.
        - mesh_name (str): name of the mesh files (******_bl1.x)
        - wall_bl (list[int]): list containing the blocks adjacent to the geometry to be optimized.
                   The block numbers should be ordered following the curvilinear abscissae of
                   the blade. Unfortunately, the present version only reads walls along i
                   located at j=0 (grid indices).
        - pitch (float): blade-to-blade pitch
        - periodic_bl (list[int]): list containing the blocks that must be translated DOWN to reform
                        the blade geometry fully. Not needed if the original mesh already
                        surrounds the blade

        """
        self.cwd: str = os.getcwd()
        self.config = config
        # study params
        self.dat_file: str = datfile if datfile else config["study"]["file"]
        self.dat_dir = '/'.join(self.dat_file.split('/')[:-1])
        self.outdir: str = config["study"]["outdir"]
        self.outfile = self.config["study"].get("outfile", self.dat_file.split("/")[-1][:-4])
        self.header: int = config["study"].get("header", 2)
        # mesh params
        self.wall_bl: list[int] = config["plot3D"]["mesh"].get("wall_bl", [0])
        self.block_info = self.get_block_info()
        self.pitch: int = config["plot3D"]["mesh"].get('pitch', 1)
        self.periodic_bl: list[int] = config["plot3D"]["mesh"].get("periodic_bl", [0])
        self.mesh_name: str = config["plot3D"]["mesh"].get("mesh_name", self.outfile)

    def write_mesh(self, outdir: str) -> str:
        """
        **Returns** path/to/MESH/musicaa_<outfile>/ .
        """
        return os.path.join(outdir, "MESH", f'musicaa_{self.outfile}')

    def get_block_info(self) -> dict:
        """
        **Returns** a dictionnary containing relevant information on the blocks
        used by MUSICAA: number of blocks, block size.
        """
        block_info = {}

        # get number of blocks
        nbl_ = read_next_line_in_file(os.path.join(self.dat_dir, "param_blocks.ini"),
                                      "Number of Blocks")
        nbl = re.findall(r"\d+", nbl_)[0]

        # iterate for each block
        for bl in range(nbl):
            bl += 1
            pattern = f"! Block #{bl}"
            with open("param_blocks.ini", "r") as f:
                filedata = f.readlines()
            for i, line in enumerate(filedata):
                if pattern in line:
                    block_info[f"nx_bl{bl}"] = re.findall(r"\d+", filedata[i + 3])[0]
                    block_info[f"ny_bl{bl}"] = re.findall(r"\d+", filedata[i + 4])[0]
                    block_info[f"nz_bl{bl}"] = re.findall(r"\d+", filedata[i + 5])[0]

        return block_info

    def read_bl(self, bl: int) -> np.ndarray:
        """
        **Returns** an array containing the block coordinates.
        """
        # read block coordinates
        with open(f'{self.dat_dir}/{self.outfile}_bl{bl}.x', 'r') as f:
            a = pandas.read_csv(f).to_numpy()

        # get block size
        nx = self.block_info[f"nx_bl{bl}"]
        ny = self.block_info[f"ny_bl{bl}"]

        # convert coordinates to a regular 2D array
        coord_list = []
        for line in a[1:]:
            b = line[0].split(' ')
            for element in b:
                if element != '':
                    coord_list.append(np.float64(element))

        # extract 2D mesh and store
        coords_x = np.array(coord_list[:nx * ny])
        coords_y = np.array(coord_list[nx * ny:2 * nx * ny])
        coords = np.vstack((coords_x, coords_y)).T

        return coords

    def write_deformed_mesh_edges(
            self,
            profile: np.ndarray,
            outdir: str,
            gid: int = 0, cid: int = 0
    ) -> str:
        """
        **Writes** the deformed profile in a format such that the MUSICAA solver
        can generate the fully deformed mesh via a Fortran routine.
        **Returns** the path to the files.
        """
        # create output directory within MESH
        check_dir(os.path.join(outdir, "MESH"))
        mesh_dir = os.path.join(outdir, "MESH", f'musicaa_{self.outfile}')
        check_dir(mesh_dir)

        # loop over blocks
        j = 0
        for bl in self.wall_bl:

            # get block dimensions
            nx = self.block_info[f'nx_bl{bl}']
            ny = self.block_info[f'ny_bl{bl}']
            nz = 1

            # open file and write specific format
            mesh_file = f"{mesh_dir}/{self.mesh_name}_g{gid}_c{cid}_pert_edges_bl{bl}.x"
            with open(mesh_file, "w") as f:
                f.write('1\n')
                f.write(str(str(nx) + '  ' + str(ny) + '  ' + str(nz) + '\n'))

                # write wall coordinates
                for i in range(nx):
                    f.write(str(profile[i + j][0]) + ' ')
                f.write('\n')
                for i in range(nx):
                    if bl in self.periodic_bl:
                        f.write(str(profile[i + j][1] + self.pitch) + ' ')
                    else:
                        f.write(str(profile[i + j][1]) + ' ')
                j += nx

        return mesh_dir

    def write_profile_from_mesh(self):
        """
        **Writes** the profile by extracting its coordinates from MUSICAA grid files.
        """
        # create storage
        coords_wall_x = []
        coords_wall_y = []

        # loop over blocks within list
        for bl in self.wall_bl:

            # read block coordinates
            coords = self.read_bl(bl)
            nx = self.block_info[f'nx_bl{bl}']

            # save only wall (located at j=0)
            coords_wall_x.append(coords[:nx, 0])
            coords_wall_y.append(coords[:nx, 1])

            # check if block must be translated in the pitchwise direction
            if bl in self.periodic_bl:
                coords_wall_y[-1] += -self.pitch

        # assemble 2D array
        coords_wall = np.vstack((np.hstack(coords_wall_x), np.hstack(coords_wall_y))).T

        # save to file
        np.savetxt(self.dat_file, coords_wall,
                   header=(f"Baseline profile {self.outfile}\n"
                           f"Extracted from mesh files in {self.dat_dir}"))

    @abstractmethod
    def deform_mesh(self, outdir: str):
        """
        **Executes** the MUSICAA mesh deformation routine.
        """
        pass

    @abstractmethod
    def build_mesh(self):
        """
        **Orchestrates** the required steps to deform the baseline mesh using the new
        deformed profile for MUSICAA.
        """
        pass
