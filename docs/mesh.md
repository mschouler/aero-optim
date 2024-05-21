## Mesh Module
The meshing module builds on the [`gmsh` Python API](https://gmsh.info/doc/texinfo/gmsh.html). Classes inheriting from a basic `Mesh` class are implemented for NACA airfoil use-cases:

* `NACABaseMesh`: implements a simple meshing routine for a naca12 profile with minimal parameters,
* `NACABlockMesh`:  implements a structured by blocks meshing routine for a naca12 profile,
* `CascadeMesh`: implements an unstructured meshing routine for a compressor cascade with periodic elements.

All three meshes are parameterized with a `json` formatted configuration file made of several dictionaries:
```json
{
  "study": {
        // study entries parameterizing the study and input/output options
    },
    "gmsh": {
        // mesh related entries parameterizing gmsh api
        "domain": {
            // domain entries parameterizing the computational domain size
        },
        "mesh": {
            // meshing entries parameterizing the boundary layer,
            // the domain boundaries and the extrusion if defined
        },
        "view": {
            // visualization entries parameterizing the GUI display
        }
    }
}
```

The meshing routine then goes through the following steps:

1) a `Mesh` object is instantiated from the configuration dictionary,

2) the `build_mesh()` method is called on the instantiated object which then triggers subsequent calls:

  * `build_2dmesh()` that builds the computational domain and defines 2D meshing parameters (e.g. number of nodes, growth ratio)
  * `split_naca()` or `reorder_blade()` that pre-processes the geometry coordinates list

3)  the mesh is finally generated, GUI options are set and outputs (e.g. meshing log, output mesh) are written.

!!! Note
    All meshing parameters are described in their respective class definition (see [Developer Guide](dev_mesh.md)).

### NACA Base
Meshing details relative to this routine reside in the `build_2dmesh()` method and its inner calls. For instance, the `split_naca()` method describes how the naca profile should be split into its upper and lower parts. This is critical to the domain construction steps in `build_2dmesh()` since the trailing and leading edges may be used as construction points. 

The `build_2dmesh()` routine of the `NACABaseMesh` class also gives the possibility to mesh the boundary layer by calling `build_bl()`. The meshing of the boundary layer is triggered by setting `"bl"`to `true` in the `"mesh"` category of the configuration file.

For this class, the computational domain is a rectangle whose inlet face (on the left) is made of a semi-circle. The domain dimensions are parameterized in the `"domain"` section of the configuration file:

- `inlet (int)`: the inlet face radius centered on the airfoil trailing edge,
- `outlet (int)`: the outlet distance to the airfoil trailing,
- `le_offset (int)`: the size (in point number) of the leading edge portion that is meshed with its own refinement level.

The `"mesh"` entry contains various meshing parameters such as the number of nodes on the domain inner and outer boundaries or the parameters of the boundary layer if needed:

- `nodes_inlet (int)`: the number of nodes to mesh the inlet boundary,
- `nodes_outlet (int)`: the number of nodes to mesh the outlet boundary,
- `side_nodes (int)`: the number of nodes to mesh the upper and lower side boundaries,
- `le (int)`: the number of nodes to mesh the leading edge portion defined earlier,
- `low (int)`: the number of nodes to mesh the trailing lower portion of the airfoil,
- `up (int)`: the number of nodes to mesh the trailing upper portion of the airfoil.

For this meshing routine, other `"mesh"` parameters can be used to parameterize the meshing of the boundary layer (BL):

- `bl (bool)`: whether to mesh the boundary layer (True) or not (False).
- `bl_thickness (float)`: the BL meshing cumulated thickness.
- `bl_ratio (float)`: the BL meshing growth ratio.
- `bl_size (float)`: the BL first element size.

Finally, the `"view"` entry contains GUI options to turn it on or off, to display quality metrics and to split the view.

### NACA Block
This meshing routine also inherits from `Mesh`. Particularities of this routine mostly lies in the `build_2dmesh()` method. Hence, the boundary layer cannot be meshed with `build_bl()` which is not called anymore. In addition, the domain is this time made of several inner blocks.

Hence for this class, the computational domain still has the same general structure (a rectangle with a semi-circular inlet) but inner blocks are defined and parameterized in `"domain"`:

- `inlet (int)`: the inlet face radius centered on the airfoil trailing edge,
- `outlet (int)`: the outlet distance to the airfoil trailing,
- `le_offset (int)`: the size (in point number) of the leading edge portion that is meshed with its own refinement level,
- `block_width (float)`: the size of the trailing blocks that encompass the remaining of the airfoil.

The `"mesh"` entry contains various meshing parameters such as the number of nodes on the domain inner/outer boundaries and blocks:

- `n_inlet (int)`: the number of nodes to mesh the inlet and the leading edge,
- `n_vertical (int)`: number of nodes to mesh the outlet and the blocks in the vertical direction,
- `r_vertical (int)`: the outlet and vertical direction growth ratio,
- `n_airfoil (int)`: the number of nodes to mesh both sides of the trailing portion of the airfoil,
- `r_airfoil (int)`: the airfoil sides growth ratio,
- `n_wake (int)`: the number of nodes in the wake direction,
- `r_wake (int)`: the wake growth ratio.

### Cascade
This meshing routine corresponds to an internal flow simulation. For this reason, the `build_bl()` method contains several specificities such as the definition of periodic elements on the upper and lower sides of the domain.

In addition, multiple gmsh "size fields" are defined to mesh the boundary layer and locally refine the mesh in the blade's wake.

!!! Note
    This meshing routine was adapted from an existing gmsh file. For this reason, many parameters are hard coded inside the class definition (e.g. the spline defining the blade and their number of nodes).

### Illustration
Examples of unstructured meshes obtained with the naca routines are given below:
<p float="left">
  <img src="../Figures/naca_base_mesh.png" width="49%" />
  <img src="../Figures/naca_block_mesh.png" width="49%" /> 
</p>

The compressor cascade mesh obtained with the cascade routine is illustrated next:
<p float="left">
  <img src="../Figures/cascade_mesh.png" width="100%" />
</p>

### Quick Experiments
The `auto_gmsh.py` scripts enables basic testing and visualization for a given configuration file.

For instance setting `"structured"` to `true` in `naca_block_mesh.json` will produce a fully structured mesh:
```sh
# from aero-optim to naca_block
cd examples/NACA12/naca_block
mesh --config=naca_block.json
```
<p float="left">
  <img src="../Figures/naca_block_mesh_structured.png" width="100%" />
</p>

It is also possible to supersede the config `"file"` entry with the `--file` input argument. Hence, any previously generated deformed geometry can be meshed according to the naca routine with the commands below:
```sh
# from naca_block to naca_base
cd ../naca_base
ffd -f ../data/naca12.dat -nc 2 -d "0. 0. 1. 1."
mesh --config=naca_base.json --file=output/naca12_g0_c0.dat
```
<p float="left">
  <img src="../Figures/naca_base_mesh_ffd.png" width="100%" />
</p>
