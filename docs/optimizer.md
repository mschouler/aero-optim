## Optimizer Module
The optimizer module is designed to define all components required by an evolution algorithm based on [`inspyred`](https://inspyred.readthedocs.io/en/latest/):

- `generator`: a function used to sample an initial population,
- `observer`: a function executed after each evaluation,
- `evaluator`: a function used to extract the quantities of interest (QoIs) of all population candidates,
- some additional optimization arguments.

For each of them, an `Optimizer` attributes or methods are passed to the [`ec.evolve`](https://inspyred.readthedocs.io/en/latest/reference.html#ec-evolutionary-computation-framework) function.

### Optimizer
The [`Optimizer`](https://github.com/mschouler/aero-optim/blob/add-docs/src/ins_optimizer.py#L21-L145) abstract class extracts general arguments from the `"optim"` and `"study"` dictionaries of the configuration file such as:

- `[optim] n_design (int)`: the number of design points i.e. the dimension of the problem,
- `[optim] doe_size (int)`: the doe/population size i.e. the number of individuals per generation,
- `[optim] max_generations (int)`: the maximal number of generations to evaluate,
- `[study] file (str)`: the baseline geometry file,
- `[study] outdir (str)`: the optimization output directory,
- `[study] study_type (str)`: the type of study i.e. the meshing routine to use.

It instantiates optimization related objects:

- `generator (Generator)`: object to sample the initial DOE,
- `ffd (FFD_2D)`: object to deform the baseline geometry,
- `gmsh_mesh (Mesh)`: object to mesh the deformed geometry.

It also implements the following three base methods:

- `process_config`: which goes through the configuration file making sure expected entries are well defined,
- `deform`: which generates the deformed candidate,
- `mesh`: which meshes the deformed candidates.

The `Optimizer` class acts as an `evaluator` and implements the `evaluate` method.

!!! Note
    All optimizer parameters are described in their respective class definition (see [`Optimizer`](https://github.com/mschouler/aero-optim/blob/master/src/ins_optimizer.py#L26-L57), [`WolfSimulator`](https://github.com/mschouler/aero-optim/blob/master/src/ins_optimizer.py#L153-L158)).

### Wolf Optimizer
The [`WolfOptimizer`](https://github.com/mschouler/aero-optim/blob/add-docs/src/ins_optimizer.py#L148-L186) class illustrates how `Optimizer` can be inherited to perform a `Wolf`-based optimization.

It first instantiates a `WolfSimulator` attribute that is then used in the `evaluate` method where for all candidates, the following steps are performed:

1) geometry deformation,

2) deformed geometry meshing,

3) simulation execution

In the end, all simulations results are post-processed to extract the appropriate QoIs returned as a list of floats.