## Optimizer Module
The optimizer module is designed to define all components required by an evolution algorithm based on the [`inspyred`](https://inspyred.readthedocs.io/en/latest/) logic:

- `generator`: a function used to sample an initial population,
- `observer`: a function executed after each evaluation,
- `evaluator`: a function used to extract the quantities of interest (QoIs) of all population candidates,
- some additional optimization arguments.

The framework currently supports two optimization libraries : [`inspyred`](https://pythonhosted.org/inspyred/index.html) and [`pymoo`](https://pymoo.org/index.html).
Thus, `Optimizer` attributes and methods are passed to the evolutionary computation algorithm (see [`ec`](https://pythonhosted.org/inspyred/reference.html#ec-evolutionary-computation-framework) framework for `inspyred` and [algorithms](https://pymoo.org/algorithms/list.html) in `pymoo`) and its optimization method (see [`ec.evolve`](https://pythonhosted.org/inspyred/reference.html?highlight=evolve#inspyred.ec.EvolutionaryComputation.evolve) for `inspyred` and [`optimize.minimize`](https://pymoo.org/interface/minimize.html?highlight=minimize) for `pymoo`).

### Optimizer
The `Optimizer` abstract class extracts general arguments from the `"optim"` and `"study"` dictionaries of the configuration file such as:

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

Regardless of the optimization library, the `Optimizer` class acts as an `evaluator` and must hence implement an `_evaluate` method that is used during the optimization. However since they both have their own specificities in terms of candidate management, typing and structure, the choice has been made to inherit the `Optimizer` class is separately for each library.

!!! Tip
    The `Generator` class is based on [`scipy.qmc`](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html) samplers. It supports three different sampling techniques: ["lhs"](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube), ["halton"](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html#scipy.stats.qmc.Halton) and ["sobol"](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html#scipy.stats.qmc.Sobol). The sampling technique is selected with the `sampler_name` entry of the `"optim"` dictionary in the configuration file.

!!! Note
    All optimizer parameters are described in their respective class definition (see [`Optimizer`](dev_optimizer.md#src.optimizer.Optimizer), [`WolfOptimizer (inspyred)`](dev_optimizer.md#src.inspyred_optimizer.WolfOptimizer), [`WolfOptimizer (pymoo)`](dev_optimizer.md#src.pymoo_optimizer.WolfOptimizer)).

### Wolf Optimizer
The `WolfOptimizer` class illustrates how `Optimizer` can be inherited to perform a `Wolf`-based optimization.

Regardless of the optimization library, tt first instantiates a `WolfSimulator` attribute that is then used in the `_evaluate` method where for all candidates, the following steps are performed:

1) geometry deformation,

2) deformed geometry meshing,

3) simulation execution,

4) post-processing i.e. QoI extraction and constraint application.

!!! Note
    Design constraints penalizing inadequate geometries both in terms of area and lift coefficient are managed with `constraint` for [`inspyred](https://inspyred.readthedocs.io/en/latest/recipes.html#constraint-selection) and `apply_inequality_constraint` for [`pymoo`](https://pymoo.org/constraints/index.html).

In the end, all simulations QoIs are returned either as a list of floats (with `inspyred`) or as a `numpy` array (with `pymoo`). In addition, after each evaluation the `_observe` method is called (automatically with `inspyred`, explicitly with `pymoo`) to write or display the results of each generation candidates.

### Quick Experiments
The `*_optim.py` scripts enable to launch full optimization loops in accordance with the configuration file specifications.

For instance, `naca_base.json` executes a single iteration of the [Evolution Strategy](https://pythonhosted.org/inspyred/examples.html#evolution-strategy) algorithm with 5 candidates and 8 variables of design sampled in [-0.5, 0.5] (in lattice units):
```py
# from aero-optim to naca_base
cd examples/NACA12/naca_base
inspyred -c naca_base.json
```
<p float="left">
  <img src="../Figures/dummy_optim.png" width="100%" />
</p>

!!! Tip
    In the configuration file, the `budget` entry should be adapted to the amount of resources available to the user.
