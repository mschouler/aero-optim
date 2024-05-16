## Welcome to AERO-Optim
AERO-Optim is a simple aerodynamic shape optimization framework coupling FreeForm Deformation (FFD), automatic meshing with [`gmsh`](https://gmsh.info/doc/texinfo/gmsh.html) and any CFD solver execution in the frame of an optimization algorithm based on [`inspyred`](https://inspyred.readthedocs.io/en/latest/). It is composed of the following core components:

* [FFD](ffd.md): which defines a class to perform 2D FFD of a given geometry,
* [Mesh](mesh.md): which defines multiples classes to generate automatic meshes,
* [Simulator](simulator.md): which defines a class to orchestrate CFD simulations including pre- and post-processing steps as well as progress monitoring,
* [Optimizer](optimizer.md): which defines a class to coordinate the optimization procedure according to [`inspyred`](https://inspyred.readthedocs.io/en/latest/) conventions.

### Quick Installation
AERO-Optim comes with few dependencies listed in [`requirements.txt`](https://github.com/mschouler/aero-optim/blob/master/requirements.txt) and recalled below:
```sh
gmsh        # to design and visualize meshes (MESH)
inspyred    # optimzation toolbox (OPTIM)
numpy       # to manipulate geometries as arrays (FFD)
matplotlib  # to visualize the generated deformed profiles (FFD)
pandas      # to load simulation results (OPTIM)
scipy       # to use quasi monte carlo samplers (FFD)
```

From the user's working directory, they can all be installed at once in a [virtual environment](https://docs.python.org/3/library/venv.html) with the following commands:
```sh
git clone https://github.com/mschouler/aero-optim.git
cd aero-optim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

!!! Warning
    So far, the framework was only installed and tested on MacOS and Linux platforms.

### First Execution
The framework was designed to automate aerodynamic optimization which, in this case, consists in:

1) deforming a baseline geometry,

2) generating its associated mesh,

3) performing a CFD computation,

as many times as required to obtain a geometry maximizing/minimizing certain quantities of interest (QoIs).

Although the chaining of these steps is performed under the supervision of [`main_optim.py`](https://github.com/mschouler/aero-optim/blob/add-docs/main_optim.py), the three subscripts presented below were introduced to facilitate each step's individual evaluation:

#### First FFD: [`auto_ffd.py`](https://github.com/mschouler/aero-optim/blob/master/auto_ffd.py)
This script performs one or multiple FFD of the geometry passed as its input argument. For instance:
```sh
# from aero-optim to naca_base
cd examples/NACA12/naca_base
ffd -f ../data/naca12.dat -nc 2 -d "0. 0. 1. 1."
```
will yield the figure below:
<p float="left">
  <img src="./Figures/naca12.png" width="100%" />
</p>
where the deformation vector is $$[D_{10}, D_{20}, D_{11}, D_{21}] = [0., 0., 1., 1.]$$ in lattice unit (see [FFD](ffd.md)).

#### First Mesh: [`auto_gmsh.py`](https://github.com/mschouler/aero-optim/blob/master/auto_gmsh.py)
This script generates a simple mesh parameterized according to its associated configuration file. For instance:
```sh
# from aero-optim to naca_base
cd examples/NACA12/naca_base
mesh --config=naca_base.json  # left figure
cd ../naca_block
mesh --config=input/naca_block.json # right figure
```
will generate the figures below:
<p float="left">
  <img src="./Figures/naca_base_mesh.png" width="49%" />
  <img src="./Figures/naca_block_mesh.png" width="49%" /> 
</p>

#### First Simulation: [`auto_simulator.py`](https://github.com/mschouler/aero-optim/blob/master/auto_simulator.py)
This script performs a single simulation according to its associated configuration file and mesh. For instance:
```sh
# from aero-optim to naca_base
cd examples/NACA12/naca_base
simulator -c naca_base.json -f output/naca_base.mesh
```
would run a [`Wolf`](https://pages.saclay.inria.fr/frederic.alauzet/software.html) simulation provided that the user has access to the solver and that they have properly specified the path to the executable:
```sh
INFO:src.simulator.simulator:g0, c0 converged in 518 it.
INFO:src.simulator.simulator:last five values:
      ResTot      CD       CL        ResCD        ResCL   x   y  Cp
1.188239e-06 0.15354 0.354171 2.169250e-15 2.664503e-15 NaN NaN NaN
7.784275e-07 0.15354 0.354171 3.615416e-16 7.836773e-16 NaN NaN NaN
5.064301e-07 0.15354 0.354171 2.530791e-15 2.037561e-15 NaN NaN NaN
3.286895e-07 0.15354 0.354171 2.530791e-15 1.410619e-15 NaN NaN NaN
2.132616e-07 0.15354 0.354171 9.038540e-16 2.037561e-15 NaN NaN NaN
>> simulation finished after 4.014565944671631 seconds
```

!!! Warning
    In order for the command to pass, don't forget to fix the `exec_cmd` executable path in `input/naca_base.json`.

!!! Note
    As of 2024, [`Wolf`](https://pages.saclay.inria.fr/frederic.alauzet/software.html) is not open-source and may not be available to the user. Guidelines on how to replace it with another solver are given in [Simulator](simulator.md).

### Coverage Report
