## AERO-Optim
AERO-Optim is a simple optimization framework coupling FreeForm Deformation (FFD), automatic meshing with [`gmsh`](https://gmsh.info/doc/texinfo/gmsh.html) and any CFD solver execution in the frame of an optimization algorithm based on [`inspyred`](https://inspyred.readthedocs.io/en/latest/). It is composed of the following core components:

* [`ffd.py`](): which defines a class to perform 2D FFD of a given geometry,
* [`*mesh.py`](): which defines multiples classes to generate automatic meshes,
* [`simulator.py`](): which defines a class to orchestrate CFD simulations including pre- and post-processing steps as well as progress monitoring,
* [ins_optimizer.py](): which defines a class to coordinate the optimization procedure according to [`inspyred`](https://inspyred.readthedocs.io/en/latest/) conventions.

The full documentation is available [**HERE**](https://mschouler.github.io/aero-optim/).

### Installation
AERO-Optim comes with few dependencies listed in [`requirements.txt`](./requirements.txt) and recalled below:
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
```

### To go further
Details about the following topics are available on the documentation:
* [First Execution](https://mschouler.github.io/aero-optim/#first-execution)
* [FFD Module](https://mschouler.github.io/aero-optim/ffd)
* [Mesh Module](https://mschouler.github.io/aero-optim/mesh)
* [Simulator Module](https://mschouler.github.io/aero-optim/simulator)
* [Optimizer Module](https://mschouler.github.io/aero-optim/optimizer)