## Free-Form Deformation Module
The deformation module relies on an abstract class `Deform` theoretically compatible with any kind of geometric deformation but it was only subclassed for Free-Form Deformation (FFD) purposes so far. FFD is a technique designed to deform solid geometric models in a free-form manner. It was originally introduced by Sederberg in 1986 ([doi](https://dl.acm.org/doi/10.1145/15886.15903)). The present implementation is based on the description in [Duvigneau 2006](https://inria.hal.science/inria-00085058/). 

The main steps of FFD are:

1) create a lattice, i.e. the minimal parallelepiped box that embeds the geometry (a rectangle in 2D),

2) project the geometry points into the lattice referential such that its coordinates now vary between [0;1] in both x and y directions,

3) for a given deformation vector applied to each control points of the lattice, the displacements of all points forming the geometry are computed with the tensor product of the Bernstein basis polynomials,

4) the deformed profile is finally projected back into the original referential.

!!! Note
    This implementation is limited to 2D geometries only. More sophisticated libraries exist such as [PyGeM](https://mathlab.github.io/PyGeM/index.html) or [pyGeo](https://mdolab-pygeo.readthedocs-hosted.com/en/latest/introduction.html) but they come with heavier dependencies and more cumbersome installation procedures.

### 2D FFD
Free-Form Deformation is implemented in `FFD_2D`, a straightforward class instantiated with two positional arguments:

- `file (str)` which indicates the filename of the baseline geometry to be deformed,
- `ncontrol (int)` which indicates the number of design points on each side of the lattice.

!!! Warning
    The input file is expected to have a specific formatting i.e. a 2 line header followed by coordinates given as tabulated entries (one point per row) with single space separators (see [`data/naca12.dat`](https://github.com/mschouler/aero-optim/blob/master/examples/NACA12/data/naca12.dat) for an example).

Once instantiated, the `apply_ffd(Delta)` method can be used to perform the deformation corresponding to the array `Delta`.

#### Illustration
An FFD with 2 control points (i.e. 4 in total) and the deformation vector `Delta = (0., 0., 1., 1.)` will yield the following profile:
<p float="left">
  <img src="../Figures/naca12.png" width="100%" />
</p>
Considering the figure notations, one notices that the deformation vector `Delta` corresponds to the list of deformations to be applied to the lower control points followed by those applied to the upper control points. In this case, `Delta` is: $$(D_{10}=0.,\, D_{20}=0.,\, D_{11}=1.,\, D_{21}=1.)$$ in lattice unit. The corner points are left unchanged so that the lattice corners remain fixed.

#### Quick Experiments
The `auto_ffd.py` scripts is called with the `ffd` command. It enables basic testing and visualization. It comes with a few options:
```py
ffd --help
usage: ffd [-h] [-f FILE] [-o OUTDIR] [-nc NCONTROL] [-np NPROFILE] [-r] [-s SAMPLER] [-d [DELTA ...]]

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  baseline geometry: --datfile=/path/to/file.dat (default: None)
  -o OUTDIR, --outdir OUTDIR
                        output directory (default: output)
  -nc NCONTROL, --ncontrol NCONTROL
                        number of control points on each side of the lattice (default: 3)
  -np NPROFILE, --nprofile NPROFILE
                        number of profiles to generate (default: 3)
  -r, --referential     plot new profiles in the lattice ref. (default: False)
  -s SAMPLER, --sampler SAMPLER
                        sampling technique [lhs, halton, sobol] (default: lhs)
  -d DELTA, --delta DELTA
                        Delta: 'D10 D20 .. D2nc' (default: None)
```

For instance, the command below will perform 3 control points FFDs for 4 random deformations sampled with an LHS sampler:
```py
# from aero-optim to naca_base
cd examples/NACA12/naca_base
ffd -f ../data/naca12.dat -np 4 -nc 3 -s lhs
```
<p float="left">
  <img src="../Figures/naca12_lhs.png" width="100%" />
</p>

!!! Note
    The positions of the deformed control points on the generated figure correspond to those of the last profile, in this case `Pid-3`.

### 2D FFD & POD
The coupling between POD and FFD was implemented in the `FFD_pod_2D` class. Its objective is to build a dataset of `n` FFD perturbed profiles sampled from LHS and to use it to perform POD-based data reduction from the FFD dimension `d` to a smaller dimension `d*`. Details on how to do so are given in ([POD 1](https://doi.org/10.2514/6.2008-6584)) and ([POD 2](https://doi.org/10.2514/6.2017-3057)).

The `FFD_pod_2D` class is instantiated with 5 positional arguments:

- `file (str)` which indicates the filename of the baseline geometry to be deformed,
- `pod_ncontrol (int)` which indicates the number of reduced design points,
- `ffd_ncontrol (int)` which indicates the number of FFD control points design points,
- `ffd_dataset_size (int)` which indicates the size of the FFD dataset used in the POD procedure,
- `ffd_bound (tuple[Any])` which set the min/max displacement ranges of the FFD dataset used in the POD procedure.

To use this functionality, the `ffd_type` of the `"study"` entry must be set to `"ffd_pod_2d"`. The number of FFD control points and boundaries are still given by the `"n_design"` and `"bound"` values of the `"optim"` entry. In the `"ffd"` entry, the POD reduced dimension is set via `"pod_ncontrol"` and the size of the FFD dataset with `"ffd_dataset_size"`.

#### Illustration
An application of this feature is illustrated in the [POD notebook](https://github.com/mschouler/aero-optim/blob/master/scripts/FFD/POD.ipynb).