## Multi-Fidelity Surrogate Module

This module implements two major bricks required to perform multi-fidelity surrogate-assisted optimization:

1. `mf_models` which defines model classes to be used for single- / multi-fidelity and single- / multi-objective prediction,
2. `mf_infill` which defines sub-optimization problems to be minimized or maximized in the context of Bayesian or non-Bayesian infill strategies.

All features related to surrogate models come with additional dependencies listed in `requirements_sm.txt` and recalled below:
```sh
torch       # Pytorch for deep learning based models (MFDNN)
smt         # for single- and multi-fidelity kriging (MfSMT)
GPy         # for co-kriging and other Gaussian processes models
emukit      # for co-kriging and other Gaussian processes models
```

These dependencies can be added to the user's environment with the following command:
```sh
# from aero-optim
pip install -r requirements_sm.txt
```

### Multi-Fidelity Models

The `MfModel` abstract class defines how any multi-fidelity surrogate model can be wrapped. It is initialized with the following positional arguments:

- `dim (int)` which indicates the dimension of the problem,
- `model_dict (dict)` which contains model specific information,
- `outdir (str)` the directory where model parameters should be saved,
- `seed (int)` the random seed the model will be initialized with,
- `x_lf_DOE (np.ndarray)` and `y_lf_DOE (np.ndarray)` the low-fidelity initial DOE,
- `x_hf_DOE (np.ndarray)` and `y_hf_DOE (np.ndarray)` the high-fidelity initial DOE.

When inherited, the following methods should also be overridden:

- `train()` which defines how the model should be trained,
- `evaluate()` which defines how the model should be evaluated,
- `evaluate_std()` which defines how the model standard deviation should be computed when the model is of Bayesian nature.

In addition, any `MfModel` based model will have access to two base methods:

- `set_DOE()` which updates the model DOEs,
- `get_DOE()` which returns the model DOEs.

Except for Neural Network based models, most multi-fidelity models available in `mf_models` are single-output models, the `MultiObjectiveModel` class is implemented to turn any `MfModel` into a multi-objective model. To do so, `MfModel` objects are built for each objective and passed as a `list` when `MultiObjectiveModel` is initialized. This way, the use of `MultiObjectiveModel` is transparent to the user. The only difference occurs when the model DOEs are updated. For such models, the arrays of objectives must be passed as a list of 1D arrays via the `set_DOE()` method.

### Infill strategies

The `mf_infill` module implements various sub-optimization problems related to both single- and bi-objective adaptive infill. The single-objective adaptive infill strategies are:

- the Lower Confidence Bound minimized via `minimize_LCB`,
- the Expected Improvement maximized via `maximize_EI`.

The bi-objective infill strategies are:

- the Probability of Improvement maximized via `maximize_PI_BO`,
- the minimal Probability of Improvement maximized via `maximize_MPI_BO`.

!!! Note
    The multi-fidelity aspect of the surrogate model used to solve one of these sub-optimization problems only explicitly intervenes with bi-objective adaptive strategies which can use the Pareto front based on the current low- or high-fidelity DOE.

Regardless of the number of objectives, the max-min Euclidean Distance sub-optimization problems provides samples that harmonize the coverage of the design space. It is solved via `maximize_ED`.

### Quick Experiments

Surrogate models can be evaluated with 1D or multi-dimensional functions with the scripts contained in `scripts/MF-SM`. For instance, the `SMT` multi-fidelity co-kriging, can be evaluated on all four Brevault's 1D functions via the command below:
```sh
# from aero-optim to scripts/MF-SM
cd scripts/MF-SM
python3 main_mf.py -c config/example.json -o mfsmt -m mfsmt -f 1d -n 1
```
where `example.json` is a configuration file specifying the parameters of the training and evaluation.

If the configuration file contains a `"function nd"` entry, the model can be trained and evaluated with Brevault's multi-dimensional functions by simply changing the command line argument `-f` from `1d` to `nd`:
```sh
python3 main_mf.py -c config/example.json -o mfsmt -m mfsmt -f nd -n 1
```

!!! Note
    For 1D functions, the `"parameter"` entry of the configuration file indicates Brevault's functions parameters while for multi-dimensional functions, it indicates their dimension.

In addition to this script, two other notebooks designed for prototyping and validation purposes for [single-](https://github.com/mschouler/aero-optim/blob/master/scripts/MF-SM/infill_SO.ipynb) and [multi-objective](https://github.com/mschouler/aero-optim/blob/master/scripts/MF-SM/infill_MO.ipynb) infill strategies are available in `scripts/MF-SM`.