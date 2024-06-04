## Customized Optimization
Although this framework was designed to perform simple forms of aerodynamic optimization, it still covers a broad range of applications and use-cases. It seems therefore unrealistic to see this framework as a universal tool. Yet, a specific care was taken when designing and assembling the building parts of it in order to make it as customizable as possible. 

Hence any optimization part of the framework can be customized with the following steps:

1) create a `<custom-script>.py` file in the working directory (e.g. `inspyred_debug.py` or `pymoo_debug.py`),

2) inherit all classes to be customized respecting the class naming format i.e respectively `CustomMesh`, `CustomSimulator`, `CustomOptimizer` and `CustomEvolution` if overriding a `Mesh`, `Simulator`, `Optimizer` or `Evolution` based class,

3) add a `custom_file` entry specifying the path to the custom script in the `"study"` sub-dictionary of the configuration file.

!!! Note
    As of now, the `FFD_2D` class is the only static element of the framework. However, extending `Optimizer` to enable its customization would be pretty straightforward.

### Quick Experiments
The scripts in the [`debug` example](https://github.com/mschouler/aero-optim/tree/master/examples/debug) illustrate how to customize the `Simulator`, `Optimizer` and `Evolution` classes both with `pymoo` and `inspyred`. Of course, this is a purely illustrative examples and the customization simply consists in overriding these upper level classes with already existing ones. It is still representative of how it can be done.

Running the following command will execute a simple optimization with `pymoo`:
```sh
# from aero-optim to debug
cd examples/debug
optim -c debug.json
```

Since the `pymoo_debug.py` script is explicitly given in `debug.json`, there is no need to specify the optimization library which is automatically selected. The script can however be superseded by passing it to `optim` with the `--file` option. Hence, running the command below will execute a simple optimization with `inspyred`:
```sh
# from aero-optim to debug
cd examples/debug
optim -c debug.json -f inspyred_debug.py
```