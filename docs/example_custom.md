## Customized Optimization
Although this framework was designed to perform simple forms of aerodynamic optimization, it still covers a broad range of applications and use-cases. It seems therefore unrealistic to see this framework as a universal tool. Yet, a specific care was taken when designing and assembling the building parts of it in order to make it as customizable as possible. 

Hence any optimization part of the framework can be customized with the following steps:

1) create a `<custom-script>.py` file in the working directory (e.g. `inspyred_debug.py` or `pymoo_debug.py`),

2) inherit all classes to be customized respecting the class naming format i.e respectively `CustomMesh`, `CustomSimulator`, `CustomOptimizer` and `CustomEvolution` if overriding a `Mesh`, `Simulator`, `Optimizer` or `Evolution` based class,

3) add a `custom_file` entry specifying the path to the custom script in the `"study"` sub-dictionary of the configuration file.

!!! Note
    As of now, the `FFD_2D` class is the only static element of the framework. However, extending `Optimizer` to enable its customization would be pretty straightforward.

### Illustration
The scripts in the [`debug` example](https://github.com/mschouler/aero-optim/tree/master/examples/debug) illustrate how to customize the `Simulator`, `Optimizer` and `Evolution` classes both with `pymoo` (or `inspyred`).

First, the `Simulator` class is customized:
```py
class CustomSimulator(DebugSimulator):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM SIMULATOR")
```

Then, the `Optimizer` class is customized:
```py
class CustomOptimizer(PymooDebugOptimizer):
    def __init__(self, config: dict):
        super().__init__(config)
        logger.info("INIT CUSTOM OPTIMIZER")
```

Finally, the `Evolution` class is customized:
```py
class CustomEvolution(PymooEvolution):
    def set_ea(self):
        logger.info("SET CUSTOM EA")
        self.ea = PSO(
            pop_size=self.optimizer.doe_size,
            sampling=self.optimizer.generator._pymoo_generator(),
            **self.optimizer.ea_kwargs
        )

    def evolve(self):
        logger.info("EXECUTE CUSTOM EVOLVE")
        res = minimize(problem=self.optimizer,
                       algorithm=self.ea,
                       termination=get_termination("n_gen", self.optimizer.max_generations),
                       seed=self.optimizer.seed,
                       verbose=True)
        self.optimizer.final_observe()

        # output results
        best = res.F
        index, opt_J = min(enumerate(self.optimizer.J), key=lambda x: abs(best - x[1]))
        gid, cid = (index // self.optimizer.doe_size, index % self.optimizer.doe_size)
        logger.info(f"optimal J: {opt_J} (J_pymoo: {best}),\n"
                    f"D: {' '.join([str(d) for d in self.optimizer.inputs[gid][cid]])}\n"
                    f"D_pymoo: {' '.join([str(d) for d in res.X])}\n"
                    f"[g{gid}, c{cid}]")
```

Of course, this is a purely illustrative example and the customization simply consists in overriding these upper level classes with already existing ones. It is still representative of how it can be done.

### Quick Experiments
Running the following commands will execute a simple optimization with `pymoo`:
```sh
# from aero-optim to debug
cd examples/debug
optim -c debug.json
```
Since the `pymoo_debug.py` script is given in `debug.json` and `CustomEvolution` inherits from `PymooEvolution`, in this case, there is no need to specify the optimization library which is automatically selected.

The custom script can also be superseded by passing it to `optim` with the `--file` option. Hence, simply running the command below will execute an optimization with `inspyred`:
```sh
# from aero-optim to debug
cd examples/debug
optim -c debug.json -f inspyred_debug.py
```