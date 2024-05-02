## Simulator Module
The simulator module is designed to orchestrate one or several simulation executions with a given solver.
Hence, an `Simulator` abstract class extracts general arguments from the `"simulator"` dictionary of the configuration file such as:

- `exec_cmd (str)`: the solver execution command that is to be launched in the terminal,
- `ref_input (str)`: the path to the solver input template,
- `sim_args (dict)`: simulation arguments with which to customize the template,
- `post_process (dict)`: post-processing arguments to indicate which variables to extract from which result files.

Then, any subclass inheriting from `Simulator` can be built by overriding two abstract methods:

1. `process_config`: which goes through the configuration file making sure expected entries are well defined,
2. `execute_sim`: which defines how a simulation should be executed.

!!! Note
    All simulator parameters are described in their respective class definition (see [`Simulator`](dev_simulator.md#src.simulator.Simulator), [`WolfSimulator`](dev_simulator.md#src.simulator.WolfSimulator)).

### Wolf Simulator
The `WolfSimulator` class illustrates how `Simulator` can be inherited to perform `Wolf` simulations. In addition to the mandatory methods, several others are introduced in order to facilitate the simulation progress monitoring (see `sim_pro` and `monitor_sim_progress`) and the results post-processing (see `post_process`).

Finally, a mechanism of fault management is introduced with a `kill_all` method. The idea here is simply to provide a function to kill all active simulations in case something goes wrong in the main program. In the context of an optimization for instance, this feature is particularly important if the user wants to interrupt the main program without having to kill all simulations by hand.

### Advanced Simulator
Considering the variety of solvers, execution environment, etc., the `Simulator` class is theoretically adaptable to any use case. In general, it is good practice to keep track of any [Python subprocess](https://docs.python.org/3/library/subprocess.html) spawned by the application. However, depending on the user's situation, more sophisticated monitoring routines may need to be implemented.

For a cluster using a [slurm batch scheduler](https://slurm.schedmd.com/documentation.html) for instance, there are multiple ways to submit simulations whether they are submitted interactively (e.g. with `srun`) or not (e.g. with `sbatch`). In the latter case, progress monitoring cannot be made through simple subprocess tracking but by interacting with the batch scheduler database (e.g. with `sacct`). Illustrative code on how to do so is available in the [melissa repository](https://gitlab.inria.fr/melissa/melissa) (see [scheduler](https://gitlab.inria.fr/melissa/melissa/-/tree/develop/melissa/scheduler)).
