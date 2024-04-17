## Simulator Module
The simulator module is designed to orchestrate one or several simulation executions with a given solver.
Hence, an asbtract [`Simulator`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L13-L77) class extracts general arguments from the `"simulator"` dictionary of the configuration file such as:

- the solver execution command that is to be launched in the terminal,
- the path to the solver input template,
- simulation arguments with which to customize the template,
- post-processing arguments to indicate which variables to extract from which result files.

Then, any subclass inheriting from `Simulator` can be build by overriding two abstract classes:

1. `process_config`: which goes through the configuration file making sure expected entries are well defined,
2. `execute_sim`: which defines how a simulation should be executed.

### Wolf Simulator
The [`WolfSimulator`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L80-L202) class illustrates how `Simulator` can be inherited to perform `Wolf` simulations. In addition to the mandatory methods, several others are introduced in order to facilitate the simulation progress monitoring (see [`sim_pro`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L93) and [`monitor_sim_progress`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L152-L169)) and the results post-processing (see [`post_process`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L171-L195)).

Finally, a mechanism of fault management is introduced with a [`kill_all`](https://github.com/mschouler/aero-optim/blob/master/src/simulator.py#L197-L202) method. The idea here is simply to provide a function to kill all active simulations in case something goes wrong in the main program. In the context of an optimization for instance, this feature is particularly important if the user wants to interrupt the main program without having to kill all simulations by hand.

### Advanced Simulator
Considering the variety of solvers, execution environment, etc., the `Simulator` class is theoretically adaptable to any use case. In general, it is good practice to keep track of any [Python subprocess](https://docs.python.org/3/library/subprocess.html) spawned by the application. However, depending on the user's situation, more sophisticated monitoring routines may need to be implemented.

For a cluster using a [slurm batch scheduler](https://slurm.schedmd.com/documentation.html) for instance, there are multiple ways to submit simulations whether they are submitted interactively (e.g. with `srun`) or not (e.g. with `sbatch`). In the latter case, progress monitoring cannot be made through simple subprocess tracking but by interacting with the batch scheduler database (e.g. with `sacct`). Illustrative code on how to do so is available in the [melissa repository](https://gitlab.inria.fr/melissa/melissa) (see [scheduler](https://gitlab.inria.fr/melissa/melissa/-/tree/develop/melissa/scheduler)).