## Surrogate based Optimization
Surrogate based optimization (SBO) is a common practice in aerodynamic shape optimization. The main idea is to build an initial design of experiments (DOE) that is then used to train a surrogate model of the problem's quantities of interest (QoIs). In the end, the surrogate model is used in the optimization loop in place of the CFD solver.

This tutorial illustrates the steps to do so with **Aero-Optim**:

1) a non-penalized DOE is built with a single generation optimization execution,

2) the results are used to train the surrogate model of the user's choosing,

3) a full optimization based on a `CustomSimulator` and a `CustomOptimizer` is performed by evaluating candidates with the trained surrogate model.

### Illustration
The `NACA12/naca_smt` example shows how this can be done for the `naca_base` use-case and the [`smt` toolbox](https://smt.readthedocs.io/en/latest/index.html#).

First, the `naca_doe.json` configuration file is built with `max_generations` set to 1 (the DOE is generated with `pymoo`):
```py
subprocess.run(["optim", "-c", "naca_doe.json", "--pymoo"],
               env=os.environ,
               stdin=subprocess.DEVNULL,
               check=True)
```

Then, the results are loaded and used to train surrogate models for both `Cd` and `Cl`:
```py
X = np.loadtxt(os.path.join(outdir, "candidates.txt"))
Y = []
with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
    df_dict = pickle.load(handle)
for gid in range(len(df_dict)):
    for cid in range(len(df_dict[gid])):
        Y. append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
Y = np.array(Y)
del df_dict
# Cd
sm_cd = KRG(theta0=[1e-2])
sm_cd.set_training_values(X, Y[:, 0])
sm_cd.train()
# Cl
sm_cl = KRG(theta0=[1e-2])
sm_cl.set_training_values(X, Y[:, 1])
sm_cl.train()
```

A straightforward `CustomSM` class is implemented to combine both models and emulate what would have been extracted from a simulation run:
```py
class CustomSM:
    def __init__(self, list_of_surrogates: list[KRG]):
        self.los: list[KRG] = list_of_surrogates

    def predict(self, x: np.ndarray) -> list[float]:
        return [sm.predict_values(x) for sm in self.los]  # [Cd, Cl]

custom_sm = CustomSM([sm_cd, sm_cl])
with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
    pickle.dump(custom_sm, handle)
```

Finally, a full optimization using this surrogate model specified is performed with the `naca_smt.json` configuration file and two additional parameters:

- `"custom_file": "custom_sm.py"` in the `"study"` entry,
- `"model_file": "output_doe/model.pkl"` in the `"simulator"` entry.

```py
subprocess.run(["optim", "-c", "naca_smt.json", "--pymoo"],
               env=os.environ,
               stdin=subprocess.DEVNULL,
               check=True)
```

For this instruction to work, the custom script `custom_sm.py` must also be implemented to define `CustomSimulator` and `CustomOptimizer`:
```py
class CustomSimulator(Simulator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.set_model(config["simulator"]["model_file"])

    def process_config(self):
        logger.info("processing config..")
        if "model_file" not in self.config["simulator"]:
            raise Exception(f"ERROR -- no <model_file> entry in {self.config['simulator']}")

    def set_solver_name(self):
        self.solver_name = "smt_model"

    def set_model(self, model_file: str):
        check_file(model_file)
        with open(model_file, "rb") as handle:
            self.model = pickle.load(handle)

    def execute_sim(self, candidates: list[float] | np.ndarray, gid: int = 0):
        logger.info(f"execute simulations g{gid} with {self.solver_name}")
        cd, cl = self.model.predict(np.array(candidates))
        self.df_dict[gid] = {
            cid: pd.DataFrame({"ResTot": 1., "CD": cd[cid], "CL": cl[cid]})
            for cid in range(len(candidates))
        }


class CustomOptimizer(PymooWolfOptimizer):
    def set_gmsh_mesh_class(self):
        self.MeshClass = None

    def execute_candidates(self, candidates, gid):
        logger.info(f"evaluating candidates of generation {self.gen_ctr}..")
        self.ffd_profiles.append([])
        self.inputs.append([])

        for cid, cand in enumerate(candidates):
            self.inputs[gid].append(np.array(cand))
            _, ffd_profile = self.deform(cand, gid, cid)
            self.ffd_profiles[gid].append(ffd_profile)

        self.simulator.execute_sim(candidates, gid)
```

!!! Tip
    Most surrogate models offer vectorized evaluation. It is therefore good practice to implement `CustomSimulator` in such a way that this aspect is leveraged.

!!! Warning
    Because the standard `pickle` library is not capable of unpickling an object whose class definition is unaccessible, [`dill`](https://github.com/uqfoundation/dill) is used instead.

### Quick Experiments
In order to run this examples, the `smt` library must first be added to the virtual environment:
```sh
pip install smt
```
Then, the `main_sm.py` script in the `NACA12/naca_smt` example folder can be used to perform all three steps at once:
```py
# from aero-optim to naca_smt
cd examples/NACA12/naca_smt
python3 -c naca_doe.json -csm naca_smt.json
```

It will produce an `output_doe` folder with the results of the initial DOE generation and `output_smt` with the results of the surrogate based optimization.

At that point, the optimal profile properties obtained with the surrogate can be compared to its corresponding CFD simulation:
 ```sh
ffd -f ../data/naca12.dat -nc 4 -d "<displacement of the optimal profile>" -o output_optim
mesh -c naca_doe.json -f output_optim/naca12_g0_c0.dat -o output_optim
simulator -c naca_doe.json -f output_optim/naca_base.mesh -o optim_profile
```