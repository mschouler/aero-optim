## Multi-Fidelity Surrogate based Optimization
Multi-Fidelity surrogate based optimization (MF-SBO) also is a common practice in aerodynamic shape optimization. In comparison to the standard SBO, the idea in this case is to build multiple initial design of experiments (DOE) with different fidelities (e.g. RANS vs LES, increasing mesh fineness, etc.). Then, these multi-fidelity datasets are used to train a multi-fidelity surrogate model of the problem's quantities of interest (QoIs). In the end, the multi-fidelity surrogate model, which is expected to be more precised than a mono-fidelity model, is used in the optimization loop in place of the CFD solver.

This tutorial illustrates the steps to do so with **Aero-Optim**:

1) a non-penalized low-fidelity DOE is built with a single generation optimization execution,

2) the best candidates amongst low-fidelity ones are used as candidates for the high-fidelity DOE that is then built with a single generation optimization execution,

3) the low-fidelity and high-fidelity results are used to train the multi-fidelity surrogate model of the user's choosing,

4) a full optimization based on a `CustomSimulator` and a `CustomOptimizer` is performed by evaluating candidates with the trained multi-fidelity surrogate model.

### Illustration
The `NACA12/naca_mf_smt` example shows how this can be done with two fidelities: the `naca_base` use-case with its coarse mesh for the first one, the `naca_adap` use-case with its adapted mesh for the second fidelity. Once again, the [`smt` toolbox](https://smt.readthedocs.io/en/latest/index.html#) is used to build and train the multi-fidelity surrogates.

First, the `naca_lf_doe.json` configuration file is built with `max_generations` set to 1 (the DOE is generated with `pymoo`):
```py
subprocess.run(["optim", "-c", "naca_lf_doe.json", "--pymoo"],
                env=os.environ,
                stdin=subprocess.DEVNULL,
                check=True)
```

Then, the results are loaded and the best candidates are selected to become those of the high-fidelity DOE:
```py
# lf data loading
X_lf = np.loadtxt(os.path.join(lf_outdir, "candidates.txt"))
Y_lf = []
with open(os.path.join(lf_outdir, "df_dict.pkl"), "rb") as handle:
    df_dict = pickle.load(handle)
for gid in range(len(df_dict)):
    for cid in range(len(df_dict[gid])):
        Y_lf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
Y_lf = np.array(Y_lf)
del df_dict
# hf candidates selection
best_candidates_idx = np.argsort(Y_lf[:, 0])
np.savetxt(
    os.path.join(hf_outdir, "custom_doe.txt"), X_lf[best_candidates_idx][:hf_doe_size]
)
# hf data generation
subprocess.run(["optim", "-c", f"naca_hf_doe.json", "--pymoo"],
                env=os.environ,
                stdin=subprocess.DEVNULL,
                check=True)
```
Where `naca_hf_doe.json` is based on the `naca_adp.json` configuration file with an additional parameter enabling the `Generator` to fetch the newly selected candidates:
- `"custom_doe": "output_hf_doe/custom_doe.txt"` in the `"optim"` entry.

The high-fidelity results are next loaded and used to train the multi-fidelity surrogate models for both `Cd` and `Cl`:
```py
X_hf = np.loadtxt(os.path.join(hf_outdir, "candidates.txt"))
Y_hf = []
with open(os.path.join(hf_outdir, "df_dict.pkl"), "rb") as handle:
    df_dict = pickle.load(handle)
for gid in range(len(df_dict)):
    for cid in range(len(df_dict[gid])):
        Y_hf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
Y_hf = np.array(Y_hf)
del df_dict
# Cd
mfsm_cd = MFK(theta0=X_lf.shape[1] * [1.0])
mfsm_cd.set_training_values(X_lf, Y_lf[:, 0], name=0)
mfsm_cd.set_training_values(X_hf, Y_hf[:, 0])
mfsm_cd.train()
# Cl
mfsm_cl = MFK(theta0=X_lf.shape[1] * [1.0])
mfsm_cl.set_training_values(X_lf, Y_lf[:, 1], name=0)
mfsm_cl.set_training_values(X_hf, Y_hf[:, 1])
mfsm_cl.train()
```

The exact same `CustomSM` class as the one introduced in the [SBO example](example_smo.md) is used to combine both models and emulate what would have been extracted from a simulation run:
```py
custom_mf_sm = CustomSM([sm_cd, sm_cl])
with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
    pickle.dump(custom_mf_sm, handle)
```

Finally, a full optimization using this multi-fidelity surrogate model is performed with the same `naca_smt.json` configuration file and two additional parameters:

- `"custom_file": "../naca_smt/custom_sm.py"` in the `"study"` entry,
- `"model_file": "output_hf_doe/model.pkl"` in the `"simulator"` entry.

```py
subprocess.run(["optim", "-c", "naca_smt.json", "--pymoo"],
               env=os.environ,
               stdin=subprocess.DEVNULL,
               check=True)
```

### Quick Experiments
In order to run this examples, the `smt` library must first be added to the virtual environment:
```sh
pip install smt
```
Then, the `main_mf_sm.py` script in the `NACA12/naca_mf_smt` example folder can be used to perform all three steps at once:
```py
# from aero-optim to naca_mf_smt
cd examples/NACA12/naca_mf_smt
python3 main_mf_sm.py -clf naca_lf_doe.json -chf naca__hf_doe.json -cmfsm naca_smt.json
```

It will produce an `output_lf_doe` folder with the results of the low-fidelity DOE generation, an `output_hf_doe` folder with the high-fidelity results and `output_smt` with the results of the multi-fidelity surrogate based optimization.

At that point, the optimal profile properties obtained with the surrogate can be compared to its corresponding CFD simulation:
```sh
# deformed profile generation
ffd -f ../data/naca12.dat -nc 4 -d "<displacement of the optimal profile>" -o output_optim
# deformed profile meshing
mesh -c naca_hf_doe.json -f output_optim/naca12_g0_c0.dat -o output_optim
# simulation execution
simulator -c naca_hf_doe.json -f output_optim/naca_base.mesh -o optim_profile
```