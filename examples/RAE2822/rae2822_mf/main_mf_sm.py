import argparse
import dill as pickle
import functools
import numpy as np
import os
import subprocess
import time

from aero_optim.ffd.ffd import FFD_POD_2D
from aero_optim.mf_sm.mf_infill import maximize_ED
from aero_optim.mf_sm.mf_models import get_model, get_sampler
from aero_optim.utils import check_config
from custom_rae2822 import execute_infill, execute_single_gen

print = functools.partial(print, flush=True)


def init_problem(config: str, pod: bool) -> tuple[dict, int, tuple[list[float], list[float]], int]:
    sm_config, _, _ = check_config(config, optim=True)
    seed = sm_config["optim"].get("seed", 123)
    if pod:
        ffd_pod = FFD_POD_2D(
            dat_file=sm_config["study"]["file"],
            pod_ncontrol=sm_config["ffd"]["pod_ncontrol"],
            ffd_ncontrol=sm_config["optim"]["n_design"],
            ffd_dataset_size=sm_config["ffd"]["ffd_dataset_size"],
            ffd_bound=sm_config["optim"]["bound"],
            seed=seed
        )
        n_design = ffd_pod.pod_ncontrol
        bound = ffd_pod.get_bound()
    else:
        n_design = sm_config["optim"]["n_design"]
        bound = sm_config["optim"]["bound"]
    return sm_config, n_design, bound, seed


def compute_lf_infill(model, infill_lf_size, infill_nb_gen, n_design, bound, seed) -> np.ndarray:
    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    infill_lf = np.array([])
    for sid in range(infill_lf_size):
        infill_lf_ED = maximize_ED(
            model=model,
            DOE=current_DOE,
            n_var=n_design,
            bound=bound,
            seed=seed,
            n_gen=infill_nb_gen
        )
        infill_lf = infill_lf_ED if sid == 0 else np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
    return infill_lf


def get_best_candidate(outdir: str) -> np.ndarray:
    # there is no penalization so the best candidates can be selected from output files
    candidates = np.loadtxt(os.path.join(outdir, "candidates.txt"))
    fitnesses = np.loadtxt(os.path.join(outdir, "fitnesses.txt"))
    best_idx = np.argmin(fitnesses)
    return candidates[best_idx]


def main():
    """
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/lconfig.json")
    parser.add_argument("-p", "--pod", action="store_true", help="perform POD data reduction")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()
    sm_config, n_design, bound, seed = init_problem(args.config, args.pod)
    outdir = sm_config["study"]["outdir"]

    # get model
    print("MFSM: model selection..")
    model_name = sm_config["optim"]["model_name"]
    model = get_model(
        model_name=model_name,
        dim=n_design,
        config_dict=sm_config.get(model_name, {}),
        outdir=outdir,
        seed=seed
    )

    # get sampler and lf / hf DOEs
    print("MFSM: sampler selection..")
    mf_sampler = get_sampler(
        dim=n_design,
        bounds=bound,
        seed=seed,
        nested_doe=model.requires_nested_doe
    )
    x_lf, x_hf = mf_sampler.sample_mf(
        n_lf=sm_config["optim"]["n_lf"],
        n_hf=sm_config["optim"]["n_hf"]
    )

    # generate lf_DOE
    print("MFSM: LF DOE computation..")
    lf_dir = os.path.join(outdir, "lf_doe")
    lf_dict = execute_single_gen(
        outdir=lf_dir,
        config=sm_config["optim"]["lf_config"],
        X=x_lf,
        name="lf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: LF DOE computation finished after {time.time() - t0} seconds")

    # generate hf_DOE
    print("MFSM: HF DOE computation..")
    hf_dir = os.path.join(outdir, "hf_doe")
    hf_dict = execute_single_gen(
        outdir=hf_dir,
        config=sm_config["optim"]["hf_config"],
        X=x_hf,
        name="hf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: HF DOE computation finished after {time.time() - t0} seconds")

    # train and save surrogates
    # lf_DOE
    lf_cl = np.array([lf_dict[0][cid]["CL"].iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_cd = np.array([lf_dict[0][cid]["CD"].iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_cl_over_cd = -lf_cl / lf_cd
    # hf_DOE
    hf_cl = np.array([hf_dict[0][cid]["CL"].iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_cd = np.array([hf_dict[0][cid]["CD"].iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_cl_over_cd = -hf_cl / hf_cd
    print("MFSM: training model..")
    model.set_DOE(x_lf=x_lf, y_lf=lf_cl_over_cd, x_hf=x_hf, y_hf=hf_cl_over_cd)
    model.train()
    print(f"MFSM: finished training after {time.time() - t0} seconds")
    # save mf-sm
    with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
        pickle.dump(model, handle)
    print(f"MFSM: model saved to {sm_config['study']['outdir']}")

    # MFSM based optimization with adaptive infill
    print("MFSM: surrogate model based optimization..")
    nite = sm_config["optim"]["infill_nb"]
    infill_size = sm_config["optim"]["infill_lf_size"]
    infill_nb_gen = sm_config["optim"]["infill_nb_gen"]
    for ite in range(nite):
        outdir_ite = os.path.join(outdir, outdir + f"_{ite}")
        # optimization
        exec_cmd = [
            "optim",
            "-c", f"{args.config}", "-o", f"{outdir_ite}", "-v", f"{args.verbose}",
            "--pymoo"
        ]
        subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
        # lf infill
        x_lf_infill = compute_lf_infill(model, infill_size, infill_nb_gen, n_design, bound, seed)
        y_lf_infill = execute_infill(
            x_lf_infill,
            config=sm_config["optim"]["lf_config"],
            n_design=sm_config["optim"]["n_design"],
            outdir=outdir_ite,
            ite=ite,
            fidelity="low"
        )
        # hf infill
        x_hf_infill = get_best_candidate(outdir_ite)
        y_hf_infill = execute_infill(
            x_hf_infill,
            config=sm_config["optim"]["hf_config"],
            n_design=sm_config["optim"]["n_design"],
            outdir=outdir_ite,
            ite=ite,
            fidelity="high"
        )
        print(f"MFSM: hf infill candidate {x_hf_infill} with fitness {y_hf_infill}")
        # update model
        model.set_DOE(x_lf=x_lf_infill, y_lf=y_lf_infill, x_hf=x_hf_infill, y_hf=y_hf_infill)
        model.train()
        print(f"MFSM: surrogate model based optimization {ite + 1}/{nite}"
              f" finished after {time.time() - t0} seconds")
    # save y_hf and final model
    np.savetxt(os.path.join(outdir, "fitnesses.txt"), model.y_hf_DOE)
    with open(os.path.join(outdir, "final_model.pkl"), "wb") as handle:
        pickle.dump(model, handle)


if __name__ == '__main__':
    main()
