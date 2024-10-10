import argparse
import copy
import dill as pickle
import functools
import numpy as np
import os
import subprocess
import time

from aero_optim.ffd.ffd import FFD_POD_2D
from aero_optim.mf_sm.mf_models import get_model, get_sampler, MultiObjectiveModel
from aero_optim.utils import check_config
from custom_cascade import execute_single_gen

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
    model_ADP = get_model(
        model_name=model_name,
        dim=n_design,
        config_dict=sm_config.get(model_name, {}),
        outdir=outdir,
        seed=seed
    )
    model_OP = copy.deepcopy(model_ADP)
    # get sampler and lf / hf DOEs
    print("MFSM: sampler selection..")
    mf_sampler = get_sampler(
        dim=n_design,
        bounds=bound,
        seed=seed,
        nested_doe=model_ADP.requires_nested_doe
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
    QoI = sm_config["optim"]["QoI"]
    lf_w_ADP = np.array([lf_dict[0][cid]["ADP"][QoI].iloc[-1] for cid in range(len(lf_dict[0]))])
    lf_w_OP = np.array(
        [0.5 * (lf_dict[0][cid]["OP1"][QoI].iloc[-1] + lf_dict[0][cid]["OP2"][QoI].iloc[-1])
         for cid in range(len(lf_dict[0]))]
    )
    # hf_DOE
    hf_w_ADP = np.array([hf_dict[0][cid]["ADP"][QoI].iloc[-1] for cid in range(len(hf_dict[0]))])
    hf_w_OP = np.array(
        [0.5 * (hf_dict[0][cid]["OP1"][QoI].iloc[-1] + hf_dict[0][cid]["OP2"][QoI].iloc[-1])
         for cid in range(len(hf_dict[0]))]
    )
    print("MFSM: training models..")
    model_ADP.set_DOE(x_lf=x_lf, y_lf=lf_w_ADP, x_hf=x_hf, y_hf=hf_w_ADP)
    model_OP.set_DOE(x_lf=x_lf, y_lf=lf_w_OP, x_hf=x_hf, y_hf=hf_w_OP)
    model = MultiObjectiveModel([model_ADP, model_OP])
    model.train()
    print(f"MFSM: finished training after {time.time() - t0} seconds")
    # saves mf-sm
    with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
        pickle.dump(model, handle)
    print(f"MFSM: models saved to {outdir}")

    # MFSM based optimization with adaptive infill
    print("MFSM: surrogate model based optimization..")
    exec_cmd = ["optim", "-c", f"{args.config}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"MFSM: surrogate model based optimization finished after {time.time() - t0} seconds")


if __name__ == '__main__':
    main()
