import argparse
import copy
import dill as pickle
import functools
import numpy as np
import os
import subprocess
import time

from aero_optim.ffd.ffd import FFD_POD_2D
from aero_optim.mf_sm.mf_models import get_model, get_sampler, MfDNN, MultiObjectiveModel
from aero_optim.utils import check_config, mv_filelist
from custom_cascade import (compute_bayesian_infill, compute_non_bayesian_infill,
                            compute_pareto, execute_single_gen, execute_infill)

print = functools.partial(print, flush=True)


def get_mo_model(
        model_name: str, n_design: int, config: dict, outdir: str, seed: int
) -> MultiObjectiveModel | MfDNN:
    if model_name == "mfsmt":
        model_ADP = get_model(model_name, n_design, config, outdir, seed)
        model_OP = copy.deepcopy(model_ADP)
        return MultiObjectiveModel([model_ADP, model_OP])
    elif model_name == "mfdnn":
        model = get_model(model_name, n_design, config, outdir, seed)
        assert isinstance(model, MfDNN)
        return model
    else:
        raise Exception(f"{model_name} is currently not supported")


def set_mo_DOE(
        model: MfDNN | MultiObjectiveModel,
        x_lf: np.ndarray,
        y_lf: list[np.ndarray],
        x_hf: np.ndarray,
        y_hf: list[np.ndarray]
):
    if isinstance(model, MfDNN):
        model.set_DOE(x_lf=x_lf, x_hf=x_hf, y_lf=np.column_stack(y_lf), y_hf=np.column_stack(y_hf))
    elif isinstance(model, MultiObjectiveModel):
        model.set_DOE(x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf)
    else:
        raise Exception(f"{type(model)} is currently not supported")


def save_results(model: MfDNN | MultiObjectiveModel, outdir: str):
    if isinstance(model, MfDNN):
        assert model.x_lf_DOE is not None and model.x_hf_DOE is not None
        assert model.y_lf_DOE is not None and model.y_hf_DOE is not None
        np.savetxt(os.path.join(outdir, "lf_candidates.txt"), model.x_lf_DOE)
        np.savetxt(os.path.join(outdir, "lf_fitnesses.txt"), model.y_lf_DOE)
        np.savetxt(os.path.join(outdir, "hf_candidates.txt"), model.x_hf_DOE)
        np.savetxt(os.path.join(outdir, "hf_fitnesses.txt"), model.y_hf_DOE)
    elif isinstance(model, MultiObjectiveModel):
        assert model.models[0].x_lf_DOE is not None and model.models[0].x_hf_DOE is not None
        np.savetxt(os.path.join(outdir, "lf_candidates.txt"), model.models[0].x_lf_DOE)
        np.savetxt(os.path.join(outdir, "hf_candidates.txt"), model.models[0].x_hf_DOE)
        assert model.models[0].y_lf_DOE is not None and model.models[0].y_hf_DOE is not None
        assert model.models[1].y_lf_DOE is not None and model.models[1].y_hf_DOE is not None
        y_lf = np.column_stack([model.models[0].y_lf_DOE, model.models[1].y_lf_DOE])
        y_hf = np.column_stack([model.models[0].y_hf_DOE, model.models[1].y_hf_DOE])
        np.savetxt(os.path.join(outdir, "lf_fitnesses.txt"), y_lf)
        np.savetxt(os.path.join(outdir, "hf_fitnesses.txt"), y_hf)
    else:
        raise Exception(f"{type(model)} is currently not supported")


def main():
    """
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/lconfig.json")
    parser.add_argument("-p", "--pod", action="store_true", help="perform POD data reduction")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()

    # intit problem
    sm_config, _, _ = check_config(args.config, optim=True)
    seed = sm_config["optim"].get("seed", 123)
    if args.pod:
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
    outdir = sm_config["study"]["outdir"]

    # get model
    print("MFSM: model selection..")
    model_name = sm_config["optim"]["model_name"]
    model = get_mo_model(model_name, n_design, sm_config, outdir, seed)

    # get sampler and lf / hf DOEs
    print("MFSM: sampler selection..")
    # FFD sampling
    mf_sampler = get_sampler(
        sm_config["optim"]["n_design"], sm_config["optim"]["bound"], seed, model.requires_nested_doe
    )
    x_lf, x_hf = mf_sampler.sample_mf(sm_config["optim"]["n_lf"], sm_config["optim"]["n_hf"])
    # if POD the reduced parameters corresponding to the FFD are reconstructed
    if args.pod:
        ffd_lf_doe_profiles = np.stack(
            [p for p in [ffd_pod.ffd.apply_ffd(delta)[:, 1] for delta in x_lf]], axis=1
        )
        ffd_hf_doe_profiles = np.stack(
            [p for p in [ffd_pod.ffd.apply_ffd(delta)[:, 1] for delta in x_hf]], axis=1
        )
        xi = np.matmul(ffd_pod.phi_tilde.transpose(), ffd_pod.phi_tilde)
        xi_inv = np.linalg.inv(xi)
        x_lf = (
            np.matmul(xi_inv, np.matmul(ffd_pod.phi_tilde.transpose(), ffd_lf_doe_profiles))
            - np.matmul(xi_inv, np.matmul(ffd_pod.phi_tilde.transpose(), ffd_pod.S_mean[:, None]))
        ).transpose()
        x_hf = (
            np.matmul(xi_inv, np.matmul(ffd_pod.phi_tilde.transpose(), ffd_hf_doe_profiles))
            - np.matmul(xi_inv, np.matmul(ffd_pod.phi_tilde.transpose(), ffd_pod.S_mean[:, None]))
        ).transpose()

    # generate lf_DOE
    print("MFSM: LF DOE computation..")
    lf_dir = os.path.join(outdir, "lf_doe")
    lf_dict = execute_single_gen(
        X=x_lf,
        config=sm_config["optim"]["lf_config"],
        outdir=lf_dir,
        name="lf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: LF DOE computation finished after {time.time() - t0} seconds")
    # generate hf_DOE
    print("MFSM: HF DOE computation..")
    hf_dir = os.path.join(outdir, "hf_doe")
    hf_dict = execute_single_gen(
        X=x_hf,
        config=sm_config["optim"]["hf_config"],
        outdir=hf_dir,
        name="hf_doe",
        n_design=sm_config["optim"]["n_design"]
    )
    print(f"MFSM: HF DOE computation finished after {time.time() - t0} seconds")

    # compute DOEs
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
    # addition of lf baseline results
    x_lf = np.row_stack([x_lf, np.zeros(x_lf.shape[-1])])
    lf_w_ADP = np.append(lf_w_ADP, sm_config["optim"]["bsl_lf_w_ADP"])
    lf_w_OP = np.append(lf_w_OP, sm_config["optim"]["bsl_lf_w_OP"])
    # addition of hf baseline results
    x_hf = np.row_stack([x_hf, np.zeros(x_hf.shape[-1])])
    hf_w_ADP = np.append(hf_w_ADP, sm_config["optim"]["bsl_hf_w_ADP"])
    hf_w_OP = np.append(hf_w_OP, sm_config["optim"]["bsl_hf_w_OP"])

    # training
    print("MFSM: training model(s)..")
    set_mo_DOE(model, x_lf=x_lf, y_lf=[lf_w_ADP, lf_w_OP], x_hf=x_hf, y_hf=[hf_w_ADP, hf_w_OP])
    model.train()
    print(f"MFSM: finished training after {time.time() - t0} seconds")
    # saves mf-sm
    with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
        pickle.dump(model, handle)
    print(f"MFSM: models saved to {outdir}")

    # MFSM based optimization with adaptive infill
    print("MFSM: surrogate model based optimization..")
    nite = sm_config["optim"]["infill_nb"]
    infill_size = sm_config["optim"]["infill_lf_size"]
    infill_nb_gen = sm_config["optim"]["infill_nb_gen"]
    bayesian_infill = sm_config["optim"]["bayesian_infill"]
    for ite in range(nite):
        outdir_ite = os.path.join(outdir, outdir + f"_{ite}")
        # optimization
        if not bayesian_infill:
            print("MFSM: executes aero-optim subprocess")
            exec_cmd = [
                "optim",
                "-c", f"{args.config}", "-o", f"{outdir_ite}", "-v", f"{args.verbose}",
                "--pymoo"
            ]
            subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
            # read pareto
            cand = np.loadtxt(os.path.join(outdir_ite, "candidates.txt"))
            fit = np.loadtxt(os.path.join(outdir_ite, "fitnesses.txt"))
            pareto_fit = compute_pareto(fit[:, 0], fit[:, 1])
            pareto_idx = [np.where(fit == p)[0][0] for p in pareto_fit]
            pareto_cand = cand[pareto_idx]
            # compute non-Bayesian infill
            x_lf_infill = compute_non_bayesian_infill(
                model,
                pareto_cand,
                pareto_fit,
                infill_size,
                infill_nb_gen,
                n_design,
                bound,
                seed
            )
            # deactivate low fidelity network pre-training
            model.model_dict["pretraining"] = False
        else:
            # compute Bayesian infill
            x_lf_infill = compute_bayesian_infill(
                model, infill_size, infill_nb_gen, n_design, bound, seed
            )
        # execute infill
        y_lf_infill = execute_infill(
            x_lf_infill,
            config=sm_config["optim"]["lf_config"],
            n_design=sm_config["optim"]["n_design"],
            outdir=outdir_ite,
            ite=ite,
            fidelity="low"
        )
        # hf infill
        x_hf_infill = x_lf_infill[0]
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
        set_mo_DOE(model, x_lf_infill, y_lf_infill, x_hf_infill, y_hf_infill)
        model.train()
        # save updated model
        mv_filelist([os.path.join(outdir, "model.pkl")],
                    [os.path.join(outdir, f"model_{ite}.pkl")])
        with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
            pickle.dump(model, handle)
        print(f"MFSM: surrogate model based optimization {ite + 1}/{nite}"
              f" finished after {time.time() - t0} seconds")
    # save datasets and final model
    save_results(model, outdir)


if __name__ == '__main__':
    main()
