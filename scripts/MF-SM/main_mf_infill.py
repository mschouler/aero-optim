import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time

from typing import Callable

from mf_functions import zdt1_hf, zdt2_hf, zdt1_lf, zdt2_lf, SimpleProblem

from aero_optim.mf_sm.mf_models import get_model, get_sampler, MfDNN, MultiObjectiveModel
from aero_optim.mf_sm.mf_infill import (compute_pareto, MPI_acquisition_function, maximize_ED,
                                        minimize_LCB, maximize_MPI_BO, maximize_RegCrit)
from aero_optim.utils import rm_filelist

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.termination import get_termination

from scipy.stats import qmc


def compute_bayesian_infill(
        model: MfDNN | MultiObjectiveModel,
        infill_lf_size: int,
        infill_nb_gen: int,
        infill_regularization: bool,
        n_design: int,
        bound: list,
        seed: int
) -> np.ndarray:
    """
    Computes the low fidelity Bayesian infill candidates.
    """
    assert isinstance(model, MultiObjectiveModel)
    # Probability of Improvement
    if infill_regularization:
        infill_lf = maximize_RegCrit(
            MPI_acquisition_function, model, n_design, bound, seed, infill_nb_gen
        )
    else:
        infill_lf = maximize_MPI_BO(model, n_design, bound, seed, infill_nb_gen)
    # Lower Confidence Bound /objective 1
    infill_lf_LCB_1 = minimize_LCB(model.models[0], n_design, bound, seed, infill_nb_gen)
    infill_lf = np.vstack((infill_lf, infill_lf_LCB_1))
    # Lower Confidence Bound /objective 2
    infill_lf_LCB_2 = minimize_LCB(model.models[1], n_design, bound, seed, infill_nb_gen)
    infill_lf = np.vstack((infill_lf, infill_lf_LCB_2))
    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    current_DOE = np.vstack((current_DOE, infill_lf))
    for _ in range(infill_lf_size - 3):
        infill_lf_ED = maximize_ED(current_DOE, n_design, bound, seed, infill_nb_gen)
        infill_lf = np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
    return infill_lf


def compute_non_bayesian_infill(
    model: MfDNN | MultiObjectiveModel,
    pareto_cand: np.ndarray,
    pareto_fit: np.ndarray,
    infill_lf_size: int,
    infill_nb_gen: int,
    n_design: int,
    bound: list,
    seed: int
) -> np.ndarray:
    """
    Computes the low fidelity non-Bayesian infill candidates.
    """
    if len(pareto_fit) == 1:
        infill_lf = pareto_cand[0]
        n_p = 1
    elif len(pareto_fit) == 2:
        infill_lf = pareto_cand[:2]
        n_p = 2
    else:
        # matrix made of the distance between each point in the ordered pareto front
        d_matrix = np.linalg.norm(pareto_fit[:, np.newaxis] - pareto_fit[np.newaxis, :], axis=-1)
        # 1d array with the distance between two consecutive points along the ordered pareto front
        s = np.array([d_matrix[i + 1, i] for i in range(len(d_matrix) - 1)])
        s_length = np.sum(s)
        # index of the closest point to the pareto set center
        idx = np.argmin([abs(np.sum(s[:i]) - s_length / 2.) for i in range(len(s))])
        infill_lf = pareto_cand[idx]
        # best candidate wrt 1st objective
        infill_lf = np.vstack((infill_lf, pareto_cand[0]))
        # best candidate wrt 2nd objective
        infill_lf = np.vstack((infill_lf, pareto_cand[-1]))
        n_p = 3
    # max-min Euclidean Distance
    current_DOE = model.get_DOE()
    for _ in range(infill_lf_size - n_p):
        infill_lf_ED = maximize_ED(current_DOE, n_design, bound, seed, infill_nb_gen)
        infill_lf = np.vstack((infill_lf, infill_lf_ED))
        current_DOE = np.vstack((current_DOE, infill_lf_ED))
    return infill_lf


def bayesian_optimization(
        model: MultiObjectiveModel,
        func_lf: Callable,
        func_hf: Callable,
        n_iter: int,
        infill_lf_size: int,
        infill_nb_gen: int,
        infill_regularization: bool,
        dim: int,
        bound: list,
        seed: int
):
    """
    Runs a Bayesian optimization with adaptive infill.
    """
    for _ in range(n_iter):
        x_lf_infill = compute_bayesian_infill(
            model, infill_lf_size, infill_nb_gen, infill_regularization, dim, bound, seed
        )
        y_lf_infill = func_lf(x_lf_infill)
        x_hf_infill = x_lf_infill[0]
        y_hf_infill = func_hf(x_hf_infill.reshape(1, -1))
        print(f"iter {_}, new x_f {x_hf_infill}, new y_hf {y_hf_infill}")
        model.set_DOE(x_lf=x_lf_infill, y_lf=[y_lf_infill[:, 0], y_lf_infill[:, 1]],
                      x_hf=x_hf_infill, y_hf=[y_hf_infill[:, 0], y_hf_infill[:, 1]])
        model.train()
        print("model retrained")


def run_NSGA2(
        function: Callable, dim: int, pop_size: int, nb_gen: int, bound: list, seed: int
) -> SimpleProblem:
    """
    Runs NSGA2 with a any evaluation function.
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    initial_doe = sampler.random(n=pop_size).tolist()
    algorithm = NSGA2(pop_size=pop_size, sampling=qmc.scale(initial_doe, *bound))
    mfdnn_problem = SimpleProblem(function, dim, bound)
    minimize(
        problem=mfdnn_problem,
        algorithm=algorithm,
        termination=get_termination("n_gen", nb_gen),
        seed=seed,
        verbose=False
    )
    return mfdnn_problem


def non_bayesian_optimization(
    model: MfDNN,
    func_lf: Callable,
    func_hf: Callable,
    n_iter: int,
    infill_lf_size: int,
    infill_nb_gen: int,
    infill_pop_size: int,
    dim: int,
    bound: list,
    seed: int
):
    """
    Runs a non-Bayesian optimization with adaptive infill.
    """
    for _ in range(n_iter):
        # run NSGA-II with MfDNN
        mfdnn_problem = run_NSGA2(model.evaluate, dim, infill_pop_size, infill_nb_gen, bound, seed)
        # extract NSGA2 Pareto candidates and fitnesses
        cand = mfdnn_problem.candidates
        fit = mfdnn_problem.fitnesses
        pareto_fit = compute_pareto(fit[:, 0], fit[:, 1])
        pareto_idx = [np.where(fit == p)[0][0] for p in pareto_fit]
        pareto_cand = cand[pareto_idx]
        x_lf_infill = compute_non_bayesian_infill(
            model, pareto_cand, pareto_fit, infill_lf_size, infill_nb_gen, dim, bound, seed
        )
        y_lf_infill = func_lf(x_lf_infill)
        x_hf_infill = x_lf_infill[0]
        y_hf_infill = func_hf(x_hf_infill.reshape(1, -1))
        print(f"iter {_}, new x_f {x_hf_infill}, new y_hf {y_hf_infill}")
        # update and retrain model
        model.set_DOE(x_lf=x_lf_infill, y_lf=y_lf_infill, x_hf=x_hf_infill, y_hf=y_hf_infill)
        model.train()
        print("model retrained")


def compute_metrics(
        model: MfDNN | MultiObjectiveModel, igd: IGD, igdp: IGDPlus
) -> tuple[float, float]:
    """
    Computes IGD and IGD+.
    """
    if isinstance(model, MfDNN):
        assert model.y_hf_DOE is not None
        model_pareto = compute_pareto(model.y_hf_DOE[:, 0], model.y_hf_DOE[:, 1])
        return igd(model_pareto), igdp(model_pareto)
    else:
        assert model.models[0].y_hf_DOE is not None
        assert model.models[1].y_hf_DOE is not None
        model_pareto = compute_pareto(model.models[0].y_hf_DOE, model.models[1].y_hf_DOE)
        return igd(model_pareto), igdp(model_pareto)


def plot_results(
        ref_pareto: np.ndarray,
        pred_pareto: np.ndarray,
        model: MfDNN | MultiObjectiveModel,
        n_hf: int,
        fig_name: str
):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "figure.dpi": 300,
        "font.size": 8,
        'legend.fontsize': 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8
    })
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(ref_pareto[:, 0], ref_pareto[:, 1], color="r", label="reference Pareto", zorder=-1)
    ax.scatter(
        pred_pareto[:, 0], pred_pareto[:, 1],
        marker="x", color="b", label="predicted Pareto"
    )
    if isinstance(model, MfDNN):
        assert model.y_hf_DOE is not None
        ax.scatter(
            model.y_hf_DOE[:n_hf, 0], model.y_hf_DOE[:n_hf, 1],
            marker="s", color="k", label="initial DOE"
        )
        ax.scatter(
            model.y_hf_DOE[n_hf:, 0], model.y_hf_DOE[n_hf:, 1],
            marker="s", color="k", facecolors="None", label=f"{model.name} hf infills"
        )
    else:
        assert model.models[0].y_hf_DOE is not None
        assert model.models[1].y_hf_DOE is not None
        ax.scatter(
            model.models[0].y_hf_DOE[:n_hf], model.models[1].y_hf_DOE[:n_hf],
            marker="s", color="k", label="initial DOE"
        )
        ax.scatter(
            model.models[0].y_hf_DOE[n_hf:], model.models[1].y_hf_DOE[n_hf:],
            marker="s", color="k", facecolors="None", label=f"{model.name} hf infills"
        )
    ax.set(xlabel='$J_1$', ylabel='$J_2$')
    plt.legend()
    plt.savefig(fig_name, bbox_inches="tight")
    plt.close()


def save_results(
        time_list: list[float],
        igd_list: list[float],
        igdp_list: list[float],
        outdir: str
):
    """
    Saves test case results and statistics.
    """
    # train_time
    mean_time = np.mean(time_list)
    std_time = np.std(time_list)
    # igd
    mean_igd = np.mean(igd_list)
    std_igd = np.std(igd_list)
    # igdp
    mean_igdp = np.mean(igdp_list)
    std_igdp = np.std(igdp_list)

    res_dict = {"time": time_list, "igd": igd_list, "igdp": igdp_list}
    df = pd.DataFrame(res_dict)
    df.to_csv(outdir + "_stat.csv")
    print(f"wrote results to {outdir + '_stat.csv'}")

    with open(outdir + "_stat.txt", 'w') as file:
        file.write(f"mean (std) - time: {mean_time} ({std_time})\n")
        file.write(f"mean (std) - igd: {mean_igd} ({std_igd})\n")
        file.write(f"mean (std) - igdp: {mean_igdp} ({std_igdp})\n")
    print(f"wrote results to {outdir + '_stat.txt'}\n")


def main():
    """
    This program evaluates multi-fidelity surrogate models with Brevault's functions.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="infill parameters and model config file")
    parser.add_argument("-o", "--out", type=str, help="output directory")
    parser.add_argument(
        "-m", "--model", type=str,
        choices=["smt", "mfsmt", "mfkpls", "mflgp", "mfdnn"],
        help="surrogate model: mfkg or mfdnn"
    )
    parser.add_argument(
        "-f", "--function", type=str,
        choices=["zdt1", "zdt2"],
        help="optimization problem: ZDT1 or ZDT2"
    )
    parser.add_argument("-n", "--nite", type=int, help="number of training with varying seed")
    args = parser.parse_args()

    t0 = time.time()

    # open config
    assert os.path.isfile(args.config)
    with open(args.config) as jfile:
        config_dict = json.load(jfile)
    # create outdir
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    # save config
    shutil.copy(args.config, args.out)

    # parse config
    if args.function.lower() == "zdt1":
        print("running evaluation script for ZDT1 optimization problem\n")
        func_lf = zdt1_lf
        func_hf = zdt1_hf
        infill_config = config_dict["zdt1"]
        problem = get_problem("zdt1")
    elif args.function.lower() == "zdt2":
        print("running evaluation script for ZDT2 optimization problem\n")
        func_lf = zdt2_lf
        func_hf = zdt2_hf
        infill_config = config_dict["zdt2"]
        problem = get_problem("zdt2")
    else:
        raise Exception(f"incorrect function specification: {args.function}")

    # infill parameters
    bound = infill_config["bound"]
    dim = infill_config["dim"]
    n_lf = infill_config["n_lf"]
    n_hf = infill_config["n_hf"]
    infill_nb = infill_config["infill_nb"]
    infill_lf_size = infill_config["infill_lf_size"]
    infill_nb_gen = infill_config["infill_nb_gen"]
    infill_pop_size = infill_config["infill_pop_size"]

    # compute reference Pareto and metrics
    ref_pareto = problem.pareto_front()
    igd = IGD(ref_pareto)
    igdp = IGDPlus(ref_pareto)

    # create output directories
    fig_path = os.path.join(args.out, "Figs")
    stat_path = os.path.join(args.out, "Stats")
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(stat_path, exist_ok=True)

    igd_list = []
    igdp_list = []
    time_list = []
    for ite in range(args.nite):
        print(f"iteration: {ite + 1} / {args.nite}..\n")
        t_ite = time.time()

        # model building
        model = get_model(args.model, dim, config_dict, args.out, seed=ite)
        if args.model == "mfdnn":
            model = get_model(args.model, dim, config_dict, args.out, ite)
        else:
            model1 = get_model(args.model, dim, config_dict, args.out, ite)
            model2 = get_model(args.model, dim, config_dict, args.out, ite)
            model = MultiObjectiveModel([model1, model2])

        # data generation
        mf_sampler = get_sampler(
            dim=dim,
            bounds=bound,
            seed=ite,
            nested_doe=model.requires_nested_doe
        )
        x_lf, x_hf = mf_sampler.sample_mf(n_lf, n_hf)
        # low fidelity dataset
        y_lf = (
            func_lf(x_lf) if isinstance(model, MfDNN) else [c_ for c_ in func_hf(x_lf).transpose()]
        )
        # high fidelity dataset
        y_hf = (
            func_hf(x_hf) if isinstance(model, MfDNN) else [c_ for c_ in func_hf(x_hf).transpose()]
        )

        # model training
        model.set_DOE(x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf)
        model.train()

        # optimization loop
        if isinstance(model, MfDNN):
            non_bayesian_optimization(
                model, func_lf, func_hf,
                infill_nb, infill_lf_size,
                infill_nb_gen, infill_pop_size,
                dim, bound, ite
            )
        elif isinstance(model, MultiObjectiveModel):
            infill_regularization = infill_config["infill_regularization"]
            bayesian_optimization(
                model, func_lf, func_hf,
                infill_nb, infill_lf_size,
                infill_nb_gen,
                infill_regularization,
                dim, bound, ite
            )

        # post-processing NSGA-II
        model_problem = run_NSGA2(model.evaluate, dim, infill_pop_size, infill_nb_gen, bound, ite)
        pred_pareto = compute_pareto(model_problem.fitnesses[:, 0], model_problem.fitnesses[:, 1])

        # metric computation
        model_igd, model_igdp = compute_metrics(model, igd, igdp)
        igd_list.append(model_igd)
        igdp_list.append(model_igdp)
        time_list.append(time.time() - t0)
        print(f"\nIGD: {model_igd}, IGD+: {model_igdp}\n")

        # results saving
        ite_id = f"ite_{ite}"
        fig_name = os.path.join(fig_path, f"{args.model}_{ite_id}.pdf")
        plot_results(ref_pareto, pred_pareto, model, n_hf, fig_name)

        if isinstance(model, MfDNN):
            print("remove saved models")
            rm_filelist(
                [os.path.join(args.out, "best_NNL.pth"), os.path.join(args.out, "best_NNH.pth")]
            )

        print(f"iteration: {ite + 1} / {args.nite} finished in {time.time() - t_ite} seconds\n")

    # save test case results
    stat_name = os.path.join(stat_path, f"{args.model}")
    print(f"saving results to {stat_path}..\n")
    save_results(time_list, igd_list, igdp_list, stat_name)

    print(f"script executed in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
