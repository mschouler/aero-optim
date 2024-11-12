import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time

import mf_functions

from mpl_toolkits.mplot3d import Axes3D
from aero_optim.mf_sm.mf_models import get_model, get_sampler, MfDNN
from aero_optim.utils import rm_filelist


def get_iterator(f_config: dict) -> itertools.product:
    """
    Creates an iterator with all parameters and data size combinations.
    """
    param = f_config["parameter"]
    n_lf = f_config["n_lf"]
    n_hf = f_config["n_hf"]
    n_tuple = [(nhh, nll) for nhh, nll in zip(n_hf, n_lf)]
    return itertools.product(param, n_tuple)


def plot_1d_results(
        x_lf: np.ndarray, y_lf: np.ndarray,
        x_hf: np.ndarray, y_hf: np.ndarray,
        x: np.ndarray, Y_lf: np.ndarray,
        Y_hf: np.ndarray, y_pred: np.ndarray,
        model: str,
        fig_name: str
):
    fig, ax = plt.subplots(dpi=600)
    ax.plot(x, Y_hf, label='$y_{hf}$', color='black')
    ax.plot(x, Y_lf, label='$y_{lf}$', color='black', linestyle='dashed')
    ax.scatter(
        x_hf, y_hf, marker='x', color="red", label='hf DOE'
    )
    ax.scatter(
        x_lf, y_lf, marker='o', facecolors="none", edgecolors="blue", label='lf DOE'
    )
    ax.plot(x, y_pred, label='model', linestyle="dashed", color='darkviolet')
    ax.set(xlabel='$x$', ylabel='$y$')
    ax.legend(loc="upper right")
    fig.suptitle(f"{model}", y=0.93, size="x-large")
    plt.savefig(f"{fig_name}_1D.png")
    plt.close(fig)


def plot_2d_results(
        x_lf: np.ndarray, y_lf: np.ndarray,
        x_hf: np.ndarray, y_hf: np.ndarray,
        x_grid: np.ndarray, y_grid: np.ndarray,
        Z_lf: np.ndarray,
        Z_hf: np.ndarray, y_pred: np.ndarray,
        model: str,
        fig_name: str
):
    ax: Axes3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=600)
    s1 = ax.plot_surface(x_grid, y_grid, Z_lf, label="$y_{lf}$", alpha=0.3)
    s2 = ax.plot_surface(x_grid, y_grid, Z_hf, label="$y_{hf}$", alpha=0.5)
    s3 = ax.plot_surface(
        x_grid, y_grid, Z_hf - y_pred, label="$y_{hf} - y_\\text{pred}$", alpha=1
    )
    for surf in [s1, s2, s3]:
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
    ax.view_init(elev=30, azim=120)
    ax.scatter(x_lf[:, 0], x_lf[:, 1], y_lf[:, 0], label="lf DOE", alpha=1, zorder=10)
    ax.scatter(x_hf[:, 0], x_hf[:, 1], y_hf[:, 0], label="hf DOE", alpha=1, zorder=11)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$f(x_1, x_2)$")
    ax.legend(loc="upper right")
    fig.suptitle(f"{model}", y=0.93, size="x-large")
    plt.savefig(f"{fig_name}_2D.png")
    plt.close(fig)


def plot_results(y_val: np.ndarray, y_val_pred: np.ndarray, model: str, fig_name: str):
    fig, ax = plt.subplots(dpi=600)
    x = np.linspace(min(y_val), max(y_val), 100)
    ax.scatter(y_val, y_val_pred, s=5)
    ax.plot(x, x, linestyle="dashed", color="red")
    ax.set(xlabel='$y_\\text{true}$', ylabel='$y_\\text{pred}$')
    fig.suptitle(f"{model}", y=0.93, size="x-large")
    plt.savefig(f"{fig_name}.png")
    plt.close(fig)


def save_results(
        t_train_list: list[float],
        rmse_list: list[float],
        r2_list: list[float],
        param: int,
        n_hf: int,
        n_lf: int,
        outdir: str
):
    """
    Saves test case results and statistics.
    """
    # training time
    mean_train_time = np.mean(t_train_list)
    std_train_time = np.std(t_train_list)
    # rmse
    mean_rmse = np.mean(rmse_list)
    std_rmse = np.std(rmse_list)
    # r2
    mean_r2 = np.mean(r2_list)
    std_r2 = np.std(r2_list)

    res_dict = {"train time": t_train_list, "rmse": rmse_list, "r2": r2_list}
    df = pd.DataFrame(res_dict)
    df.to_csv(outdir + "_stat.csv")
    print(f"wrote results to {outdir + '_stat.csv'}")

    with open(outdir + "_stat.txt", 'w') as file:
        file.write(f"mean (std) - training time: {mean_train_time} ({std_train_time})\n")
        file.write(f"mean (std) - rmse: {mean_rmse} ({std_rmse})\n")
        file.write(f"mean (std) - r2: {mean_r2} ({std_r2})\n")
    print(f"wrote results to {outdir + '_stat.txt'}\n")


def main():
    """
    This program evaluates multi-fidelity surrogate models with Brevault's functions.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="test functions and model config file")
    parser.add_argument("-o", "--out", type=str, help="output directory")
    parser.add_argument("-m", "--model", type=str, help="surrogate model: mfkg or mfdnn")
    parser.add_argument("-f", "--function", type=str, help="test function: 1d or nd")
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
    if args.function.lower() == "1d":
        print("running validation script for brevault_1d functions\n")
        f_config = config_dict["function 1d"]
        func_lf = mf_functions.brevault_f1d_lf
        func_hf = mf_functions.brevault_f1d_hf
        dim = 1
    elif args.function.lower() == "nd":
        print("running validation script for brevault_nd functions\n")
        f_config = config_dict["function nd"]
        func_lf = mf_functions.brevault_fnd_lf
        func_hf = mf_functions.brevault_fnd_hf
        dim = -1
    else:
        raise Exception(f"incorrect function specification: {args.function}")
    bounds = f_config["bounds"]
    n_val = f_config["n_val"]

    # loop over functions
    test_function_iterator = get_iterator(f_config)
    for param, (n_hf, n_lf) in test_function_iterator:
        print(f"test function with parameter: {param}, n_hf: {n_hf}, n_lf {n_lf}\n")
        # create output directories
        sub_outdir = f"param_{param}_nhf_{n_hf}_nlf_{n_lf}"
        fig_path = os.path.join(args.out, sub_outdir, "Figs")
        stat_path = os.path.join(args.out, sub_outdir, "Stats")
        os.makedirs(fig_path, exist_ok=True)
        os.makedirs(stat_path, exist_ok=True)

        rmse_list = []
        r2_list = []
        t_train_list = []
        for ite in range(args.nite):
            print(f"iteration: {ite + 1} / {args.nite}..\n")
            # model building
            dim = dim if dim > 0 else param
            model = get_model(args.model, dim, config_dict, args.out, seed=ite)

            # data generation
            mf_sampler = get_sampler(
                dim=dim,
                bounds=bounds,
                seed=ite,
                nested_doe=model.requires_nested_doe
            )
            x_lf, x_hf = mf_sampler.sample_mf(n_lf, n_hf)
            # low fidelity dataset
            y_lf = func_lf(x_lf)
            # high fidelity dataset
            y_hf = func_hf(x_hf, a=param) if dim == 1 else func_hf(x_hf)
            # validation dataset
            val_sampler = get_sampler(dim=dim, bounds=bounds, seed=ite + 1, nested_doe=False)
            x_val = val_sampler.sample(n_val)
            y_val_hf = func_hf(x_val, a=param) if dim == 1 else func_hf(x_val)

            # model training
            t0_train = time.time()
            model.set_DOE(x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf)
            model.train()
            t_train_list.append(time.time() - t0_train)

            # model evaluation
            y_val_pred = model.evaluate(x_val)

            # metric computation
            rmse = mf_functions.get_RMSE(y_val_hf.flatten(), y_val_pred.flatten())
            r2 = mf_functions.get_R2(y_val_hf.flatten(), y_val_pred.flatten())
            rmse_list.append(rmse)
            r2_list.append(r2)
            print(f"\nRMSE: {rmse}, R2: {r2}\n")

            # results saving
            ite_id = f"ite_{ite}"
            fig_name = os.path.join(fig_path, f"{args.model}_{ite_id}")
            if dim == 1:
                X = np.linspace(bounds[0], bounds[-1], 200).reshape(-1, 1)
                Y_lf = func_lf(X)
                Y_hf = func_hf(X, a=param)
                y_pred = model.evaluate(X)
                plot_1d_results(x_lf, y_lf, x_hf, y_hf, X, Y_lf, Y_hf, y_pred, model.name, fig_name)
            elif dim == 2:
                nx = ny = 100
                X = np.linspace(*bounds, nx)
                Y = np.linspace(*bounds, ny)
                x_grid, y_grid = np.meshgrid(X, Y)
                Z_hf = func_hf(
                    np.row_stack((x_grid.ravel(), y_grid.ravel())).transpose()
                ).reshape(nx, ny)
                Z_lf = func_lf(
                    np.row_stack((x_grid.ravel(), y_grid.ravel())).transpose()
                ).reshape(nx, ny)
                y_pred = model.evaluate(
                    np.column_stack((x_grid.ravel(), y_grid.ravel()))
                ).reshape(nx, ny)
                print(f"mesh grid RMSE: {mf_functions.get_RMSE(y_pred.flatten(), Z_hf.flatten())}")
                plot_2d_results(
                    x_lf, y_lf, x_hf, y_hf, x_grid, y_grid, Z_lf, Z_hf, y_pred, model.name, fig_name
                )
            plot_results(y_val_hf, y_val_pred, model.name, fig_name)
            if isinstance(model, MfDNN):
                print("remove saved models")
                rm_filelist(
                    [os.path.join(args.out, "best_NNL.pth"), os.path.join(args.out, "best_NNH.pth")]
                )

            print(f"iteration: {ite + 1} / {args.nite} finished in {time.time() - t0} seconds\n")

        # save test case results
        stat_name = os.path.join(stat_path, f"{args.model}")
        print(f"saving results to {stat_path}..\n")
        save_results(t_train_list, rmse_list, r2_list, param, n_hf, n_lf, stat_name)

    print(f"script executed in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
