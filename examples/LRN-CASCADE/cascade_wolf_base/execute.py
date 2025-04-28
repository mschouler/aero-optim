import argparse
import itertools
import numpy as np
import os
import pandas as pd
import sys
import time

from aero_optim.utils import run

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cascade_base.execute import main as base_main # noqa
from cascade_base.execute import FAILURE, SUCCESS # noqa


GMF2VTK = "gmf2vtk"
PVPYTHON = "/home/mschouler/Documents/Sorbonne/ParaView-5.12.1-MPI-Linux-Python3.10-x86_64/bin/pvpython" # noqa
INTERPOL = "plot_over_line.py"

MPLOSS = "MPLossCoef.dat"
MIXEDOUTLOSS = "MixedoutLossCoef.dat"
OUTFLOW = "OutflowAngle.dat"


def compute_loss(input_data: pd.DataFrame, output_data: pd.DataFrame, outfile: str):
    """
    Computes the standard loss coefficient at the measurement planes and saves it to outfile.

    Note:
        - Vector2:0 = rho*u
        - Vector2:1 = rho*v
        - Scalar1_input_1 = P
        - Scalar2 = P0
    """
    nx = 0.5
    ny = 0.
    # MP1
    q1 = np.sum(input_data[" Vector2:0"] * nx + input_data[" Vector2:1"] * ny)
    P1 = np.sum(
        input_data[" Scalar1_input_1"]
        * (input_data[" Vector2:0"] * nx + input_data[" Vector2:1"] * ny)
    ) / q1
    P01 = np.sum(
        input_data[" Scalar2"] * (input_data[" Vector2:0"] * nx + input_data[" Vector2:1"] * ny)
    ) / q1
    # MP2
    q2 = np.sum(output_data[" Vector2:0"] * nx + output_data[" Vector2:1"] * ny)
    P02 = np.sum(
        output_data[" Scalar2"] * (output_data[" Vector2:0"] * nx + output_data[" Vector2:1"] * ny)
    ) / q2
    # Loss
    w = (P01 - P02) / (P01 - P1)
    # save to file
    with open(outfile, "w") as f:
        f.write("# MPLossCoef\n")
        f.write(str(w))
    print(f">> MPLossCoef: {w} written to {outfile}")


def compute_mixedout_qty(input_data: pd.DataFrame) -> tuple[float, float]:
    """
    Computes and returns the mixedout pressure and total pressure in the measurement planes:
    see A. Prasad (2004): https://doi.org/10.1115/1.1928289

    Note:
        - Vector2:0 = rho*u
        - Vector2:1 = rho*v
        - Scalar1 = rho
        - Scalar1_input_1 = P
        - Scalar2 = P0
    """
    # conservation of mass
    m_bar = np.nanmean(input_data[" Vector2:0"])
    rho_bar = np.nanmean(input_data[" Scalar1"])
    p_bar = np.nanmean(input_data[" Scalar1_input_1"])
    u_bar = m_bar / rho_bar
    v_bar = np.nanmean(input_data[" Vector2:1"]) / rho_bar
    uu_bar = u_bar**2
    vv_bar = v_bar**2

    # conservation of momentum
    x_mom = m_bar * u_bar + p_bar
    y_mom = m_bar * v_bar

    # conservation of energy
    gamma = 1.4
    R = 8.314
    E = m_bar * gamma / (gamma - 1) * p_bar / rho_bar + m_bar / 2. * (uu_bar + vv_bar)

    # quadratic equation
    Q = 1 / m_bar**2 * (1 - 2 * gamma / (gamma - 1))
    L = 2 / m_bar**2 * (gamma / (gamma - 1) * x_mom - x_mom)
    C = 1 / m_bar**2 * (x_mom**2 + y_mom**2) - 2 * E / m_bar

    # select subsonic root
    p_bar = (-L - np.sqrt(L**2 - 4 * Q * C)) / 2 / Q
    T_bar = p_bar / rho_bar / R
    T0_bar = (gamma - 1) / (gamma * R) * E / m_bar
    p0_bar = p_bar * (T0_bar / T_bar)**(gamma / (gamma - 1))
    return p_bar, p0_bar


def compute_mixedout_loss(input_data: pd.DataFrame, output_data: pd.DataFrame, outfile: str):
    """
    Computes the mixedout loss coefficient at the measurement planes and saves it to outfile.

    Note:
        - Vector2:0 = rho*u
        - Vector2:1 = rho*v
        - Scalar1_input_1 = P
        - Scalar2 = P0
    """
    P1, P01 = compute_mixedout_qty(input_data)
    _, P02 = compute_mixedout_qty(output_data)
    # Loss
    w = (P01 - P02) / (P01 - P1)
    # save to file
    with open(outfile, "w") as f:
        f.write("# MixedoutLossCoef\n")
        f.write(str(w))
    print(f">> MixedoutLossCoef: {w} written to {outfile}")


def compute_outflow_angle(output_data: pd.DataFrame, outfile: str):
    """
    Computes the outflow angle at the measurement planes and saves it to outfile.

    Note:
        - Vector2:0 = rho*u
        - Vector2:1 = rho*v
    """
    u_mean = np.nanmean(output_data[" Vector2:0"])
    v_mean = np.nanmean(output_data[" Vector2:1"])
    outflow_angle = np.arctan(v_mean / u_mean) / np.pi * 180
    # save to file
    with open(outfile, "w") as f:
        f.write("# OutflowAngle\n")
        f.write(str(outflow_angle))
    print(f">> outflow angle: {outflow_angle} written to {outfile}")


def post_process(input_mesh: str, simdir: str):
    """
    Performs the following extra processing step:
    1. calls gmf2vtk to convert solution from .solb to .vtu
    2. calls pvpython to interpolate data at the measurement planes
    3. compute and save the standard and mixedout loss coefficients and the outflow angle
       in MFLossCoef.dat, LossCoef.dat and OutflowAngle.dat

    Note: the execution commands assume that both gmf2vtk and pvppython are
          installed and in the PATH.
    """
    input = input_mesh.split(".")[0]
    # pres
    gmf_cmd = [GMF2VTK, "-in", input_mesh, "-sol", "pres.solb", "-out", "pres.vtu"]
    run(gmf_cmd, "gmf2vtk_pres.out")
    # o.sol
    gmf_cmd = [GMF2VTK, "-in", input_mesh, "-sol", f"{input}.o.solb", "-out", "final.vtu"]
    run(gmf_cmd, "gmf2vtk_final.out")
    # pvpython
    pvpython_cmd = [PVPYTHON, INTERPOL, "--mplan"]
    run(pvpython_cmd, "paraview.out")
    # compute/save files
    input_data = pd.read_csv("plotMP1.csv")
    output_data = pd.read_csv("plotMP2.csv")
    compute_loss(input_data, output_data, MPLOSS)
    compute_mixedout_loss(input_data, output_data, MIXEDOUTLOSS)
    compute_outflow_angle(output_data, OUTFLOW)


def check_files(list_of_file: list[str], timeout: float, ts: float = 1.) -> int:
    """
    Waits for the files in list_of_file to be found at most until timeout.
    """
    tt = 0.
    while sum([os.path.isfile(file) for file in list_of_file]) < len(list_of_file):
        time.sleep(ts)
        tt += ts
        if tt > timeout:
            print("ERROR -- all files not found before timeout")
            return FAILURE
    return SUCCESS


def main() -> int:
    """
    Wrapper around cascade_base.main with an extra post-processing step
    to compute the loss coefficient at the measurement planes and the outflow angle.
    Both quantities are respectively saved to LossCoeff.dat and OutflowAngle.dat.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input", type=str, help="input mesh file i.e. input.mesh")
    parser.add_argument("-ms", type=int, help="number of parallel simulations", default=-1)
    parser.add_argument("-adp", action="store_true", help="simulate ADP")
    parser.add_argument("-op1", action="store_true", help="simulate OP1")
    parser.add_argument("-op2", action="store_true", help="simulate OP2")

    # Parse only what this script needs; ignore the rest
    wrapper_args, _ = parser.parse_known_args()
    t0 = time.time()

    # get the script arguments and pass it to the original main
    # Note: calling base_main with adp, op1 or op2 will change
    #       the cwd to the simulation directory
    status = base_main()
    if status == FAILURE:
        return FAILURE

    if wrapper_args.ms > 0:
        files = [os.path.join(simdir, file)
                 for simdir, file
                 in list(itertools.product(["ADP", "OP1", "OP2"], [MPLOSS, MIXEDOUTLOSS, OUTFLOW]))]
        # wait for files to be produced
        print(">> wait for files to be produced..")
        return check_files(files, 100)
    if wrapper_args.adp:
        print(">> post-processing ADP..")
        post_process(wrapper_args.input, "ADP")
    if wrapper_args.op1:
        print(">> post-processing OP1..")
        post_process(wrapper_args.input, "OP1")
    if wrapper_args.op2:
        print(">> post-processing OP2..")
        post_process(wrapper_args.input, "OP2")
    print(f">> program finished successfully in {time.time() - t0:.2f} seconds")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
