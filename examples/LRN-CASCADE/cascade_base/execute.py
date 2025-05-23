import argparse
import functools
import math
import os
import shutil
import sys
import time

from subprocess import CalledProcessError
from aero_optim.utils import cp_filelist, run, sed_in_file, submit_popen_process, wait_for_it

FAILURE: int = 1
SUCCESS: int = 0

WOLF: str = "/home/mschouler/bin/wolf_sabcm"
METRIX: str = "/home/mschouler/bin/metrix"
FEFLO: str = "/home/mschouler/bin/fefloa_margaret"
INTERPOL: str = "/home/mschouler/bin/interpol"
SPYDER: str = "/home/mschouler/bin/spyder"

print = functools.partial(print, flush=True)


def get_residual(res_file: str = "residual.dat", entry: int = -2) -> float:
    res_line = open(res_file, "r").read().splitlines()[-1]
    res_list = list(map(float, res_line.split()))
    return float(res_list[entry])


def get_turbocoef(res_file: str = "turbocoef.dat") -> list[float]:
    # Iter (0) CPU (1)  DebitIn (2)  DebitOut (3)  DebitRat (4)  PreTotIn (5)  PreTotOut (6)
    # PreTotRat (7) TemTotIn (8) TemTotOut (9) TemTotRat (10) IsentropicEff (11) LossCoef (12)
    # HeatFlux (13) TempCoef (14)
    res_line = open(res_file, "r").read().splitlines()[-1]
    return list(map(float, res_line.split()))


def main() -> int:
    """
    This program runs a WOLF CFD simulation at ADP.
    In multisimulation mode, OP1 (+5°) and OP2 (-5°) simulations are also executed.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-in", "--input", type=str, help="input mesh file i.e. input.mesh")
    parser.add_argument("-nproc", type=int, help="number of procs", default=1)
    parser.add_argument("-ms", type=int, help="number of parallel simulations", default=-1)
    parser.add_argument("-adp", action="store_true", help="simulate ADP")
    parser.add_argument("-op1", action="store_true", help="simulate OP1")
    parser.add_argument("-op2", action="store_true", help="simulate OP2")

    args = parser.parse_args()
    t0 = time.time()
    print(f"simulations performed with: {args}\n")

    input = args.input.split(".")[0]
    wolf_cmd = [WOLF, "-in", f"{input}", "-out", f"{input}.o", "-nproc", f"{args.nproc}", "-v", "6"]

    if args.ms > 0:
        # tracking variables
        l_proc = []
        # process cmd line
        exec_args = sys.argv
        ms_idx = exec_args.index("-ms")
        del exec_args[ms_idx + 1]
        del exec_args[ms_idx]
        # adp
        exec_cmd = [sys.executable] + exec_args + ["-adp"]
        l_proc.append(submit_popen_process("ADP", exec_cmd))
        # op1
        exec_cmd = [sys.executable] + exec_args + ["-op1"]
        if len(l_proc) < args.ms:
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        else:
            wait_for_it(l_proc, args.ms)
            l_proc.append(submit_popen_process("OP1", exec_cmd))
        # op2
        exec_cmd = [sys.executable] + exec_args + ["-op2"]
        if len(l_proc) < args.ms:
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        else:
            wait_for_it(l_proc, args.ms)
            l_proc.append(submit_popen_process("OP2", exec_cmd))
        # wait for all processes to finish
        wait_for_it(l_proc, 1)

    if args.adp:
        print("** ADP SIMULATION **")
        print("** -------------- **")
        sim_dir = "ADP"
        shutil.rmtree(sim_dir, ignore_errors=True)
        os.mkdir(sim_dir)
        cp_filelist([f"{input}.wolf", f"{input}.mesh"], [sim_dir] * 2)
        os.chdir(sim_dir)
        try:
            run(wolf_cmd, "wolf.job")
            turbocoef = get_turbocoef()
            print(f">> residual : {get_residual()}")
            print(f">> debit ratio: {turbocoef[4]}")
            print(f">> total pressure ratio: {turbocoef[7]}")
            print(f">> total temperature ratio: {turbocoef[10]}")
            print(f">> isentropic efficiency: {turbocoef[11]}")
            print(f">> loss coefficient: {turbocoef[12]}\n")
        except CalledProcessError:
            print(f">> ADP failed after {time.time() - t0} seconds.")
            return FAILURE
        print(f">> ADP finished successfully in {time.time() - t0} seconds.\n")

    if args.op1:
        print("** OP1 SIMULATION (+5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP1"
        os.mkdir(sim_dir)
        cp_filelist([f"{input}.wolf", f"{input}.mesh"], [sim_dir] * 2)
        os.chdir(sim_dir)
        # update input velocity
        cos = math.cos((43 + 5) / 180 * math.pi)
        sin = math.sin((43 + 5) / 180 * math.pi)
        u = 199.5 * cos
        v = 199.5 * sin
        # update input file
        sim_args = {
            "PhysicalState": {"inplace": False, "param": [f"  0.1840 {u} {v} 0. 14408 1.7039e-5"]},
            "BCInletVelocityDirection": {"inplace": False, "param": [f"{cos} {sin} 0."]}
        }
        sed_in_file(f"{input}.wolf", sim_args)
        try:
            run(wolf_cmd, "wolf.job")
            turbocoef = get_turbocoef()
            print(f">> residual : {get_residual()}")
            print(f">> debit ratio: {turbocoef[4]}")
            print(f">> total pressure ratio: {turbocoef[7]}")
            print(f">> total temperature ratio: {turbocoef[10]}")
            print(f">> isentropic efficiency: {turbocoef[11]}")
            print(f">> loss coefficient: {turbocoef[12]}\n")
        except CalledProcessError:
            print(f">> OP1 failed after {time.time() - t0} seconds.")
            return FAILURE
        print(f">> OP1 finished successfully in {time.time() - t0} seconds.\n")

    if args.op2:
        print("** OP2 SIMULATION (-5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP2"
        os.mkdir(sim_dir)
        cp_filelist([f"{input}.wolf", f"{input}.mesh"], [sim_dir] * 2)
        os.chdir(sim_dir)
        # update input velocity
        cos = math.cos((43 - 5) / 180 * math.pi)
        sin = math.sin((43 - 5) / 180 * math.pi)
        u = 199.5 * cos
        v = 199.5 * sin
        # update input file
        sim_args = {
            "PhysicalState": {"inplace": False, "param": [f"  0.1840 {u} {v} 0. 14408 1.7039e-5"]},
            "BCInletVelocityDirection": {"inplace": False, "param": [f"{cos} {sin} 0."]}
        }
        sed_in_file(f"{input}.wolf", sim_args)
        try:
            run(wolf_cmd, "wolf.job")
            turbocoef = get_turbocoef()
            print(f">> residual : {get_residual()}")
            print(f">> debit ratio: {turbocoef[4]}")
            print(f">> total pressure ratio: {turbocoef[7]}")
            print(f">> total temperature ratio: {turbocoef[10]}")
            print(f">> isentropic efficiency: {turbocoef[11]}")
            print(f">> loss coefficient: {turbocoef[12]}\n")
        except CalledProcessError:
            print(f">> OP2 failed after {time.time() - t0} seconds.")
            return FAILURE

    print(f">> simulations finished successfully in {time.time() - t0} seconds.")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
