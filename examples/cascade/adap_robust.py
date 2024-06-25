import argparse
import functools
import glob
import os
import shutil
import subprocess
import sys
import time

FAILURE: int = 1
SUCCESS: int = 0

WOLF: str = "/home/mschouler/bin/wolf"
METRIX: str = "/home/mschouler/bin/metrix2"
FEFLO: str = "/home/mschouler/bin/fefloa_margaret"
INTERPOL: str = "/home/mschouler/bin/interpol2"

resTgt: float = 1e-6
m_field: str = "mach"

print = functools.partial(print, flush=True)


def sed_in_file(fname: str, key: str, new_key: str):
    fcontent = open(fname, "r").read().splitlines()
    fcontent[fcontent.index(key)] = new_key
    with open(fname, 'w') as ftw:
        ftw.write("\n".join(fcontent))


def rm_filelist(deletion_list: list[str]):
    [os.remove(f) for f_pattern in deletion_list for f in glob.glob(f_pattern)]  # type: ignore


def cp_filelist(in_files: list[str], out_files: list[str]):
    for in_f, out_f in zip(in_files, out_files):
        try:
            shutil.copy(in_f, out_f)
        except FileNotFoundError:
            print(f"WARNING -- {in_f} not found")


def get_residual(res_file: str = "residual.dat", entry: int = -2) -> float:
    res_line = open(res_file, "r").read().splitlines()[-1]
    res_list = list(map(float, res_line.split()))
    return float(res_list[entry])


def get_aerocoef(res_file: str = "aerocoef.dat", entry: int = 2) -> float:
    res_line = open(res_file, "r").read().splitlines()[-1]
    res_list = list(map(float, res_line.split()))
    return float(res_list[entry])


def check_cv(new_coef: float, old_coef: float, target_delta: float) -> tuple[bool, float]:
    delta = abs(new_coef - old_coef) / new_coef * 100
    return delta < target_delta, delta


def main() -> int:
    """
    This program runs a CFD simulation with mesh adaptation i.e. coupling WOLF, METRIX and FEFLO.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-in", "--input", type=str, help="input mesh file i.e. input.mesh")
    parser.add_argument("-cmp", type=int, help="targetted complexity")
    parser.add_argument("-cmax", type=int, help="maximal complexity")
    parser.add_argument("-gro", type=float, help="complexity growth factor", default=1.5)
    parser.add_argument("-nproc", type=int, help="number of procs", default=1)
    parser.add_argument("-nite", type=int, help="number of adaptation iterations", default=5)
    parser.add_argument("-smax", type=int, help="max. number of adaptation perturbation", default=3)
    parser.add_argument("-delta", type=float, help="aerocoef convergence criteria", default=1)

    args = parser.parse_args()
    t0 = time.time()
    print(f"simulation performed with: {args}")

    # assert required input files exist
    input = args.input.split(".")[0]
    assert os.path.isfile(input + ".wolf")
    assert os.path.isfile(input + ".metrix")
    mesh = "meshb" if os.path.isfile(input + ".meshb") else "mesh"
    assert os.path.isfile(f"{input}.{mesh}")

    # Initialization
    # generate InitialSol if missing
    if not os.path.isfile(input + ".solb"):
        print("** INITIAL SOLUTION COMPUTATION WITH ~2000 ITERATIONS **")
        print("** -------------------------------------------------- **")
        wolf_cmd = [WOLF, "-in", f"{input}", "-out", f"{input}", "-nproc", f"{args.nproc}"]
        with open("wolf.job", "wb") as out:
            subprocess.run(wolf_cmd + ["-ite", "2000"], stdout=out, stderr=out, check=True)
        sed_in_file(f"{input}.wolf", "UniformSol", "InitialSol")
        print(f">> execution time: {time.time() - t0} seconds.\n")

    # setup working directory
    open("tmp.surf", 'a').close()
    cp_filelist(
        [f"{input}.wolf", f"{input}.solb", f"{input}.solb", f"{input}.{mesh}", f"{input}.metrix"],
        ["file.wolf", "file.solb", "file.o.solb", f"file.{mesh}", "adap.metrix"]
    )
    os.symlink(f"file.{mesh}", f"adap.{mesh}")
    # get Cd
    old_Cd = get_aerocoef()
    print(f">> initial Cd = {old_Cd}\n")
    # update Wolf cmd
    wolf_cmd = [WOLF, "-in", "file", "-out", "file.o", "-nproc", f"{args.nproc}", "-v", "6"]

    # adaptation loop
    gro = args.gro
    cmp = args.cmp
    ite, sub_ite = 1, 0
    aero_cv = False
    while ite <= args.nite:

        print(f"** ITERATION {ite} - COMPLEXITY {cmp} **")
        print(f"** ----------{'-' * len(str(ite))}--------------{'-' * len(str(cmp))} **")
        init_files = [f"file.{mesh}", "file.o.solb", "file.metric.solb", f"{m_field}.solb"]
        backup_files = [
            f"file.back.{mesh}", "file.back.solb", "file.metric.back.solb", f"{m_field}.back.solb"
        ]
        cp_filelist(init_files, backup_files)

        print("** METRIC CONSTRUCTION **")
        cp_filelist([f"{m_field}.solb"], ["adap.solb"])
        metrix_cmd = [METRIX, "-O", "1", "-in", "adap", "-out", "adap.met.solb", "-v", "6", "-Cmp",
                      f"{cmp}", "-Cmax", f"{args.cmax}", "-hmax", "0.005"]
        with open("metrix.job", "wb") as out:
            subprocess.run(metrix_cmd, stdout=out, stderr=out, check=True)

        print("** MESH ADAPTATION **")
        cp_filelist([f"file.{mesh}", "file.metric.solb"], [f"adap.met.{mesh}", "adap.met.solb"])
        feflo_cmd = [FEFLO, "-in", "adap.met", "-met", "adap.met", "-out", f"file.{mesh}",
                     "-noref", "-nordg", "-hgrad", "1.5", "-adap-line-ids", "1-9,13,21"]
        with open("fefloa.job", "wb") as out:
            subprocess.run(feflo_cmd, stdout=out, stderr=out, check=True)

        print("** SOLUTION INTERPOLATION **")
        rm_filelist(["file.solb"])
        interpol_cmd = [INTERPOL, "-O", "1", "-in", "file", "-back", "file.back"]
        with open("interpol.job", "wb") as out:
            subprocess.run(interpol_cmd, stdout=out, stderr=out, check=True)

        print("** SOLUTION COMPUTATION **")
        rm_filelist(["file.o.solb"])
        with open("wolf.job", "wb") as out:
            subprocess.run(wolf_cmd, stdout=out, stderr=out, check=True)
        rm_filelist(["localCfl.*.solb"])

        res = get_residual()
        if res < resTgt:
            print(f">> WOLF converged with residual {res}")
            cp_filelist(["file.o.solb", f"file.{mesh}", "residual.dat"],
                        [f"fin.{ite}.solb", f"fin.{ite}.{mesh}", f"fin.res.{ite}.dat"])
            print(f"fin.{ite}.{mesh} & fin.{ite}.solb CREATED")
            new_Cd = get_aerocoef()
            aero_cv, delta = check_cv(new_Cd, old_Cd, args.delta)
            print(f">> new Cd = {new_Cd} ({delta} %)")
            print(f">> aerodynamic convergence {'reached' if aero_cv else 'not reached'}")
            old_Cd = new_Cd

            cmp *= gro if ite % args.smax == 0 else 1.
            ite += 1
            sub_ite = 1
        else:
            print(f">> WOLF has not converged (residual {res})")
            sub_ite += 1
            cmp *= 1.01
            # restore from backup
            cp_filelist(backup_files, init_files)
            if sub_ite > args.smax:
                print(f">> WOLF failed to converge in {sub_ite - 1} sub iterations "
                      f"({time.time() -t0} seconds)")
                return FAILURE

        print(f">> execution time: {time.time() - t0} seconds.\n")

    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
