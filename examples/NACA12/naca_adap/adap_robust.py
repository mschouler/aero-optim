import argparse
import functools
import os
import sys
import time

from aero_optim.utils import cp_filelist, ln_filelist, rm_filelist, run, sed_in_file

FAILURE: int = 1
SUCCESS: int = 0

WOLF: str = "/home/mschouler/bin/wolf"
METRIX: str = "/home/mschouler/bin/metrix2"
FEFLO: str = "/home/mschouler/bin/fefloa_margaret"
INTERPOL: str = "/home/mschouler/bin/interpol2"

print = functools.partial(print, flush=True)


def get_residual(res_file: str = "residual.dat", entry: int = -2) -> float:
    res_line = open(res_file, "r").read().splitlines()[-1]
    res_list = list(map(float, res_line.split()))
    return float(res_list[entry])


def get_aerocoef(res_file: str = "aerocoef.dat", entry: int = 2) -> list[float]:
    # Iter (0) CPU (1)  CD (2)  CDp (3)  CDf (4)  ResCD (5)  CL (6)
    # CLp (7) CLf (8) ResCL (9) Slip (10) Slip_p (11) Roll (12)
    # Roll_p (13) Pitch (14) Pitch_p (15) Yaw (16) Yaw_p (17)
    res_line = open(res_file, "r").read().splitlines()[-1]
    return list(map(float, res_line.split()))


def check_cv(
        new_coef: float, old_coef_2: float, old_coef_1: float, target_delta: float
) -> tuple[bool, float, float]:
    delta_1 = abs(new_coef - old_coef_1) / abs(new_coef)
    delta_2 = abs(new_coef - old_coef_2) / abs(new_coef)
    return delta_1 < target_delta and delta_2 < target_delta, delta_2, delta_1


def main() -> int:
    """
    This program runs a CFD simulation with mesh adaptation i.e. coupling WOLF, METRIX and FEFLO.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-in", "--input", type=str, help="input mesh file i.e. input.mesh")
    parser.add_argument("-cmp", type=int, help="targetted complexity")
    parser.add_argument("-nproc", type=int, help="number of procs", default=1)
    parser.add_argument("-nite", type=int, help="number of complexities", default=2)
    parser.add_argument("-smax", type=int, help="max. number of adaptations at iso comp", default=3)
    parser.add_argument("-ntol", type=int, help="max. number of failures before abort", default=3)

    args = parser.parse_args()
    t0 = time.time()
    cwd = os.getcwd()
    print(f"simulation performed with: {args}\n")

    # assert required input files exist
    input = args.input.split(".")[0]
    assert os.path.isfile(input + ".wolf")
    assert os.path.isfile(input + ".metrix")
    assert os.path.isfile(f"{input}.mesh")

    # adaptation variables
    res_tgt: float = 1e-3
    m_field: str = "mach"
    cv_tgt_tab: list[float] = [0.01, 0.005, 0.001] + [0.001] * (args.nite - 3)
    tol_fail = args.ntol

    # Initialization
    print("** INITIAL SOLUTION COMPUTATION WITH 1000 ITERATIONS **")
    print("** ------------------------------------------------- **")
    os.makedirs("SolIni", exist_ok=True)
    cp_filelist([f"{input}.mesh", f"{input}.wolf", f"{input}.metrix"], ["SolIni"] * 3)
    os.chdir("SolIni")
    wolf_cmd = [WOLF, "-in", f"{input}", "-out", f"{input}", "-nproc", f"{args.nproc}"]
    run(wolf_cmd + ["-ite", "1000"], "wolf.job")
    # get iter number
    aerocoef = get_aerocoef()
    sim_iter = int(aerocoef[0])
    print(f">> initial Cd = {aerocoef[2]}, Cl = {aerocoef[6]}\n")
    print(f">> execution time: {time.time() - t0} seconds.\n")
    os.chdir(cwd)

    # main loop
    cmp = args.cmp
    ite = 1
    while ite <= args.nite:

        print(f"** ITERATION {ite} - COMPLEXITY {cmp} **")
        print(f"** ----------{'-' * len(str(ite))}--------------{'-' * len(str(cmp))} **")
        # set computational directory
        pdir = f"adap_{ite - 1}" if ite > 1 else "SolIni"
        cdir = f"adap_{ite}"
        os.mkdir(cdir)
        # copy mesh and initial sol to cdir
        if ite == 1:
            cp_filelist(
                [f"{pdir}/{input}.wolf", f"{pdir}/{input}.solb", f"{pdir}/{input}.solb",
                 f"{pdir}/{input}.mesh", f"{pdir}/{input}.metrix", f"{pdir}/{m_field}.solb"],
                [f"{cdir}/file.wolf", f"{cdir}/file.solb", f"{cdir}/file.o.solb",
                 f"{cdir}/file.mesh", f"{cdir}/adap.metrix", f"{cdir}/{m_field}.solb"]
            )
            # update wolf file from Uniform to InitialSol
            sim_args = {"UniformSol": {"inplace": True, "param": ["InitialSol"]}}
            sed_in_file(f"{cdir}/file.wolf", sim_args)
        else:
            cp_filelist(
                [f"{pdir}/file.wolf", f"{pdir}/final.solb", f"{pdir}/final.solb",
                 f"{pdir}/final.mesh", f"{pdir}/adap.metrix", f"{pdir}/{m_field}.solb"],
                [f"{cdir}/file.wolf", f"{cdir}/file.solb", f"{cdir}/file.o.solb",
                 f"{cdir}/file.mesh", f"{cdir}/adap.metrix", f"{cdir}/{m_field}.solb"]
            )
        # sync latest mesh as metrix input mesh
        os.symlink(f"{cwd}/{cdir}/file.mesh", f"{cwd}/{cdir}/adap.mesh")
        # copy residual files
        cp_filelist([f"{pdir}/residual.{sim_iter}.dat", f"{pdir}/res.{sim_iter}.dat",
                     f"{pdir}/aerocoef.{sim_iter}.dat"],
                    [f"{cdir}"] * 3)
        os.chdir(cdir)

        # convergence at fixed complexity
        sub_ite = 1
        n_fail = 0
        cd3 = cd2 = cd1 = 1.
        cl3 = cl2 = cl1 = 1.
        while sub_ite <= args.smax:
            print(f"** SUBITERATION {sub_ite} - ISOCMP {cmp} **")
            print(f"** -------------{'-' * len(str(sub_ite))}----------{'-' * len(str(cmp))} **")
            # create backup files
            init_files = ["file.mesh", "file.o.solb", f"{m_field}.solb"]
            backup_files = ["file.back.mesh", "file.back.solb", f"{m_field}.back.solb"]
            cp_filelist(init_files, backup_files)

            print("** METRIC CONSTRUCTION **")
            cp_filelist([f"{m_field}.solb"], ["adap.solb"])
            metrix_cmd = [METRIX, "-O", "1", "-in", "adap", "-out", "adap.met.solb",
                          "-v", "6", "-Cmp", f"{cmp}", "-hmax", "2"]
            run(metrix_cmd, "metrix.job")

            print("** MESH ADAPTATION **")
            cp_filelist(["file.mesh"], ["adap.met.mesh"])
            feflo_cmd = [FEFLO, "-in", "adap.met", "-met", "adap.met", "-out", "file.mesh",
                         "-noref", "-nordg", "-hgrad", "1.5", "-keep-line-ids", "1, 2, 3"]
            run(feflo_cmd, "feflo.job")

            print("** SOLUTION INTERPOLATION **")
            rm_filelist(["file.solb"])
            interpol_cmd = [INTERPOL, "-O", "1", "-in", "file", "-back", "file.back"]
            run(interpol_cmd, "interpol.job")

            print("** SOLUTION COMPUTATION **")
            rm_filelist(["file.o.solb"])
            wolf_cmd = [WOLF, "-in", "file", "-out", "file.o", "-nproc", f"{args.nproc}", "-v", "6"]
            run(wolf_cmd, f"wolf.{sub_ite}.job")
            rm_filelist(["localCfl.*.solb"])

            res = get_residual()
            if res < res_tgt:
                print(f">> WOLF converged: residual {res} < {res_tgt}")
            else:
                print(f"WARNING -- WOLF did not converge: residual {res} > {res_tgt}")
                n_fail += 1
                # restore backup
                cp_filelist(backup_files, init_files)
                # perturb complexity
                cmp *= 1.01
                # restart convergence at fixed complexity
                sub_ite = 1
                if n_fail > tol_fail:
                    print(f"ERROR -- number of tolerated failures exceeded: {n_fail} > {tol_fail}")
                    return FAILURE
                else:
                    print(f">> restart mesh convergence at fixed complexity {cmp}\n")
                    continue
            cp_filelist(["file.o.solb", "file.mesh"],
                        [f"fin.{sub_ite}.solb", f"fin.{sub_ite}.mesh"])
            cp_filelist([f"{m_field}.solb"], [f"fin.{m_field}.{sub_ite}.solb"])
            ln_filelist([f"fin.{sub_ite}.mesh"], [f"fin.{m_field}.{sub_ite}.mesh"])
            cp_filelist(["file.o.solb", "file.mesh"], ["final.solb", "final.mesh"])
            ln_filelist(["final.mesh", "final.mesh"], ["final.metric.mesh", "final.norot.mesh"])
            cp_filelist(["logCfl.o.solb", "logCfl.o.solb"],
                        [f"fin.logCfl.{sub_ite}.solb", "final.logCfl.solb"])
            print(f">> fin.{sub_ite}.mesh & fin.{sub_ite}.solb created")
            print(f">> execution time: {time.time() - t0} seconds.\n")

            # aerocoef extraction
            aerocoef = get_aerocoef()
            # Cd
            cd1 = cd2
            cd2 = cd3
            cd3 = aerocoef[2]
            print(f">> Cd: {cd3}")
            # Cl
            cl1 = cl2
            cl2 = cl3
            cl3 = aerocoef[6]
            print(f">> Cl: {cl3}\n")
            # convergence check
            cd_cv, cd_d2, cd_d1 = check_cv(cd3, cd2, cd1, cv_tgt_tab[ite - 1])
            cl_cv, cl_d2, cl_d1 = check_cv(cl3, cl2, cl1, cv_tgt_tab[ite - 1])
            if sub_ite >= 3:
                print(f">> Cd {'converged' if cd_cv else 'did not converge'}, "
                      f"E2={cd_d2}, E1={cd_d1}")
                print(f">> Cl {'converged' if cl_cv else 'did not converge'}, "
                      f"E2={cl_d2}, E1={cl_d1}\n")
                sub_ite = args.smax + 1 if cd_cv and cl_cv else sub_ite + 1
            else:
                sub_ite += 1

        sim_iter = int(aerocoef[0])
        cp_filelist(["aerocoef.dat", "wall.dat", "residual.dat"], [f"{cwd}"] * 4)
        os.chdir(cwd)
        ite += 1
        cmp *= 2.

    cp_filelist(
        [f"{cdir}/final.mesh", f"{cdir}/final.solb", f"{cdir}/mach.solb", f"{cdir}/pres.solb"],
        [f"{cwd}"] * 4
    )
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
