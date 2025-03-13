import argparse
import functools
import math
import os
import shutil
import subprocess
import sys
import time

from subprocess import CalledProcessError, TimeoutExpired
from aero_optim.utils import cp_filelist, ln_filelist, mv_filelist, rm_filelist, run, sed_in_file

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


def check_cv(
        new_coef: float, old_coef_2: float, old_coef_1: float, target_delta: float
) -> tuple[bool, float, float]:
    delta_1 = abs(new_coef - old_coef_1) / abs(new_coef)
    delta_2 = abs(new_coef - old_coef_2) / abs(new_coef)
    return delta_1 < target_delta and delta_2 < target_delta, delta_2, delta_1


def clean_repo(cwd: str, sim_dir: str, input_name: str, sim_args: dict = {}):
    os.chdir(cwd)
    shutil.rmtree(sim_dir)
    os.mkdir(sim_dir)
    cp_filelist([f"{input_name}.wolf", f"{input_name}.mesh"], [sim_dir] * 2)
    os.chdir(sim_dir)
    if sim_args:
        sed_in_file(f"{input_name}.wolf", sim_args)


def execute_simulation(
        args: argparse.Namespace,
        t0: float,
        cv_tgt: list[float],
        ite_tgt: list[int],
        default_res_tgt: float,
        hgrad: float = 1.5,
        cfac: float = 2.,
        subite_restart: int = 3,
        init_ite: int = 1,
        init_res_tgt: float = 0,
        preprocess: bool = True,
) -> tuple[int, int, bool, bool]:
    """
    **Performs** a multi-iteration mesh adaptation run for a given simulation
    and **enforces** robustness at the subiteration level in case of failure.

    **Input**:

    - args (Namespace): various input args such as nite, input, cmp, nproc.
    - t0 (float): the script start time used to compute the execution time.
    - cv_tgt (list[float]): list of QoI convergence targets for each adaptation iteration.
    - ite_tgt (list[int]): max number of subiteration for each complexity.
    - default_res_tgt (float): threshold residual for a simulation to be considered as converged.
    - hgrad (float): metrix gradation parameter.
    - cfac (float): feflo orientation/coarsening parameter.
    - subite_restart (int): limits the number of times a subiteration is allowed to restart.
    - init_ite (int): adaptation initiation from where to start.
    - init_res_tgt (float): temporary residual target to be used in case of repeated failure.
    - preprocess (bool): whether to coarsen/reorient the initial mesh (True) or not (False).

    Note:
        deterministic and non-deterministic errors can occur at the iteration or subiteration level:
        - when a feflo subprocess times out during the periodic adaptation, the adaptation iteration
        is restarted with a slightly decreased gradation parameter.
        - when a wolf subprocess fails or if the solution has not converged, the subiteration is
        restarted up to subite_restart times with an altered complexity. If the problem persists,
        the adaptation iteration is restarted with a greater residual target or a smaller cfac.
    """
    cwd = os.getcwd()

    # assert required input files exist
    input = args.input.split(".")[0]
    assert os.path.isfile(input + ".wolf")
    assert os.path.isfile(f"{input}.mesh") or os.path.isfile(f"{input}.meshb")

    # adaptation variables
    m_field = "mach"
    cfl = 0.1
    cv_tgt_tab = cv_tgt

    # initialization
    if init_ite == 1:
        if preprocess:
            print("** MESH PREPROCESSING **")
            print("** ------------------ **")
            # re-orient mesh
            cp_filelist([f"{input}.mesh"], [f"{input}_ORI.mesh"])
            spyder_cmd = [SPYDER, "-in", f"{input}_ORI", "-out", f"{input}", "-v", "6"]
            run(spyder_cmd + ["-corner", "-nbrcor", "4", "-corlst", "10", "11", "21", "28"],
                "spyder.job")
            print(">> mesh re-oriented by spyder")
            feflo_cmd = [
                FEFLO, "-in", f"{input}", "-out", f"{input}_fine", "-novol", "-nosurf", "-nordg"
            ]
            run(feflo_cmd, "feflo.job")
            print(">> mesh re-oriented by feflo")
            # create background mesh
            feflo_cmd = [FEFLO, "-in", f"{input}_fine", "-prepro", "-nordg", "-out", "tmp"]
            run(feflo_cmd, "feflo.job")
            print(">> background mesh created")
            mv_filelist(
                ["tmp.back.meshb", "tmp.back.solb"], [f"{input}.back.meshb", f"{input}.back.solb"]
            )
            rm_filelist(["tmp.*"])
            # check orientation and coarsen mesh
            feflo_cmd = [
                FEFLO, "-in", f"{input}_fine", "-back", f"{input}.back", "-hmsh",
                "-cfac", f"{cfac}", "-nordg", "-hmax", "0.01", "-out", f"{input}"
            ]
            run(feflo_cmd + ["-keep-line-ids", "10,11,21,28"], "feflo.job")
            print(f">> mesh re-orientation/coarsening succeeded with cfac {cfac}\n")

        print("** INITIAL SOLUTION COMPUTATION WITH ~1000 ITERATIONS **")
        print("** -------------------------------------------------- **")
        # sol. initialization (order 1)
        os.makedirs("So1", exist_ok=True)
        cp_filelist([f"{input}.meshb", f"{input}.wolf"],
                    [f"So1/{input}.meshb", f"So1/{input}.wolf"])
        os.chdir("So1")
        sim_args = {"Order": {"inplace": False, "param": ["1"]},
                    "OrderTrb": {"inplace": False, "param": ["1"]}}
        sed_in_file(f"{input}.wolf", sim_args)
        wolf_cmd = [WOLF, "-in", f"{input}", "-nproc", f"{args.nproc}"]
        try:
            run(wolf_cmd + ["-uni"], "wolf.job")
        except CalledProcessError:
            print("ERROR -- 1st order solution computation failed >> retry once")
            run(wolf_cmd + ["-uni"], "wolf.job")
        print(f">> Order 1 - residual: {get_residual()}, "
              f"execution time: {time.time() - t0} seconds.\n")
        # get iter number
        sim_iter = int(get_turbocoef()[0])
        os.chdir(cwd)

        # sol. initialization (order 2)
        os.makedirs("So2", exist_ok=True)
        cp_filelist([f"{input}.meshb", f"{input}.wolf", f"So1/{input}.o.solb"],
                    [f"So2/{input}.meshb", f"So2/{input}.wolf", f"So2/{input}.solb"])
        cp_filelist([f"So1/residual.{sim_iter}.dat", f"So1/res.{sim_iter}.dat",
                    f"So1/aerocoef.{sim_iter}.dat", f"So1/turbocoef.{sim_iter}.dat"],
                    ["So2"] * 4)
        os.chdir("So2")
        try:
            run(wolf_cmd, "wolf.job")
        except CalledProcessError:
            print("ERROR -- 2nd order solution computation failed >> retry once")
            run(wolf_cmd, "wolf.job")
        print(f">> Order 2 - residual: {get_residual()}, "
              f"execution time: {time.time() - t0} seconds.\n")
        # get iter number
        sim_iter = int(get_turbocoef()[0])
        os.chdir(cwd)

        # setup working directory
        cp_filelist([f"So2/{input}.o.solb", f"So2/{input}.metric.solb"],
                    [f"{input}.solb", f"{input}.metric.solb"])

    # adaptation loop
    n_restart = 0
    cmp = args.cmp * 2**(init_ite - 1)
    ite = init_ite
    res_tgt = init_res_tgt if init_res_tgt else default_res_tgt
    while ite <= args.nite:
        print(f"** ITERATION {ite} - COMPLEXITY {cmp} **")
        print(f"** ----------{'-' * len(str(ite))}--------------{'-' * len(str(cmp))} **")
        # set computational directory
        pdir = f"adap_{ite - 1}" if ite > 1 else "So2"
        cdir = f"adap_{ite}"
        print(f">> current directory: {cdir}")
        shutil.rmtree(cdir, ignore_errors=True)
        os.mkdir(cdir)
        # copy background mesh
        cp_filelist([f"{input}.back.meshb", f"{input}.back.solb"], [f"{cdir}"] * 2)

        if ite == 1:
            # copy input files
            cp_filelist([f"{input}.wolf", f"{input}.meshb", f"{input}.solb"], [f"{cwd}/{cdir}"] * 3)
        else:
            # copy input files
            cp_filelist(
                [f"{input}.wolf", f"{pdir}/final.meshb", f"{pdir}/final.solb",
                 f"{pdir}/final.metric.solb"],
                [f"{cdir}/{input}.wolf", f"{cdir}/{input}.meshb", f"{cdir}/{input}.solb",
                 f"{cdir}/{input}.metric.solb"]
            )
            # retrieve sim_iter
            sim_iter = int(get_turbocoef(f"{pdir}/turbocoef.dat")[0])
        # copy residual files
        cp_filelist([f"{pdir}/residual.{sim_iter}.dat", f"{pdir}/res.{sim_iter}.dat",
                    f"{pdir}/aerocoef.{sim_iter}.dat", f"{pdir}/turbocoef.{sim_iter}.dat"],
                    [f"{cdir}"] * 4)

        os.chdir(cdir)
        cp_filelist(
            [f"{input}.wolf", f"{input}.wolf", f"{input}.solb", f"{input}.solb", f"{input}.meshb"],
            ["file.wolf", "DEFAULT.wolf", "file.solb", "file.o.solb", "file.meshb"]
        )
        ln_filelist(["file.meshb", "file.meshb", "file.back.meshb"],
                    ["file.metric.meshb", "logCfl.meshb", "logCfl.back.meshb"])

        # compute metric from initial condition
        wolf_cmd = [WOLF, "-in", "file", "-out", "file.o", "-nproc", f"{args.nproc}", "-v", "6"]
        try:
            run(wolf_cmd + ["-ite", "1", "-C", f"{cmp}", "-cfl", f"{cfl}"], "wolf.job")
        except CalledProcessError:
            print("ERROR -- metric computation failed >> retry once")
            run(wolf_cmd + ["-ite", "1", "-C", f"{cmp}", "-cfl", f"{cfl}"], "wolf.job")
        print(">> computed metric from initial condition\n")

        # convergence at fixed complexity
        sub_ite = 1
        q_ratio3 = q_ratio1 = q_ratio2 = 1.
        ptot_ratio3 = ptot_ratio1 = ptot_ratio2 = 1.
        ttot_ratio3 = ttot_ratio1 = ttot_ratio2 = 1.
        loss_coef3 = loss_coef1 = loss_coef2 = 1.
        iseff3 = iseff1 = iseff2 = 1.
        while sub_ite <= ite_tgt[ite - 1]:
            print(f"** SUBITERATION {sub_ite} - ISOCMP {cmp} **")
            print(f"** -------------{'-' * len(str(sub_ite))}----------{'-' * len(str(cmp))} **")
            # create backup files
            init_files = ["file.meshb", "file.o.solb", "file.meshb", "file.metric.solb"]
            backup_files = [
                "file.back.meshb", "file.back.solb", "adap.met.meshb", "file.back.metric.solb"
            ]
            cp_filelist(init_files, backup_files)
            rm_filelist(["Back.meshb"])
            print("** PERIODIC ITERATION **")
            print("** ------------------ **")

            # metric gradation
            metrix_cmd = [METRIX, "-in", "file.metric", "-met", "file.metric",
                          "-out", "adap.met.solb", "-v", "6", "-nproc", f"{args.nproc}",
                          "-Cmp", f"{cmp}", "-hgrad", f"{hgrad}", "-cofgrad", "1.5"]
            run(metrix_cmd, "metrix.job")
            print(f">> metric gradation succeeded with -hgrad {hgrad}")

            # mesh adaptation
            # feflo CycleMet
            shutil.rmtree(f"fefloa_{sub_ite}", ignore_errors=True)
            os.mkdir(f"fefloa_{sub_ite}")
            cp_filelist(["adap.met.meshb", "adap.met.solb"], [f"fefloa_{sub_ite}"] * 2)
            feflo_cmd = [FEFLO, "-in", "adap.met", "-met", "adap.met", "-out",
                         "CycleMet.meshb", "-keep-line-ids", "11,28", "-nordg",
                         "-back", f"{input}.back" , "-itp", "file.back.solb"]
            try:
                run(feflo_cmd, f"feflo.{sub_ite}.job", timeout=30.)
            except TimeoutExpired:
                print("ERROR -- feflo CycleMet subprocess timed out")
                os.chdir(cwd)
                return FAILURE, ite, True, False
            except CalledProcessError:
                print("ERROR -- feflo CycleMet subprocess failed")
                os.chdir(cwd)
                return FAILURE, ite, False, False

            print(">> feflo CycleMet succeeded")
            mv_filelist(["CycleMet.solb", "file.back.itp.solb"],
                        ["CycleMet.Met.solb", "CycleMet.solb"])
            rm_filelist(["Forth.meshb"])

            # wolf cycleforth
            wolf_cmd = [WOLF, "-in", "CycleMet", "-cycleforth", "-cyclenbl", "4"]
            run(wolf_cmd, "cycleforth.job")
            print(">> wolf cycleforth succeeded")
            assert os.path.isfile("Forth.meshb")
            mv_filelist(["ForthMet.solb"], ["CycleMetForth.solb"])

            # feflo CycleMetForth
            feflo_cmd = [FEFLO, "-in", "Forth", "-met", "CycleMetForth", "-out",
                         "Cycleadap.meshb", "-keep-line-ids", "32125,32126,1,2,3,4"]
            try:
                run(feflo_cmd, f"feflo.{sub_ite}.job", timeout=30.)
            except TimeoutExpired:
                print("ERROR -- feflo CycleMetForth subprocess timed out")
                os.chdir(cwd)
                return FAILURE, ite, True, False
            print(">> feflo CycleMetForth succeeded")

            # wolf cycleback
            wolf_cmd = [WOLF, "-in", "Cycleadap", "-cycleback"]
            try:
                run(wolf_cmd, "cycleback.job")
            except CalledProcessError:
                print("ERROR -- wolf cycleback subprocess failed")
                if n_restart >= subite_restart:
                    print(f"ERROR -- maximal subite restart number reached ({n_restart})")
                    os.chdir(cwd)
                    return FAILURE, ite, False, False
                else:
                    cp_filelist(backup_files, init_files)
                    cmp *= 1.01
                    n_restart += 1
                    print(f"ERROR -- sub_ite restart with Cmp {cmp} ({n_restart})")
                    continue
            print(">> wolf cycleback succeeded")

            print(f"** INTERPOLATION {sub_ite} **")
            print(f"** --------------{'-' * len(str(sub_ite))} **")
            mv_filelist(["Back.meshb"], ["file.meshb"])
            rm_filelist(["file.solb"])
            interpol_cmd = [INTERPOL, "-O", "1", "-p3", "-in", "file",
                            "-back", "file.back", "-nproc", f"{args.nproc}"]
            try:
                run(interpol_cmd, "interpol.job")
            except CalledProcessError:
                print("ERROR -- interpolation subprocess failed >> sub_ite restart")
                cp_filelist(backup_files, init_files)
                continue
            print(">> interpolation succeeded")

            print(f"** SOLUTION COMPUTATION {sub_ite} **")
            print(f"** ---------------------{'-' * len(str(sub_ite))} **")
            rm_filelist(["file.o.solb", f"wolf.{sub_ite}.job"])
            wolf_cmd = [WOLF, "-in", "file", "-out", "file.o", "-nproc", f"{args.nproc}", "-v", "6"]
            try:
                run(wolf_cmd + ["-C", f"{cmp}", "-profile", "-cfl", f"{cfl}"],
                    f"wolf.{sub_ite}.job")
            except CalledProcessError:
                print("ERROR -- wolf subprocess failed")
                if n_restart >= subite_restart:
                    print(f"ERROR -- maximal subite restart number reached ({n_restart})\n")
                    os.chdir(cwd)
                    return FAILURE, ite, False, True
                else:
                    cp_filelist(backup_files, init_files)
                    cmp *= 1.01
                    n_restart += 1
                    print(f"ERROR -- sub_ite restart with Cmp {cmp} ({n_restart})\n")
                    continue
            rm_filelist(["localCfl.*.solb"])

            # residual monitoring
            res = get_residual()
            if res < res_tgt:
                print(f">> WOLF converged: residual {res} < {res_tgt}")
                res_tgt = default_res_tgt
            else:
                print(f"ERROR -- WOLF did not converge: residual {res} > {res_tgt}")
                if n_restart >= subite_restart:
                    print(f"ERROR -- maximal subite restart number reached ({n_restart})\n")
                    os.chdir(cwd)
                    return FAILURE, ite, False, False
                else:
                    cp_filelist(backup_files, init_files)
                    cmp *= 1.01
                    n_restart += 1
                    print(f"ERROR -- wolf did not converge "
                          f">> sub_ite restart with Cmp {cmp} res tgt {res_tgt} ({n_restart})\n")
                    continue

            # save results files
            cp_filelist(
                ["file.o.solb", "file.meshb", "file.metric.solb"],
                [f"fin.{sub_ite}.solb", f"fin.{sub_ite}.meshb", f"fin.metric.{sub_ite}.solb"]
            )
            cp_filelist([f"{m_field}.solb"], [f"fin.{m_field}.{sub_ite}.solb"])
            ln_filelist([f"fin.{sub_ite}.meshb"], [f"fin.{m_field}.{sub_ite}.meshb"])
            cp_filelist(["file.o.solb", "file.meshb", "file.metric.solb"],
                        ["final.solb", "final.meshb", "final.metric.solb"])
            ln_filelist(["final.meshb", "final.meshb"], ["final.metric.meshb", "final.norot.meshb"])
            cp_filelist(["logCfl.o.solb", "logCfl.o.solb"],
                        [f"fin.logCfl.{sub_ite}.solb", "final.logCfl.solb"])
            print(f">> fin.{sub_ite}.meshb & fin.{sub_ite}.solb created")
            print(f">> execution time: {time.time() - t0} seconds.\n")

            # turbocoef extraction
            turbocoef = get_turbocoef()
            # debit
            q_ratio1 = q_ratio2
            q_ratio2 = q_ratio3
            q_ratio3 = turbocoef[4]
            print(f">> debit ratio: {q_ratio3}")
            # total pressure
            ptot_ratio1 = ptot_ratio2
            ptot_ratio2 = ptot_ratio3
            ptot_ratio3 = turbocoef[7]
            print(f">> total pressure ratio: {ptot_ratio3}")
            # total temperature
            ttot_ratio1 = ttot_ratio2
            ttot_ratio2 = ttot_ratio3
            ttot_ratio3 = turbocoef[10]
            print(f">> total temperature ratio: {ttot_ratio3}")
            # isentropic efficiency
            iseff1 = iseff2
            iseff2 = iseff3
            iseff3 = turbocoef[11]
            print(f">> isentropic efficiency: {iseff3}")
            # loss coefficient
            loss_coef1 = loss_coef2
            loss_coef2 = loss_coef3
            loss_coef3 = turbocoef[12]
            print(f">> loss coefficient: {loss_coef3}\n")
            # convergence check
            debit_cv, dd2, dd1 = check_cv(q_ratio3, q_ratio2, q_ratio1, cv_tgt_tab[ite - 1])
            ptot_cv, pd2, pd1 = check_cv(ptot_ratio3, ptot_ratio2, ptot_ratio1, cv_tgt_tab[ite - 1])
            _, td2, td1 = check_cv(ttot_ratio3, ttot_ratio2, ttot_ratio1, cv_tgt_tab[ite - 1])
            _, ied2, ied1 = check_cv(iseff3, iseff2, iseff1, cv_tgt_tab[ite - 1])
            loss_cv, lcd2, lcd1 = check_cv(loss_coef3, loss_coef2, loss_coef1, cv_tgt_tab[ite - 1])
            if sub_ite >= 3:
                print(f">> debit ratio {'converged' if debit_cv else 'did not converge'}, "
                      f"E2={dd2}, E1={dd1}")
                print(f">> Ptot ratio {'converged' if ptot_cv else 'did not converge'}, "
                      f"E2={pd2}, E1={pd1}")
                print(f">> LossCoef {'converged' if loss_cv else 'did not converge'}, "
                      f"E2={lcd2}, E1={lcd1}")
                print(f">> isentropic efficiency relative differences: E2={ied2}, E1={ied1}")
                print(f">> Ttot ratio relative differences: E2={td2}, E1={td1}\n")
                sub_ite = ite_tgt[ite - 1] + 1 if debit_cv and ptot_cv and loss_cv else sub_ite + 1
            else:
                sub_ite += 1

        sim_iter = int(turbocoef[0])
        cp_filelist(["aerocoef.dat", "turbocoef.dat", "wall.dat", "residual.dat"], [f"{cwd}"] * 4)
        os.chdir(cwd)
        ite += 1
        cmp = args.cmp * 2**(ite - 1)
        n_restart = 0

    cp_filelist(
        [f"{cdir}/final.meshb", f"{cdir}/final.solb", f"{cdir}/mach.solb", f"{cdir}/pres.solb"],
        [f"{cwd}"] * 4
    )
    return SUCCESS, ite, False, False


def robust_execution(
        args: argparse.Namespace,
        t0: float,
        cv_tgt: list[float],
        ite_tgt: list[int],
        ite_restart: int,
        subite_restart: int,
        default_res_tgt: float = 1e-2,
        preprocess: bool = True,
        hgrad: float = 1.5,
        cfac: float = 2.
) -> int:
    """
    Wrapper around execute_simulation with enhanced robustness at the iteration level.
    **Returns** the exit_status to prevent infinite loops with ill-conditioned simulations.

    Note:
        in spite of the robustness measures implemented in execute_simulation, adaptation or
        convergence issues may still require a full iteration restart.
        Nondeterminism induced by the parallel execution of wolf and feflo may be enough to solve
        the problem but sometimes decreasing the gradation parameter or cfac may help pass the
        blocking adaptation iteration.
    """
    exit_status, ite, dhgrad, dcfac = execute_simulation(
        args, t0, cv_tgt, ite_tgt, default_res_tgt,
        subite_restart=subite_restart,
        preprocess=preprocess
    )
    new_ite, restart = ite, 0
    while exit_status == FAILURE:
        if restart < ite_restart:
            restart += 1
            print(f"ERROR -- sim. did not converge >> restart from ite {new_ite} ({restart})\n")
            exit_status, ite, dhgrad, dcfac = execute_simulation(
                args, t0, cv_tgt, ite_tgt, default_res_tgt,
                subite_restart=subite_restart,
                init_ite=new_ite,
                preprocess=preprocess,
                hgrad=hgrad - 0.025 if dhgrad else hgrad,
                cfac=cfac - 0.2 if dcfac else cfac
            )
        else:
            return exit_status
    return exit_status


def submit_process(name: str, exec_cmd: list[str]) -> subprocess.Popen[str]:
    """
    **Submits** exec_cmd and **returns** the corresponding process.
    """
    with open(f"{name}.out", "wb") as out:
        with open(f"{name}.err", "wb") as err:
            print(f"INFO -- execute {name}")
            proc = subprocess.Popen(exec_cmd,
                                    env=os.environ,
                                    stdin=subprocess.DEVNULL,
                                    stdout=out,
                                    stderr=err,
                                    universal_newlines=True)
    return proc


def monitor_returncode(l_proc: list[subprocess.Popen[str]]) -> int:
    """
    **Checks** the state of the processes and **returns** the number of living ones.
    """
    finished_sim = []
    for id, p_id in enumerate(l_proc):
        returncode = p_id.poll()
        if returncode is None:
            pass  # simulation still running
        elif returncode == 0:
            print(f"INFO -- simulation {p_id} finished")
            finished_sim.append(id)
            break
        else:
            print("ERROR -- one process failed, all simulations will be killed")
            for p in l_proc:
                p.terminate()
            raise Exception(f"ERROR -- simulation {p_id} failed")
    return len([p_id for id, p_id in enumerate(l_proc) if id not in finished_sim])


def wait_for_it(l_proc: list[subprocess.Popen[str]]):
    """
    **Waits** until at lest one processes finishes.
    """
    n_proc = len(l_proc)
    while monitor_returncode(l_proc) >= n_proc:
        time.sleep(1)


def main() -> int:
    """
    This program runs a CFD simulation with mesh adaptation i.e. coupling WOLF, METRIX and FEFLO.
    In multisimulation mode, the OP simulations try to start from the ADP adapted meshes (from
    highest to lowest complexity).

    Note:
        robustness is ensured through a failure reaction protocol at both the iteration
        and the subiteration levels.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-in", "--input", type=str, help="input mesh file i.e. input.mesh")
    parser.add_argument("-cmp", type=int, help="targetted complexity")
    parser.add_argument("-nproc", type=int, help="number of procs", default=1)
    parser.add_argument("-nite", type=int, help="number of complexities", default=2)
    parser.add_argument("-ms", type=int, help="number of parallel simulations", default=-1)
    parser.add_argument("-adp", action="store_true", help="simulate ADP")
    parser.add_argument("-op1", action="store_true", help="simulate OP1")
    parser.add_argument("-op2", action="store_true", help="simulate OP2")

    args = parser.parse_args()
    t0 = time.time()
    print(f"simulations performed with: {args}\n")

    input = args.input.split(".")[0]
    cv_tgt = [0.01, 0.01, 0.005, 0.003] + [0.003] * max(0, args.nite - 4)
    ite_tgt = [15, 15, 10, 5] + [5] * max(0, args.nite - 4)

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
        l_proc.append(submit_process("ADP", exec_cmd))
        # op1
        exec_cmd = [sys.executable] + exec_args + ["-op1"]
        if len(l_proc) < args.ms:
            l_proc.append(submit_process("OP1", exec_cmd))
        else:
            wait_for_it(l_proc)
            l_proc.append(submit_process("OP1", exec_cmd))
        # op2
        exec_cmd = [sys.executable] + exec_args + ["-op2"]
        if len(l_proc) < args.ms:
            l_proc.append(submit_process("OP2", exec_cmd))
        else:
            wait_for_it(l_proc)
            l_proc.append(submit_process("OP2", exec_cmd))
        # wait for all processes to finish
        wait_for_it(l_proc)

    if args.adp:
        print("** ADP SIMULATION **")
        print("** -------------- **")
        sim_dir = "ADP"
        shutil.rmtree(sim_dir, ignore_errors=True)
        os.mkdir(sim_dir)
        cp_filelist([f"{input}.wolf", f"{input}.mesh"], [sim_dir] * 2)
        os.chdir(sim_dir)
        exit_status = robust_execution(args, t0, cv_tgt, ite_tgt, ite_restart=3, subite_restart=5)
        if exit_status == FAILURE:
            print(f"ERROR -- adaptation failed after {time.time() - t0} seconds")
            return FAILURE

    if args.op1:
        print("** OP1 SIMULATION (+5 deg.) **")
        print("** ------------------------ **")
        sim_dir = "OP1"
        shutil.rmtree(sim_dir, ignore_errors=True)
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
        exit_status = robust_execution(args, t0, cv_tgt, ite_tgt, ite_restart=3, subite_restart=5)
        if exit_status == FAILURE:
            print(f"ERROR -- adaptation failed after {time.time() - t0} seconds")
            return FAILURE

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
        exit_status = robust_execution(args, t0, cv_tgt, ite_tgt, ite_restart=3, subite_restart=5)
        if exit_status == FAILURE:
            print(f"ERROR -- adaptation failed after {time.time() - t0} seconds")
            return FAILURE

    print(f">> simulations finished successfully in {time.time() - t0} seconds.")
    return SUCCESS


if __name__ == "__main__":
    sys.exit(main())
