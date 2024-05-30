import argparse
import logging
import os
import shutil
import traceback

from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.optim.pymoo_optimizer import DEBUGOptimizer, WolfOptimizer
from src.utils import (check_file, check_config, check_dir, configure_logger,
                       get_log_level_from_verbosity, catch_signal)

logger = logging.getLogger()


def main():
    """
    This program orchestrates a GA optimization loop with pymoo
    https://pymoo.org/
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-D", "--DEBUG", action="store_true", help="use DEBUG mode")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    # check config and copy to outdir
    check_file(args.config)
    config, _ = check_config(args.config, optim=True)
    check_dir(config["study"]["outdir"])
    shutil.copy(args.config, config["study"]["outdir"])

    # set logger
    log_level = get_log_level_from_verbosity(args.verbose)
    configure_logger(logger, os.path.join(config["study"]["outdir"], "aero-optim.log"), log_level)

    # instantiate optimizer and pymoo objects
    if args.DEBUG:
        opt = DEBUGOptimizer(config)
    else:
        opt = WolfOptimizer(config)

    algorithm = PSO(pop_size=opt.doe_size, sampling=opt.generator._pymoo_generator())

    # signal interruption management
    catch_signal()

    # optimization
    try:
        res = minimize(problem=opt,
                       algorithm=algorithm,
                       termination=get_termination("n_gen", opt.max_generations),
                       seed=opt.seed,
                       verbose=True)

        opt.final_observe()

        # output results
        best = res.F
        index, opt_J = min(enumerate(opt.J), key=lambda x: abs(best - x[1]))
        gid, cid = (index // opt.doe_size, index % opt.doe_size)
        logger.info(f"optimal(J): {opt_J} ({best}), "
                    f"D: {' '.join([str(d) for d in res.X])} "
                    f"[g{gid}, c{cid}]")

    except Exception as e:
        logger.error(
            f"ERROR -- something went wrong in the optimization loop which raised Exception: {e}"
        )
        logger.error(f"Traceback message:\n{traceback.format_exc()}")
        opt.simulator.kill_all()  # kill all remaining processes


if __name__ == '__main__':
    main()
