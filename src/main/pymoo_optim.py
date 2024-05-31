import argparse
import logging
import traceback

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from src.optim.pymoo_optimizer import DEBUGOptimizer, WolfOptimizer, select_strategy
from src.utils import (check_config, set_logger, catch_signal)


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
    config, _ = check_config(args.config, optim=True)

    # set logger
    logger = set_logger(
        logging.getLogger(), config["study"]["outdir"], "aero-optim.log", args.verbose
    )

    # signal interruption management
    catch_signal()

    # instantiate optimizer and pymoo objects
    if args.DEBUG:
        opt = DEBUGOptimizer(config)
    else:
        opt = WolfOptimizer(config)

    algorithm = select_strategy(
        opt.strategy, opt.doe_size, opt.generator._pymoo_generator(), opt.ea_kwargs
    )

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
