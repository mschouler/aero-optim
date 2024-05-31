import argparse
import inspyred
import logging
import operator
import traceback

from src.optim.inspyred_optimizer import DEBUGOptimizer, WolfOptimizer, select_strategy
from src.utils import (catch_signal, check_config, get_evolutionary_computation, set_logger)


def main():
    """
    This program orchestrates a GA optimization loop with inspyred
    https://pythonhosted.org/inspyred/
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/config.json")
    parser.add_argument("-D", "--DEBUG", action="store_true", help="use DEBUG mode")
    parser.add_argument("-cf", "--custom-file", type=str, help="/path/to/custom.py", default="")
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

    # instantiate optimizer and inspyred objects
    if args.custom_file:
        CustomClass = get_evolutionary_computation(args.custom_file)
        custom_ea = CustomClass(config)
    else:
        if args.DEBUG:
            opt = DEBUGOptimizer(config)
        else:
            opt = WolfOptimizer(config)
        ea = select_strategy(opt.strategy, opt.prng)
        ea.observer = opt._observe
        ea.terminator = inspyred.ec.terminators.generation_termination

    # optimization
    try:
        if args.custom_file:
            custom_ea.custom_evolve()
        else:
            final_pop = ea.evolve(generator=opt.generator._ins_generator,
                                  evaluator=opt._evaluate,
                                  pop_size=opt.doe_size,
                                  max_generations=opt.max_generations,
                                  bounder=inspyred.ec.Bounder(*opt.bound),
                                  maximize=opt.maximize,
                                  **opt.ea_kwargs)

            opt.final_observe()

            # output results
            best = max(final_pop)
            index, opt_J = (
                max(enumerate(opt.J), key=operator.itemgetter(1)) if opt.maximize else
                min(enumerate(opt.J), key=operator.itemgetter(1))
            )
            gid, cid = (index // opt.doe_size, index % opt.doe_size)
            logger.info(f"optimal(J): {opt_J}, "
                        f"D: {' '.join([str(d) for d in best.candidate[:opt.n_design]])} "
                        f"[g{gid}, c{cid}]")

    except Exception as e:
        logger.error(
            f"ERROR -- something went wrong in the optimization loop which raised Exception: {e}"
        )
        logger.error(f"Traceback message:\n{traceback.format_exc()}")
        opt.simulator.kill_all()  # kill all remaining processes


if __name__ == '__main__':
    main()
