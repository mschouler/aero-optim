import argparse
import inspyred
import logging
import operator
import os
import shutil
import signal
import traceback

from src.ins_optimizer import WolfOptimizer
from src.utils import (check_file, check_config, check_dir,
                       configure_logger, get_log_level_from_verbosity)
from types import FrameType

signals = [signal.SIGINT, signal.SIGPIPE, signal.SIGTERM]

logger = logging.getLogger()


def handle_signal(signo: int, frame: FrameType | None):
    """
    Raises exception in case of interruption signal.
    """
    signame = signal.Signals(signo).name
    logger.info(f"clean handling of {signame} signal")
    raise Exception("Program interruption")


if __name__ == '__main__':
    """
    This program orchestrates a GA optimization loop.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    # check config and copy to outdir
    check_file(args.config)
    config, study_type = check_config(args.config, optim=True)
    check_dir(config["study"]["outdir"])
    shutil.copy(args.config, config["study"]["outdir"])

    # set logger
    log_level = get_log_level_from_verbosity(args.verbose)
    configure_logger(logger, os.path.join(config["study"]["outdir"], "aero-optim.log"), log_level)

    # instantiate optimizer and inspyred objects
    opt = WolfOptimizer(config)
    ea = inspyred.ec.ES(opt.prng)
    ea.observer = opt.observe
    ea.terminator = inspyred.ec.terminators.generation_termination

    # signal interruption management
    for s in signals:
        signal.signal(s, handle_signal)

    # optimization
    try:
        final_pop = ea.evolve(generator=opt.generator.custom_generator,
                              evaluator=opt.evaluate,
                              pop_size=opt.doe_size,
                              max_generations=opt.max_generations,
                              bounder=inspyred.ec.Bounder(*opt.bound),
                              maximize=opt.maximize)

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
