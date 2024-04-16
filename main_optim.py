import argparse
import inspyred
import logging
import os
import signal

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


def my_observer(population, num_generations, num_evaluations, args):
    best = max(population)
    print('{0:6} -- {1} : {2}'.format(num_generations,
                                      best.fitness,
                                      str(best.candidate)))


if __name__ == '__main__':
    """
    This program orchestrates a GA optimization loop.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    # check config
    check_file(args.config)
    config, study_type = check_config(args.config, optim=True)

    # set logger
    check_dir(config["study"]["outdir"])
    log_level = get_log_level_from_verbosity(args.verbose)
    configure_logger(logger, os.path.join(config["study"]["outdir"], "aero-optim.log"), log_level)

    # instantiate optimizer and inspyred objects
    opt = WolfOptimizer(config)
    ea = inspyred.ec.ES(opt.prng)
    ea.observer = my_observer
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
                              bounder=opt.bound,
                              maximize=False)
        best = max(final_pop)
        logger.info('Best Solution: \n{0}'.format(str(best)))

    except Exception as e:
        logger.error(
            f"ERROR -- something went wrong in the optimization loop which raised Exception: {e}"
        )
        opt.simulator.kill_all()  # kill all remaining processes
