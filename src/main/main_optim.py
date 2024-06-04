import argparse
import logging
import time
import traceback

from src.optim.evolution import PymooEvolution, InspyredEvolution
from src.utils import (catch_signal, check_config, get_custom_class, set_logger)


def main():
    """
    This program orchestrates a GA optimization loop with pymoo or inspyred
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/config.json")
    parser.add_argument("-p", "--pymoo", action="store_true", help="use the pymoo library")
    parser.add_argument("-i", "--inspyred", action="store_true", help="use the inspyred library")
    parser.add_argument("-f", "--custom-file", type=str, help="/path/to/custom_file.py", default="")
    parser.add_argument("-d", "--debug", action="store_true", help="use DEBUG mode")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()

    # check config and copy to outdir
    config, custom_file, _ = check_config(args.config, args.custom_file, optim=True)

    # set logger
    logger = set_logger(
        logging.getLogger(), config["study"]["outdir"], "aero-optim.log", args.verbose
    )

    # signal interruption management
    catch_signal()

    # instantiate optimizer and pymoo objects
    EvolutionClass = get_custom_class(custom_file, "CustomEvolution") if custom_file else None
    if not EvolutionClass:
        if args.pymoo:
            EvolutionClass = PymooEvolution
            logger.info("pymoo based evolution")
        elif args.inspyred:
            EvolutionClass = InspyredEvolution
            logger.info("inspyred based evolution")
        else:
            raise Exception("ERROR -- some optimizer must be specified")
    evolution = EvolutionClass(config, debug=args.debug)

    # optimization
    try:
        evolution.evolve()
        print(f"successful execution in {time.time() - t0} seconds.")

    except Exception as e:
        logger.error(
            f"ERROR -- something went wrong in the optimization loop which raised Exception: {e}"
        )
        logger.error(f"Traceback message:\n{traceback.format_exc()}")
        evolution.optimizer.simulator.kill_all()  # kill all remaining processes


if __name__ == '__main__':
    main()
