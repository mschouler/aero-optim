import argparse
import logging
import time
import sys

from src.simulator import WolfSimulator
from src.utils import check_file, check_config

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if __name__ == "__main__":
    """
    This program executes a Wolf simulation.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-f", "--file", type=str,
                        help="input mesh file: --file=/path/to/file.mesh", default="")
    parser.add_argument("-o", "--outdir", type=str, help="simulation output directory", default="")
    args = parser.parse_args()

    check_file(args.config)
    config, study_type = check_config(args.config, sim=True)

    if args.outdir:
        print(f">> output directory superseded with {args.outdir}")
        config["study"]["outdir"] = args.outdir

    t0 = time.time()

    simulator = WolfSimulator(config)
    simulator.execute_sim(meshfile=args.file)

    while True:
        if simulator.monitor_sim_progress() > 0:
            print(f">> Simulation under execution {time.time() - t0} s")
            time.sleep(1)
        else:
            break

    print(f">> simulation finished after {time.time() - t0} seconds")
