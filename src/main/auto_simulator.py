import argparse
import logging
import time
import sys

from src.simulator.simulator import WolfSimulator
from src.utils import check_config, get_custom_class

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main():
    """
    This program executes a Wolf simulation.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--config", type=str, help="config: --config=/path/to/config.json")
    parser.add_argument("-f", "--file", type=str,
                        help="input mesh file: --file=/path/to/file.mesh", default="")
    parser.add_argument("-o", "--outdir", type=str, help="simulation output directory", default="")
    args = parser.parse_args()

    config, custom_file, _ = check_config(args.config, sim=True)

    if args.outdir:
        print(f">> output directory superseded with {args.outdir}")
        config["study"]["outdir"] = args.outdir

    t0 = time.time()

    SimulatorClass = get_custom_class(custom_file, "CustomSimulator") if custom_file else None
    if not SimulatorClass:
        SimulatorClass = WolfSimulator
    simulator = SimulatorClass(config)
    simulator.execute_sim(meshfile=args.file)

    while True:
        if simulator.monitor_sim_progress() > 0:
            print(f">> Simulation under execution {time.time() - t0} s")
            time.sleep(1)
        else:
            break

    print(f">> simulation finished after {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
