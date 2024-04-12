import argparse
import inspyred

from src.ins_optimizer import WolfOptimizer
from src.utils import check_file, check_config


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

    args = parser.parse_args()
    check_file(args.config)
    config, study_type = check_config(args.config, optim=True)

    opt = WolfOptimizer(config)

    problem = inspyred.benchmarks.Rastrigin(opt.n_design)
    ea = inspyred.ec.ES(opt.prng)
    ea.observer = my_observer
    ea.terminator = inspyred.ec.terminators.generation_termination

    try:
        final_pop = ea.evolve(generator=opt.generator.custom_generator,
                              evaluator=opt.evaluate,
                              pop_size=opt.doe_size,
                              max_generations=opt.max_generations,
                              bounder=opt.bound,
                              maximize=False)
        best = max(final_pop)
        print('Best Solution: \n{0}'.format(str(best)))
    except Exception as e:
        print(f"ERROR -- something went wrong in the optimization loop which raised Exception {e}")
        opt.simulator.kill_all()  # kill all remaining processes
