import argparse
import dill as pickle
import numpy as np
import os
import subprocess
import time

from smt.surrogate_models import KRG
from src.utils import check_config


class CustomSM:
    def __init__(self, list_of_surrogates: list[KRG]):
        self.los: list[KRG] = list_of_surrogates

    def predict(self, x: np.ndarray) -> list[float]:
        return [sm.predict_values(x) for sm in self.los]  # [Cd, Cl]


def main():
    """
    Core script.
    1.a builds DOE i.e. single generation optimization, [optional]
    1.b loads results,                                  [optional]
    1.c trains and saves surrogate,                     [optional]
    2. performs SM based optimization.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=str, help="/path/to/config.json")
    parser.add_argument("-csm", "--config-sm", type=str, help="/path/to/config_sm.json")
    parser.add_argument("-l", "--load", action="store_true", help="load trained model")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()

    if not args.load:
        # 1a. initial doe generation
        print("SM: initial DOE computation..")
        exec_cmd = ["optim", "-c", f"{args.config}", "-v", f"{args.verbose}", "--pymoo"]
        subprocess.run(exec_cmd,
                       env=os.environ,
                       stdin=subprocess.DEVNULL,
                       check=True)
        print(f"SM: initial DOE computation finished after {time.time() - t0} seconds")
        # 1.b loads results
        config, _, _ = check_config(args.config)
        outdir = config["study"]["outdir"]
        print(f"SM: data loading from {outdir}..")
        X = np.loadtxt(os.path.join(outdir, "candidates.txt"))
        Y = []
        with open(os.path.join(outdir, "df_dict.pkl"), "rb") as handle:
            df_dict = pickle.load(handle)
        for gid in range(len(df_dict)):
            for cid in range(len(df_dict[gid])):
                Y. append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
        Y = np.array(Y)
        del df_dict
        # 1.c trains and saves surrogates
        print("SM: training models..")
        # Cd
        sm_cd = KRG(theta0=[1e-2])
        sm_cd.set_training_values(X, Y[:, 0])
        sm_cd.train()
        # Cl
        sm_cl = KRG(theta0=[1e-2])
        sm_cl.set_training_values(X, Y[:, 1])
        sm_cl.train()
        print(f"SM: finished training after {time.time() - t0} seconds")
        # saves combined sm
        custom_sm = CustomSM([sm_cd, sm_cl])
        with open(os.path.join(outdir, "model.pkl"), "wb") as handle:
            pickle.dump(custom_sm, handle)
        print(f"SM: model saved to {outdir}")

    # 2. SM based optimization
    print("SM: surrogate model based optimization..")
    exec_cmd = ["optim", "-c", f"{args.config_sm}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd,
                   env=os.environ,
                   stdin=subprocess.DEVNULL,
                   check=True)
    print(f"SM: surrogate model based optimization finished after {time.time() - t0} seconds")


if __name__ == '__main__':
    main()
