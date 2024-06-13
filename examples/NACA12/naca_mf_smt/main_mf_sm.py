import argparse
import dill as pickle
import numpy as np
import os
import subprocess
import time

from smt.applications.mfk import MFK
from aero_optim.utils import check_config


class CustomSM:
    def __init__(self, list_of_surrogates: list[MFK]):
        self.los: list[MFK] = list_of_surrogates

    def predict(self, x: np.ndarray) -> list[float]:
        return [sm.predict_values(x) for sm in self.los]  # [Cd, Cl]


def main():
    """
    Core script.
    1.a builds LF DOE i.e. single generation optimization,  [optional]
    1.b loads lf results,                                   [optional]
    1.c builds HF DOE from the best candidates              [optional]
    1.d loads hf results                                    [optional]
    1.e trains and saves surrogate,                         [optional]
    2. performs MF SM based optimization.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-clf", "--lf-config", type=str, help="/path/to/lf_config.json")
    parser.add_argument("-chf", "--hf-config", type=str, help="/path/to/hf_config.json")
    parser.add_argument("-cmfsm", "--config-mfsm", type=str, help="/path/to/config_mfsm.json")
    parser.add_argument("-l", "--load", action="store_true", help="load trained model")
    parser.add_argument("-v", "--verbose", type=int, help="logger verbosity level", default=3)
    args = parser.parse_args()

    t0 = time.time()

    if not args.load:
        # 1a. low-fidelity doe generation
        print("SM: LF DOE computation..")
        exec_cmd = ["optim", "-c", f"{args.lf_config}", "-v", f"{args.verbose}", "--pymoo"]
        subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
        print(f"SM: initial DOE computation finished after {time.time() - t0} seconds")

        # 1.b loads LF results
        lf_config, _, _ = check_config(args.lf_config)
        lf_outdir = lf_config["study"]["outdir"]
        print(f"SM: LF data loading from {lf_outdir}..")
        X_lf = np.loadtxt(os.path.join(lf_outdir, "candidates.txt"))
        Y_lf = []
        with open(os.path.join(lf_outdir, "df_dict.pkl"), "rb") as handle:
            df_dict = pickle.load(handle)
        for gid in range(len(df_dict)):
            for cid in range(len(df_dict[gid])):
                Y_lf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
        Y_lf = np.array(Y_lf)
        del df_dict

        # 1.c high-fidelity doe generation
        # setting optim to ensures the output dir is created if necessary
        hf_config, _, _ = check_config(args.hf_config, optim=True)
        hf_outdir = hf_config["study"]["outdir"]
        hf_doe_size = hf_config["optim"]["doe_size"]
        best_candidates_idx = np.argsort(Y_lf[:, 0])
        np.savetxt(
            os.path.join(hf_outdir, "custom_doe.txt"), X_lf[best_candidates_idx][:hf_doe_size]
        )
        print("SM: HF DOE computation..")
        exec_cmd = ["optim", "-c", f"{args.hf_config}", "-v", f"{args.verbose}", "--pymoo"]
        subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)

        # 1.d loads HF results
        print(f"SM: HF data loading from {hf_outdir}..")
        X_hf = np.loadtxt(os.path.join(hf_outdir, "candidates.txt"))
        Y_hf = []
        with open(os.path.join(hf_outdir, "df_dict.pkl"), "rb") as handle:
            df_dict = pickle.load(handle)
        for gid in range(len(df_dict)):
            for cid in range(len(df_dict[gid])):
                Y_hf.append([df_dict[gid][cid]["CD"].iloc[-1], df_dict[gid][cid]["CL"].iloc[-1]])
        Y_hf = np.array(Y_hf)
        del df_dict

        # 1.d trains and saves surrogates
        print("SM: training models..")
        # Cd
        mfsm_cd = MFK(theta0=X_lf.shape[1] * [1.0])
        mfsm_cd.set_training_values(X_lf, Y_lf[:, 0], name=0)
        mfsm_cd.set_training_values(X_hf, Y_hf[:, 0])
        mfsm_cd.train()
        # Cl
        mfsm_cl = MFK(theta0=X_lf.shape[1] * [1.0])
        mfsm_cl.set_training_values(X_lf, Y_lf[:, 1], name=0)
        mfsm_cl.set_training_values(X_hf, Y_hf[:, 1])
        mfsm_cl.train()
        print(f"SM: finished training after {time.time() - t0} seconds")

        # saves combined sm
        custom_mf_sm = CustomSM([mfsm_cd, mfsm_cl])
        with open(os.path.join(hf_outdir, "model.pkl"), "wb") as handle:
            pickle.dump(custom_mf_sm, handle)
        print(f"SM: model saved to {hf_outdir}")

    # 2. SM based optimization
    print("SM: surrogate model based optimization..")
    exec_cmd = ["optim", "-c", f"{args.config_mfsm}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"SM: surrogate model based optimization finished after {time.time() - t0} seconds")


if __name__ == '__main__':
    main()
