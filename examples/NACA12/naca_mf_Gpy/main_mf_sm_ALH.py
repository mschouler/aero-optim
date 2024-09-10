import argparse
import dill as pickle
import numpy as np
import os
import subprocess
import time
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from scipy.stats.qmc import LatinHypercube
import emukit.test_functions
import emukit.multi_fidelity
import GPy

from smt.applications.mfk import MFK
from aero_optim.utils import check_config


class CustomSM:
    def __init__(self, list_of_surrogates: list[MFK]):
        self.los: list[MFK] = list_of_surrogates

    def predict(self, x: np.ndarray) -> list[float]:
        return [sm.predict_values(x) for sm in self.los]  # [Cd, Cl]
    
class mfsm_Gpy:
    def __init__(self, X_lf, X_hf, Y_lf, Y_hf, n_design):
        self.X_lf = X_lf
        self.X_hf = X_hf
        self.Y_lf = Y_lf.reshape((len(Y_lf), 1))
        self.Y_hf = Y_hf.reshape((len(Y_hf), 1))
        self.n_design = n_design
        self.model_def()
        #y_train_l = y_train_l.reshape((len(y_train_l), 1))
        
    def model_def(self):
        print("shape(self.X_lf) =", np.shape(self.X_lf) )
        # print("shape(self.X_hf) =", np.shape(self.X_hf) )
        # print("shape(self.Y_lf) =", np.shape(self.Y_lf) )
        # print("shape(self.Y_hf) =", np.shape(self.Y_hf) )
        # print("self.Y_lf =", self.Y_lf)
        # print("self.Y_hf =", self.Y_hf)
        X_train, Y_train = convert_xy_lists_to_arrays(
            [self.X_lf, self.X_hf], [self.Y_lf, self.Y_hf])
        

        kernels = [GPy.kern.RBF(input_dim=self.n_design),
                   GPy.kern.RBF(input_dim=self.n_design)]
        
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
            kernels)
        
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, lin_mf_kernel, n_fidelities=2)
        
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0.)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0.)
        
        lin_mf_model = GPyMultiOutputWrapper(
            gpy_lin_mf_model, 2, n_optimization_restarts=50) #5
        
        self.model = lin_mf_model

    def model_train(self):
        self.model.optimize()
    
    def predict_values(self, x):
        self.mean, self.var = self.model.predict(x)
        return self.mean


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
        lf_doe_size = lf_config["optim"]["doe_size"]
        ##### ALH method
        
        n_design = hf_config["optim"]["n_design"]
        bound = hf_config["optim"]["bound"]

        # Compute bound interval size
        bound_size = np.diff(bound)[0]
        lower_bound = bound[0]
        
        x_min = np.zeros(n_design)
        X_hf = np.empty((0, n_design))

        engine = LatinHypercube(d=n_design)
        Xlhs_hf = engine.random(n=hf_doe_size)*bound_size + lower_bound

        for k in range(hf_doe_size):
            distmin = np.inf
            for j in range(lf_doe_size):
                distk = np.linalg.norm(Xlhs_hf[k] - X_lf[j])
                if distk < distmin:
                    distmin = distk
                    x_min = X_lf[j]
            X_hf = np.append(X_hf, [x_min], axis=0)
        
        ##### END ALH method
        np.savetxt(
            os.path.join(hf_outdir, "custom_doe.txt"), X_hf
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
        
        ##### Gpy Kriging
        
        # 1.d trains and saves surrogates
        print("SM: training models..")
        # Cd
        
        mfsmGpy_cl = mfsm_Gpy(X_lf, X_hf, Y_lf[:, 1], Y_hf[:, 1], n_design)
        mfsmGpy_cl.model_def()
        mfsmGpy_cl.model_train()
        
        # Cl
        
        mfsmGpy_cd = mfsm_Gpy(X_lf, X_hf, Y_lf[:, 0], Y_hf[:, 0], n_design)
        mfsmGpy_cd.model_def()
        mfsmGpy_cd.model_train()
        
        # saves combined sm
        customGpy_mf_sm = CustomSM([mfsmGpy_cd, mfsmGpy_cl])
        
        with open(os.path.join(hf_outdir, "modelGpy.pkl"), "wb") as handle:
            pickle.dump(customGpy_mf_sm, handle)
        print(f"SM: model saved to {hf_outdir}")
        
        ##### End Gpy Kriging

    # 2. SM based optimization
    print("SM: surrogate model based optimization..")
    exec_cmd = ["optim", "-c", f"{args.config_mfsm}", "-v", f"{args.verbose}", "--pymoo"]
    subprocess.run(exec_cmd, env=os.environ, stdin=subprocess.DEVNULL, check=True)
    print(f"SM: surrogate model based optimization finished after {time.time() - t0} seconds")


if __name__ == '__main__':
    main()
