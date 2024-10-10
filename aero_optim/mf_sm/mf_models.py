import numpy as np
import GPy
import emukit.multi_fidelity
import emukit.test_functions
import torch

from abc import ABC, abstractmethod
from aero_optim.mf_sm.mf_dnn import NNH, NNL, MfDNN_train, NNL_pretrain, weights_init
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_x_list_to_array, convert_xy_lists_to_arrays
)
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from smt.applications.mfk import MFK
from smt.surrogate_models import KRG


class MfModel(ABC):
    """
    Wrapping class around a multifidelity model to define it, train it and evaluate it.

    Attributes:
        dim (int): problem dimension.
        model_dict (dict): dictionary containing all necessary model parameters.
        outdir (str): path to the model folder.
        seed (int): seed to enforce reproducibility.
        x_lf_DOE (np.ndarray): the lf_DOE the model was trained with.
        x_hf_DOE (np.ndarray): the hf_DOE the model was trained with.
        y_lf_DOE (np.ndarray): the lf_DOE the model was trained with.
        y_hf_DOE (np.ndarray): the hf_DOE the model was trained with.
    """
    def __init__(self, dim: int, model_dict: dict, outdir: str, seed: int):
        self.dim = dim
        self.model_dict = model_dict
        self.outdir = outdir
        self.seed = seed
        self.requires_nested_doe: bool
        self.x_lf_DOE: np.ndarray | None = None
        self.x_hf_DOE: np.ndarray | None = None
        self.y_lf_DOE: np.ndarray | None = None
        self.y_hf_DOE: np.ndarray | None = None
        # seed numpy
        np.random.seed(seed)

    @abstractmethod
    def train(self):
        """
        Trains a model with low and high fidelity data.
        """

    @abstractmethod
    def evaluate(self, x: np.ndarray):
        """
        Returns a model prediction for a given input x.
        """

    def evaluate_std(self, x: np.ndarray):
        raise Exception("Not implemented")

    def get_DOE(self) -> np.ndarray:
        """
        Returns the DOE the model was trained with.
        """
        assert self.x_lf_DOE is not None and self.x_hf_DOE is not None
        if self.requires_nested_doe:
            return np.copy(self.x_lf_DOE)
        else:
            return np.vstack((self.x_lf_DOE, self.x_hf_DOE))

    def set_DOE(self, x_lf: np.ndarray, x_hf: np.ndarray, y_lf: np.ndarray, y_hf: np.ndarray):
        """
        Sets the hf and lf DOEs the model was trained with.
        """
        if self.x_lf_DOE is None:
            self.x_lf_DOE = x_lf
            self.x_hf_DOE = x_hf
            self.y_lf_DOE = y_lf
            self.y_hf_DOE = y_hf
        else:
            assert (
                self.x_lf_DOE is not None
                and self.x_hf_DOE is not None
                and self.y_lf_DOE is not None
                and self.y_hf_DOE is not None
            )
            self.x_lf_DOE = np.vstack((self.x_lf_DOE, x_lf))
            self.x_hf_DOE = np.vstack((self.x_hf_DOE, x_hf))
            if self.y_lf_DOE.ndim == 2:
                self.y_lf_DOE = np.vstack((self.y_lf_DOE, y_lf))
                self.y_hf_DOE = np.vstack((self.y_hf_DOE, y_hf))
            else:
                self.y_lf_DOE = np.hstack((self.y_lf_DOE, y_lf))
                self.y_hf_DOE = np.hstack((self.y_hf_DOE, y_hf))

    def get_y_star(self) -> float | np.ndarray:
        """
        Returns the current best lf fitness (SOO) or pareto front (MOO).
        """
        # single objective, y_star is simply the min value
        assert self.y_lf_DOE is not None
        return min(self.y_lf_DOE)


class MfDNN(MfModel):
    """
    Wrapping class around a torch multifidelity deep neural network model.
    """
    def __init__(self, dim: int, model_dict: dict, outdir: str, seed: int):
        super().__init__(dim, model_dict, outdir, seed)
        self.requires_nested_doe = model_dict.get("nested_doe", False)
        self.name = model_dict.get("name", "MF-DNN")
        # seed torch
        torch.manual_seed(seed)
        # select device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # the first layer of both network is created here
        self.model_dict["NNL"]["layer_sizes_NNL"].insert(0, dim)
        self.model_dict["NNH"]["layer_sizes_NNH1"].insert(0, dim + 1)
        self.model_dict["NNH"]["layer_sizes_NNH2"].insert(0, dim + 1)
        # low fidelity network
        self.NNL = NNL(self.model_dict["NNL"]["layer_sizes_NNL"])
        # high fidelity network
        self.NNH = NNH(
            self.model_dict["NNH"]["layer_sizes_NNH1"],
            self.model_dict["NNH"]["layer_sizes_NNH2"]
        )
        # to device
        self.NNL.to(self.device)
        self.NNH.to(self.device)
        # init weights
        self.NNL.apply(weights_init)
        self.NNH.apply(weights_init)

    def train(self):
        # pretrain low fidelity model
        weight_decay_NNL = self.model_dict["NNL"]["optimizer"].get("weight_decay", 0)
        if self.model_dict["pretraining"]:
            loss_target = self.model_dict["NNL"].get("loss_target", 1e-3)
            niter = self.model_dict["NNL"].get("niter", 10000)
            # define optimizer
            lr = self.model_dict["NNL"]["optimizer"].get("lr", 1e-3)
            NNL_optimizer = torch.optim.Adam(
                self.NNL.parameters(),
                lr=lr,
                weight_decay=weight_decay_NNL
            )
            # define scheduler
            if "scheduler" in self.model_dict["NNL"]:
                NNL_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    NNL_optimizer, **self.model_dict["NNL"]["scheduler"]
                )
            else:
                NNL_scheduler = None
            # pretrain
            assert self.x_lf_DOE is not None and self.y_lf_DOE is not None
            NNL_pretrain(
                self.NNL,
                NNL_optimizer,
                self.x_lf_DOE, self.y_lf_DOE,
                loss_target,
                niter,
                self.device,
                scheduler=NNL_scheduler
            )
        # coupled training
        loss_target = self.model_dict["NNH"].get("loss_target", 1e-5)
        niter = self.model_dict["NNH"].get("niter", 15000)
        # define optimizer
        lr = self.model_dict["NNH"]["optimizer"].get("lr", 1e-4)
        weight_decay_NNH1 = (
            self.model_dict["NNH"]["optimizer"].get("weight_decay_NNH1", 0)
        )
        weight_decay_NNH2 = (
            self.model_dict["NNH"]["optimizer"].get("weight_decay_NNH2", 0)
        )
        MfDNN_optimizer = torch.optim.Adam(
            [{'params': self.NNH.NNH1.parameters(), 'weight_decay': weight_decay_NNH1},
             {'params': self.NNH.NNH2.parameters(), 'weight_decay': weight_decay_NNH2},
             {'params': self.NNH.alpha},
             {'params': self.NNL.parameters(), 'weight_decay': weight_decay_NNL}], lr=lr
        )
        # define scheduler
        if "scheduler" in self.model_dict["NNH"]:
            MfDNN_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                MfDNN_optimizer,
                **self.model_dict["NNH"]["scheduler"]
            )
        else:
            MfDNN_scheduler = None
        # train
        assert self.x_lf_DOE is not None and self.y_lf_DOE is not None
        assert self.x_hf_DOE is not None and self.y_hf_DOE is not None
        MfDNN_train(
            self.NNL,
            self.NNH,
            MfDNN_optimizer,
            self.x_lf_DOE, self.y_lf_DOE,
            self.x_hf_DOE, self.y_hf_DOE,
            loss_target,
            niter,
            self.device,
            self.outdir,
            scheduler=MfDNN_scheduler
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_torch = torch.from_numpy(x).float()
        x_torch = x_torch.to(self.device)
        y_lo_hi = self.NNL.eval()(x_torch)
        y = self.NNH.eval()(torch.cat((x_torch, y_lo_hi), dim=1))
        return y.cpu().detach().numpy()


class MfSMT(MfModel):
    """
    Wrapping class around an smt multifidelity cokriging model.
    """
    def __init__(self, dim: int, model_dict: dict, outdir: str, seed: int):
        super().__init__(dim, model_dict, outdir, seed)
        self.requires_nested_doe = model_dict.get("nested_doe", True)
        self.name = model_dict.get("name", "AR1")
        self.model = MFK(theta0=dim * [1.0], print_global=False)

    def train(self):
        self.model.set_training_values(self.x_lf_DOE, self.y_lf_DOE, name=0)
        self.model.set_training_values(self.x_hf_DOE, self.y_hf_DOE)
        self.model.train()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = self.model.predict_values(x)
        return y

    def evaluate_std(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self.model.predict_variances(x))


class SfSMT(MfModel):
    """
    Wrapping class around an smt single fidelity kriging model.
    """
    def __init__(self, dim: int, model_dict: dict, outdir: str, seed: int):
        super().__init__(dim, model_dict, outdir, seed)
        self.requires_nested_doe = model_dict.get("nested_doe", False)
        self.name = model_dict.get("name", "GP")
        self.model = KRG(theta0=dim * [1e-2], print_global=False)

    def train(self):
        self.model.set_training_values(self.x_hf_DOE, self.y_hf_DOE)
        self.model.train()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = self.model.predict_values(x)
        return y

    def evaluate_std(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self.model.predict_variances(x))


class MfLGP(MfModel):
    """
    Wrapping class around a Gpy/emukit linear GP multifidelity model.

    Adapted from emukit tutorials:
    https://nbviewer.org/github/emukit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity.ipynb # noqa
    """
    def __init__(self, dim: int, model_dict: dict, outdir: str, seed: int):
        super().__init__(dim, model_dict, outdir, seed)
        self.requires_nested_doe = model_dict.get("nested_doe", True)
        self.name = model_dict.get("name", "AR1")

    def train(self):
        X_train, Y_train = convert_xy_lists_to_arrays(
            [self.x_lf_DOE, self.x_hf_DOE], [self.y_lf_DOE, self.y_hf_DOE]
        )
        kernels = [GPy.kern.RBF(self.x_lf_DOE.shape[1]), GPy.kern.RBF(self.x_lf_DOE.shape[1])]
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_lin_mf_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, lin_mf_kernel, n_fidelities=2
        )
        gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
        gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
        # Wrap the model using the given 'GPyMultiOutputWrapper'
        self.model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
        self.model.optimize()

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = convert_x_list_to_array([x, x])
        X = X[len(x):]
        y, var = self.model.predict(X)
        std = np.sqrt(var)
        return y, std

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)[0]

    def evaluate_std(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)[-1]


class MultiObjectiveModel(MfModel):
    """
    Multi-objective surrogate model.
    """
    def __init__(self, list_of_models: list[MfSMT | MfDNN | SfSMT | MfLGP]):
        self.models: list[MfSMT | MfDNN | SfSMT | MfLGP] = list_of_models

    def train(self):
        for model in self.models:
            model.train()

    def evaluate(self, x: np.ndarray) -> list[np.ndarray]:
        return [model.evaluate(x) for model in self.models]

    def evaluate_std(self, x: np.ndarray) -> list[np.ndarray]:
        return [model.evaluate_std(x) for model in self.models]

    def get_DOE(self) -> np.ndarray:
        return self.models[0].get_DOE()

    def set_DOE(
            self,
            x_lf: np.ndarray, x_hf: np.ndarray,
            y_lf: np.ndarray | list[np.ndarray], y_hf: np.ndarray | list[np.ndarray]
    ):
        for model, yy_lf, yy_hf in zip(self.models, y_lf, y_hf):
            model.set_DOE(x_lf, x_hf, yy_lf, yy_hf)

    def get_y_star(self) -> float | np.ndarray:
        """
        Returns the current best lf fitness (SO) or pareto front (MO).
        """
        raise Exception("Not implemented")

    def compute_pareto(self) -> np.ndarray:
        """
        Returns the current lf fitness pareto front.
        see implementation from the stackoverflow thread below:
        https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python/40239615#40239615 # noqa
        """
        assert self.models[0].y_lf_DOE is not None and self.models[1].y_lf_DOE is not None
        costs = np.column_stack((self.models[0].y_lf_DOE, self.models[1].y_lf_DOE))
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        costs = costs[is_efficient]
        sorted_idx = np.argsort(costs, axis=0)[:, 0]
        return costs[sorted_idx]


def get_model(
        model_name: str, dim: int, config_dict: dict, outdir: str, seed: int
) -> SfSMT | MfSMT | MfLGP | MfDNN:
    """
    Returns a multifidelity model based on the user's specifications.
    """
    if model_name.lower() == "smt":
        return SfSMT(dim, config_dict.get("smt", {}), outdir, seed)
    elif model_name.lower() == "mfsmt":
        return MfSMT(dim, config_dict.get("mfsmt", {}), outdir, seed)
    elif model_name.lower() == "mflgp":
        return MfLGP(dim, config_dict.get("mflgp", {}), outdir, seed)
    elif model_name.lower() == "mfdnn":
        return MfDNN(dim, config_dict["mfdnn"], outdir, seed)
    else:
        raise Exception(f"unrecognized model: {model_name}")


class MfSampler(ABC):
    """
    Wrapping class around LHS samplers.

    Attributes:
        dim (int): problem dimension.
        seed (int): seed to enforce reproducibility.
        bounds (list[float]): sampling bounds.
    """
    def __init__(self, dim: int, bounds: list[float], seed: int):
        self.dim = dim
        self.bounds = bounds
        self.seed = seed

    @abstractmethod
    def sample_mf(self, n_lf: int, n_hf: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns x_lf and x_hf with shape (n_lf, dim) and (n_hf, dim).
        """


class LHSSampler(MfSampler):
    """
    Wrapping class around scipy LHS sampler.
    """
    def __init__(self, dim: int, bounds: list[float], seed: int):
        super().__init__(dim, bounds, seed)
        self.sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)

    def sample_mf(self, n_lf: int, n_hf: int) -> tuple[np.ndarray, np.ndarray]:
        lf_sample = self.sampler.random(n=n_lf)
        x_lf = qmc.scale(lf_sample, *self.bounds)
        hf_sample = self.sampler.random(n=n_hf)
        x_hf = qmc.scale(hf_sample, *self.bounds)
        return x_lf, x_hf

    def sample(self, n: int) -> np.ndarray:
        sample = self.sampler.random(n=n)
        x = qmc.scale(sample, *self.bounds)
        return x


class NestedLHSSampler(MfSampler):
    """
    2 level nested LHS sampler based on scipy LHS.

    Adapted from SMT Nested LHS implementation:
    https://github.com/SMTorg/smt/blob/5f7ad79a4364c9b40ce1979d1685f69714bdec89/smt/applications/mfk.py#L38 # noqa
    """
    def __init__(self, dim: int, bounds: list[float], seed: int):
        super().__init__(dim, bounds, seed)
        self.sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)

    def sample_mf(self, n_lf: int, n_hf: int) -> tuple[np.ndarray, np.ndarray]:
        # high fidelity
        hf_sample = self.sampler.random(n=n_hf)
        x_hf = qmc.scale(hf_sample, *self.bounds)
        # low fidelity
        lf_sample = self.sampler.random(n=n_lf)
        x_lf = qmc.scale(lf_sample, *self.bounds)
        # nearest neighbours deletion
        ind = []
        d = cdist(x_hf, x_lf, "euclidean")
        for j in range(x_hf.shape[0]):
            dj = np.sort(d[j, :])
            k = dj[0]
            ll = (np.where(d[j, :] == k))[0][0]
            m = 0
            while ll in ind:
                m = m + 1
                k = dj[m]
                ll = (np.where(d[j, :] == k))[0][0]
            ind.append(ll)

        x_lf = np.delete(x_lf, ind, axis=0)
        x_lf = np.vstack((x_lf, x_hf))
        return x_lf, x_hf


def get_sampler(
        dim: int,
        bounds: list[float],
        seed: int,
        nested_doe: bool
) -> NestedLHSSampler | LHSSampler:
    """
    Returns a nested or non-nested LHS sampler.
    """
    if nested_doe:
        return NestedLHSSampler(dim, bounds, seed)
    else:
        return LHSSampler(dim, bounds, seed)
