import numpy as np

from random import Random
from scipy.stats import qmc
from typing import Any, Optional


class Generator:
    """
    This class defines a custom generator based on scipy.qmc samplers.
    """
    # some samplers available from scipy.qmc
    # see https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    sampler_list: list[str] = ["lhs", "halton", "sobol", "custom"]

    def __init__(self,
                 seed: int,
                 ndesign: int,
                 doe_size: int,
                 sampler_name: str,
                 bound: tuple[Any, ...],
                 custom_doe: str = ""):
        """
        Instantiates the Generator class with some optimization parameters and the sampler name.

        **Input**

        - seed (int): seed number of the sampler random number generator.
        - ndesign (int): the number of design variables (dimensions of the problem).
        - doe_size (int): the size of the initial and subsequent generations.
        - sampler_name (str): name of the sampling algorithm used to generate samples.
        - bound (tuple[Any, ...]): design variables boundaries.
        - custom_doe (str): path to the text file containing a custom doe.

        **Inner**

        - initial_doe (list[list[float]]): the initial generation sampled from the generator.
        """
        self.seed: int = seed
        self.ndesign: int = ndesign
        self.doe_size: int = doe_size
        self.sampler: Optional[qmc.QMCEngine] = self.get_sampler(
            "custom" if custom_doe else sampler_name
        )
        self.bound: tuple[Any, ...] = bound
        self.initial_doe: list[list[float]] = self.sample_doe(custom_doe)

    def get_sampler(self, sampler_name: str) -> Optional[qmc.QMCEngine]:
        """
        **Returns** scipy qmc sampler.
        """
        if sampler_name not in self.sampler_list:
            raise Exception(f"Unrecognized sampler {sampler_name}")
        else:
            return (
                qmc.LatinHypercube(d=self.ndesign, seed=self.seed) if sampler_name == "lhs"
                else qmc.Halton(d=self.ndesign, seed=self.seed) if sampler_name == "halton"
                else qmc.Sobol(d=self.ndesign, seed=self.seed) if sampler_name == "sobol"
                else None
            )

    def sample_doe(self, custom_doe: str) -> list[list[float]]:
        return (
            self.sampler.random(n=self.doe_size).tolist() if self.sampler
            else [
                [float(xi) for xi in X.strip().split()]
                for X in open(custom_doe, "r").read().splitlines()
            ]
        )

    def _ins_generator(self, random: Random, args: dict) -> list[float]:
        """
        **Returns** a single sample from the initial generation.

        Note:
            __random__ and __args__ are inspyred mandatory arguments</br>
            see https://pythonhosted.org/inspyred/tutorial.html#the-generator
        """
        element = self.initial_doe.pop(0)
        return qmc.scale([element], *self.bound).tolist()[0]

    def _pymoo_generator(self) -> np.ndarray:
        """
        **Returns** all samples from the initial generation.
        """
        return qmc.scale(self.initial_doe, *self.bound)
