import numpy as np

from random import Random
from scipy.stats import qmc
from typing import Any


class Generator:
    """
    This class defines a custom generator based on scipy.qmc samplers.
    """
    # some samplers available from scipy.qmc
    # see https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    sampler_list: list[str] = ["lhs", "halton", "sobol"]

    def __init__(self,
                 seed: int,
                 ndesign: int,
                 doe_size: int,
                 sampler_name: str,
                 bound: tuple[Any, ...]):
        """
        Instantiates the Generator class with some optimization parameters and the sampler name.

        **Input**

        - seed (int): seed number of the sampler random number generator.
        - ndesign (int): the number of design variables (dimensions of the problem).
        - doe_size (int): the size of the initial and subsequent generations.
        - sampler_name (str): name of the sampling algorithm used to generate samples.
        - bound (tuple[Any, ...]): design variables boundaries.

        **Inner**

        - initial_doe (list[list[float]]): the initial generation sampled from the generator.
        """
        self.seed: int = seed
        self.ndesign: int = ndesign
        self.doe_size: int = doe_size
        self.sampler: qmc = self.get_sampler(sampler_name)
        self.bound: tuple[Any, ...] = bound
        self.initial_doe: list[list[float]] = self.sampler.random(n=self.doe_size).tolist()

    def get_sampler(self, sampler_name: str):
        """
        **Returns** scipy qmc sampler.
        """
        if sampler_name not in self.sampler_list:
            raise Exception(f"Unrecognized sampler {sampler_name}")
        else:
            return (
                qmc.LatinHypercube(d=self.ndesign, seed=self.seed) if sampler_name == "lhs"
                else qmc.Halton(d=self.ndesign, seed=self.seed) if sampler_name == "halton"
                else qmc.Sobol(d=self.ndesign, seed=self.seed)
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
