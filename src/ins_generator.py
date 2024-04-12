from scipy.stats import qmc


class Generator:
    """
    This class defines a custom generator based on scipy.qmc samplers.
    """
    # some samplers available from scipy.qmc
    # see https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    sampler_list: list[str] = ["lhs", "halton", "sobol"]

    def __init__(
            self, seed: int, ndesign: int, doe_size: int, sampler_name: str, bound: tuple[float]
    ):
        """
        Instantiate the Generator class with some optimization parameters and the sampler name.
        """
        self.seed: int = seed
        self.ndesign: int = ndesign
        self.doe_size: int = doe_size
        self.sampler: qmc = self.get_sampler(sampler_name)
        self.initial_doe: list[list[float]] = self.sampler.random(n=self.doe_size).tolist()
        self.bound = bound

    def get_sampler(self, sampler_name: str):
        """
        Build scipy qmc sampler.
        """
        if sampler_name not in self.sampler_list:
            raise Exception(f"Unrecognized sampler {sampler_name}")
        else:
            return (
                qmc.LatinHypercube(d=self.ndesign, seed=self.seed) if sampler_name == "lhs"
                else qmc.Halton(d=self.ndesign, seed=self.seed) if sampler_name == "halton"
                else qmc.Sobol(d=self.ndesign, seed=self.seed)
            )

    def custom_generator(self, random, args) -> list[float]:
        """
        Define a generator to sample elements one by one from the initial generation.
        """
        element = self.initial_doe.pop(0)
        return qmc.scale([element], *self.bound).tolist()[0]
