import pytest

from scipy.stats import qmc
from src.ins_generator import Generator

seed: int = 123
ndesign: int = 4
doe_size: int = 16
bound: tuple[float, float] = (-1., 1.)
sampler_list: list[str] = Generator.sampler_list
sampler_type_list: list[type] = [qmc.LatinHypercube, qmc.Halton, qmc.Sobol]


def get_Generator(sampler_name: str) -> Generator:
    return Generator(seed, ndesign, doe_size, sampler_name, bound)


def test_get_sampler():
    for type_id, s_name in enumerate(sampler_list):
        gen = get_Generator(s_name)
        assert isinstance(gen.sampler, sampler_type_list[type_id])


def test_custom_generator():
    for s_name in sampler_list:
        gen = get_Generator(s_name)
        for _ in range(doe_size):
            assert all([bound[1] > val > bound[0] for val in gen.custom_generator(None, None)])
        with pytest.raises(IndexError):
            gen.custom_generator(None, None)
