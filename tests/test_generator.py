import pytest

from scipy.stats import qmc
from aero_optim.optim.generator import Generator

seed: int = 123
ndesign: int = 4
doe_size: int = 16
bound: tuple[float, float] = (-1., 1.)
sampler_list: list[str] = Generator.sampler_list
sampler_type_list: list[type] = [qmc.LatinHypercube, qmc.Halton, qmc.Sobol, type(None)]


def get_Generator(sampler_name: str, custom_doe: str = "") -> Generator:
    return Generator(seed, ndesign, doe_size, sampler_name, bound, custom_doe)


def test_get_sampler():
    # qmc standard samplers
    for type_id, s_name in enumerate(sampler_list[:-1]):
        gen = get_Generator(s_name)
        assert isinstance(gen.sampler, sampler_type_list[type_id])
    # custom sampler
    gen = get_Generator("custom", "tests/extras/dummy_doe.txt")
    assert isinstance(gen.sampler, sampler_type_list[-1])


def test_custom_generator():
    # qmc standard samplers
    for s_name in sampler_list[:-1]:
        gen = get_Generator(s_name)
        assert all([bound[1] > val > bound[0] for elt in gen._pymoo_generator() for val in elt])
        for _ in range(doe_size):
            assert all([bound[1] > val > bound[0] for val in gen._ins_generator(None, None)])
        with pytest.raises(IndexError):
            gen._ins_generator(None, None)
    # custom sampler
    gen = get_Generator("custom", "tests/extras/dummy_doe.txt")
    assert all([bound[1] > val > bound[0] for elt in gen._pymoo_generator() for val in elt])
    for _ in range(doe_size):
        assert all([bound[1] > val > bound[0] for val in gen._ins_generator(None, None)])
    with pytest.raises(IndexError):
        gen._ins_generator(None, None)
