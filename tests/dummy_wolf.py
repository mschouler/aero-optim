import argparse
import time

from random import randrange


def main():
    """
    This program mocks the behaviour of a WOLF execution.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-in", "--input", type=str, help="meshfile")
    _ = parser.parse_args()

    sleep: float = randrange(10) / 10.
    time.sleep(sleep)

    # write dummy residual file
    fname: str = "residual.dat"
    output = ["# Iter    ResTot",
              "1    0.1",
              "2    0.2",
              "3    0.3",
              "4    0.4",
              "5    0.5"]
    with open(fname, 'w') as ftw:
        ftw.write("\n".join(output))
        print(f"output saved to {fname}")

    # write dummy aerocoef file from simulation cid
    fname = "aerocoef.dat"
    output = ["# Iter    CD    CL    ResCD    ResCL",
              f"1    {0.1 + sleep / 10.}    {0.3 + sleep / 10.}    0.01    0.01",
              f"2    {0.1 + sleep / 10.}    {0.3 + sleep / 10.}    0.01    0.01",
              f"3    {0.1 + sleep / 10.}    {0.3 + sleep / 10.}    0.01    0.01",
              f"4    {0.1 + sleep / 10.}    {0.3 + sleep / 10.}    0.01    0.01",
              f"5    {0.1 + sleep / 10.}    {0.3 + sleep / 10.}    0.01    0.01"]
    with open(fname, 'w') as ftw:
        ftw.write("\n".join(output))
        print(f"output saved to {fname}")

    # empty wall file
    fname = "wall.dat"
    output = ["# Iter    DummyEntry"]
    with open(fname, 'w') as ftw:
        ftw.write("\n".join(output))
        print(f"output saved to {fname}")


if __name__ == "__main__":
    main()
