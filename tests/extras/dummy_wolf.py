import argparse
import os
import random
import re
import time


def get_CD_CL(gid: int, cid: int) -> list[float]:
    return [0.1 + (gid**cid + cid + gid) / 100. + cid / 200, 0.3 + (gid + cid) / 100.]


def main():
    """
    This program mocks the behaviour of a WOLF execution.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-in", "--input", type=str, help="meshfile")
    _ = parser.parse_args()

    sleep = random.uniform(0, 1)
    time.sleep(sleep)

    cwd = os.getcwd()
    gid, cid = re.findall(r'\d+', cwd.split('/')[-1])
    CD, CL = get_CD_CL(int(gid), int(cid))

    # write dummy residual file
    fname = "residual.dat"
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
              f"1    {CD}    {CL}    0.01    0.01",
              f"2    {CD}    {CL}    0.01    0.01",
              f"3    {CD}    {CL}    0.01    0.01",
              f"4    {CD}    {CL}    0.01    0.01",
              f"5    {CD}    {CL}    0.01    0.01"]
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
