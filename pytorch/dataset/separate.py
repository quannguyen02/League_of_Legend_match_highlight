import json
from glob import glob
import os
import random


if __name__ == "__main__":
    fpaths = glob("timelines/*.json")
    random.shuffle(fpaths)

    split = 0.1
    idx = int(len(fpaths) * split)

    for fpath in fpaths[:idx]:
        os.system(f"mv {fpath} test_{fpath}")
