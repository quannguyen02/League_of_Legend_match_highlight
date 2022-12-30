from torch import nn
from data.timeframe_dataset import TimeframeDataset
from models.timeframe import TimeframeModel
from torch.utils.data import DataLoader
import torch

import sys
import math
import matplotlib.pyplot as plt


if __name__ == '__main__':
    model_path, data_path = sys.argv[1:]

    dataset = TimeframeDataset(data_path, split=1.0)
    loader = DataLoader(dataset, 1)
    print("Dataset Loaded")

    model = TimeframeModel(89)
    model.load_state_dict(torch.load(model_path))
    print("Model created")

    acc = [[0, 0] for _ in range(11)]
    corr = []
    model.eval()

    match_line = []
    for i, data in enumerate(loader):
        x = data['x']
        p = int(x[0][0] * 10)
        output = model(x)[0][0].data
        if torch.round(output) == data['y'][0]:
            acc[p][0] += 1
        acc[p][1] += 1

        if p == 10:
            j = len(match_line) - 1
            while j >= 0 and match_line[j] == data['y'][0]:
                j -= 1
            if j + 1 < 11:
                corr.append(int((j + 1) * 100 / (len(match_line) - 1)))
            match_line.clear()     
        else:
            match_line.append(torch.round(output))

    plt.hist(corr, facecolor = '#2ab0ff', edgecolor='#169acf')
    plt.xlabel("Time took until the result is settled")
    plt.ylabel("Number of matches")
    plt.show()

    xs = [i * 10 for i in range(0, 11)]
    ys = [a[0] / a[1] for a in acc]
    # ci = [1.96 * math.sqrt((a * (1 - a)) / n[1]) for a, n in zip(ys, acc)]
    # print(ci)
    plt.bar(xs, ys, facecolor = '#2ab0ff', edgecolor='#169acf', width=10)
    for x, v in zip(xs, ys):
        plt.text(x - 5, v + 0.01, f'{v:.2f}', color='black', fontweight='bold')
    plt.xlabel("Game Percentage")
    plt.ylabel("Proportion of correctly predicted result")
    plt.show()




