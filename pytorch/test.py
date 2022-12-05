from data.timeframe_dataset import TimeframeDataset
from torch.utils.data import DataLoader
from models.timeframe import TimeframeModel
import torch

import numpy as np


if __name__ == '__main__':
    test_set = TimeframeDataset('dataset/avg_timeline.csv')
    print("Dataset Loaded")

    model = TimeframeModel(89)
    model.load_state_dict(torch.load('checkpoints/model.pth'))
    print("Model created")
    model.eval()

    correct_pred = 0
    total = 0
    with torch.no_grad():
        for x, y in zip(test_set.x_test, test_set.y_test):
            output = model(torch.Tensor(x[np.newaxis, :]))[0].numpy()
            pred = round(output[0])
            real = y
            if pred == real:
                correct_pred += 1
            total += 1
    print(f'Predicted {correct_pred} out of {total}: '
          f'{correct_pred * 100 // total}%')
