import torch.optim as optim
from torch import nn
from data.timeframe_dataset import TimeframeDataset
from models.timeframe import TimeframeModel
from torch.utils.data import DataLoader
import torch

import matplotlib.pyplot as plt


if __name__ == '__main__':
    EPOCH = 30
    BATCH_SIZE = 32

    loader = DataLoader(TimeframeDataset('dataset/avg_timeline.csv'), BATCH_SIZE)
    print("Dataset Loaded")
    loss_criterion = nn.BCELoss()

    model = TimeframeModel(89)
    print("Model created")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    losses = []
    for epoch in range(1, EPOCH + 1):
        loss_data = 0
        for i, data in enumerate(loader):
            output = model(data['x'])

            loss = loss_criterion(output, data['y'].unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_data = loss.data
            print(f'Epoch {epoch}: {i * 100 / len(loader):.2f}%', end='\r')
        print(f'Epoch {epoch}: {loss_data}' + ' '*10)
        losses.append(loss_data)

    torch.save(model.state_dict(), f'checkpoints/epoch{EPOCH}_batch{BATCH_SIZE}_timeframe.pth')
    plt.plot(range(1, len(losses) + 1), losses)
    plt.show()
