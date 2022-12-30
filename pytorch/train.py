import torch.optim as optim
from torch import nn
from data.timeframe_dataset import TimeframeDataset
from models.timeframe import TimeframeModel
from torch.utils.data import DataLoader
import torch

import sys
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dpath, epoch, batch_size, lr = sys.argv[1:]

    EPOCH = int(epoch)
    BATCH_SIZE = int(batch_size)
    LEARNING_RATE = float(lr)

    print(f"""
    ===
    EPOCH = {EPOCH}
    BATCH_SIZE = {BATCH_SIZE}
    LEARNING_RATE = {LEARNING_RATE}
    ===
    """)

    print(f"Reading dataset from {dpath}")
    dataset = TimeframeDataset(dpath)
    loader = DataLoader(dataset, BATCH_SIZE)
    val_loader = DataLoader(dataset.validation_set, BATCH_SIZE)
    print("Dataset Loaded")
    loss_criterion = nn.MSELoss()

    model = TimeframeModel(89)
    print("Model created")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []
    val_losses = []
    for epoch in range(1, EPOCH + 1):
        loss_data = 0
        model.train(True)
        for i, data in enumerate(loader):
            output = model(data['x'])

            loss = loss_criterion(output, data['y'].unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_data = loss.data
            print(f'Epoch {epoch}: {i * 100 / len(loader):.2f}%', end='\r')

        val_loss = 0
        model.eval()
        for i, data in enumerate(val_loader):
            output = model(data['x'])

            loss = loss_criterion(output, data['y'].unsqueeze(1))
            val_loss = loss.data
            print(f'Validation {epoch}: {i * 100 / len(val_loader):.2f}%', end='\r')

        print(f'Epoch {epoch}: loss = {loss_data} validation = {val_loss}' + ' '*10)
        losses.append(loss_data)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), f'checkpoints/epoch{EPOCH}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_timeframe.pth')
    plt.plot(range(1, len(losses) + 1), losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
