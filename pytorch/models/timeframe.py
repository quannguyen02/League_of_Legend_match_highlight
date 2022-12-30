import torch.nn as nn


class TimeframeModel(nn.Module):
    def __init__(self, in_feature):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feature, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
