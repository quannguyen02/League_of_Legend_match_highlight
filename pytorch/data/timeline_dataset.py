import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class TimelineDataset(Dataset):
    def __init__(self, dataroot):
        self.matches = []
        add_x = self.matches.append
        self.df = pd.read_csv(dataroot)
        self.labels = self.df['label']
        self.df = self.df.drop('label', axis=1)
        percs = self.df['game_percentage']
        # self.df = self.df.drop('game_percentage', axis=1)
        s, e = 0, 0
        for p, row in zip(percs, self.df.iloc):
            if p == 0:
                s = e
            if p == 1:
                add_x(self.df.iloc[s:e+1].to_numpy(dtype='float32'))
            e += 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'y': self.labels[idx], 'x': np.array(self.matches[idx]) }
