# import numpy as
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset


class TimeframeDataset(Dataset):
    def __init__(self, dataroot, split=0.7):
        self.df = pd.read_csv(dataroot)
        self.labels = self.df['label'].to_numpy('float32')
        self.df = self.df.drop('label', axis=1)
        self.matches = self.df.to_numpy(dtype='float32')

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.matches, self.labels, train_size=split, shuffle=True)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return {'y': self.y_train[idx], 'x': self.x_train[idx] }
