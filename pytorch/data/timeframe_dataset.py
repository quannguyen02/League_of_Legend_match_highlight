# import numpy as
import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset


class TimeframeValidationSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {'y': self.Y[idx], 'x': self.X[idx] }


class TimeframeDataset(Dataset):
    def __init__(self, dataroot, split=0.7):
        self.df = pd.read_csv(dataroot)
        self.labels = self.df['label'].to_numpy('float32')
        self.df = self.df.drop('label', axis=1)
        self.matches = self.df.to_numpy(dtype='float32')
        if split < 1:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.matches, self.labels, train_size=split, shuffle=True)
        else:
            self.x_train = self.matches
            self.y_train = self.labels
    @property
    def validation_set(self):
        return TimeframeValidationSet(self.x_test, self.y_test)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return {'y': self.y_train[idx], 'x': self.x_train[idx] }
