from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame, split: float, seed: int, normalize=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = df['label']
    df = df.drop('label', axis=1)
    for col in df.columns:
        if df[col].dtype not in ('int', 'float'):
            y = pd.get_dummies(df[col], prefix=col)
            df = df.drop(col, axis=1)
            df = df.join(y)
        else:
            if normalize:
                r = df[col].max() - df[col].min() if df[col].max() > df[col].min() else df[col].max()
                r = r if r > 0 else 1
                df[col] = (df[col] - df[col].min()) / r
    X = df.to_numpy()
    y = labels.to_numpy()
    return train_test_split(X, y, train_size=split, shuffle=True, random_state=seed)
