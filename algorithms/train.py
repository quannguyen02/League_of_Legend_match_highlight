import sys
import pickle

from data.timeline_dataset import TimelineDataset
from dataset.preprocess import preprocess

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor 
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/ (1 + np.exp(-x))


if __name__ == "__main__":
    fname, seed = sys.argv[1:]
    df = pd.read_csv(fname)
    X_train, X_test, y_train, y_test = preprocess(df, 0.75, int(seed), normalize=False)

    clf =  MLPRegressor((100, 50, 30), max_iter=500, learning_rate_init=0.01, random_state=int(seed), verbose=True)
    clf.fit(X_train, y_train)

    pred = sigmoid(clf.predict(X_test)) > 0.5

    print(accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))

    with open(f'checkpoints/mlp_s{seed}.pkl', 'wb') as f:
        pickle.dump(clf, f)

    t = TimelineDataset(fname)

    match = t[132]
    result, frames = match['y'], match['x']
    pred = clf.predict(frames)
    plt.plot(frames[:,0], sigmoid(pred))
    print(f"Result: {result}")
    plt.show()
