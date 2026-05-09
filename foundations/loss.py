import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        N=len(y_true)
        loss=0
        e=10**(-7)
        for i in range(N):
            yi=y_true[i]
            pi=y_pred[i]+e
            loss+=yi*np.log(pi)+(1-yi)*np.log(1-pi)
        return round(loss*(-1/N),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss=0
        e=10**(-7)
        N=len(y_true)
        C=len(y_true[0])
        for i in range(N):
            for c in range(C):
                yic=y_true[i][c]
                pic=y_pred[i][c]+e
                loss+=yic*np.log(pic)

        return round(loss*(-1/N),4)
