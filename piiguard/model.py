"""Multinomial logistic regression in plain NumPy (full batch Adam, L2).

Kept deliberately simple: the point of the project is the uncertainty layer
on top, and conformal guarantees hold for any score function, however good
or bad. A small, transparent model makes the honest failure modes visible.
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    def __init__(self, n_classes: int, l2: float = 1e-4, lr: float = 0.05,
                 epochs: int = 400, seed: int = 0):
        self.k = n_classes
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.W = None
        self.b = None
        self.mu = None
        self.sd = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftmaxRegression":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        Xs = self._prep(X)
        n, d = Xs.shape
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, size=(d, self.k))
        self.b = np.zeros(self.k)
        Y = np.eye(self.k)[y]
        mW = np.zeros_like(self.W); vW = np.zeros_like(self.W)
        mb = np.zeros_like(self.b); vb = np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            P = softmax(Xs @ self.W + self.b)
            G = (P - Y) / n
            gW = Xs.T @ G + self.l2 * self.W
            gb = G.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW**2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb**2
            self.W -= self.lr * (mW / (1 - b1**t)) / (np.sqrt(vW / (1 - b2**t)) + eps)
            self.b -= self.lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self._prep(X) @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
