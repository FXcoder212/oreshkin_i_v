"""Wrap the success model to shift logit of class-A (and optionally B) attempts."""
import numpy as np
from sim2 import SFH
CI = SFH.index('cls_i')


class Shift:
    def __init__(self, b, dA, dB=0.0):
        self.b, self.dA, self.dB = b, dA, dB

    def predict(self, X, num_threads=1):
        p = np.clip(self.b.predict(X, num_threads=num_threads), 1e-5, 1 - 1e-5)
        lg = np.log(p / (1 - p)) + np.where(X[:, CI] == 0, self.dA, np.where(X[:, CI] == 1, self.dB, 0.0))
        return 1 / (1 + np.exp(-lg))
