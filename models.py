import numpy as np


class Interaction:
    def __init__(self, observation: np.ndarray, action: int, reward: float):
        self.observation = observation
        self.action = action
        self.reward = reward
