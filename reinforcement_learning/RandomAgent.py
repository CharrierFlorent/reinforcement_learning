import random
import numpy as np

class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def action(self, observation, reward):
        return random.choice(np.arange(-1.0,2.0,1.0))
