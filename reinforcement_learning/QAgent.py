from QLearn import QLearn
import numpy as np

class QAgent():
    def __init__(self,action_space, nb_state):
        self.ai = None
        self.ai = QLearn(actions=action_space, alpha=0.1, gamma=0.999, epsilon=0.9)
        self.lastState = None
        self.lastAction = None
        self.nb_state = nb_state

    def calculate_state(self,obs):
        pos = -1
        speed = -1
        step = 1.8/self.nb_state
        step_speed = 0.14/self.nb_state
        j = 0
        for i in np.arange(-1.2,0.6,step):
            if min(i, i+step) <= obs[0] <= max(i, i+step):
                pos = j
            j += 1
        
        j = 0
        for i in np.arange(-0.07,0.07,step_speed):
            if min(i, i+step_speed) <= obs[1] <= max(i, i+step_speed):
                speed = j 
            j += 1

        return (pos,speed)

    def action(self, obs, reward):
        state = self.calculate_state(obs)   
        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, state, reward)

        action = self.ai.choose_action(state)
        self.lastState = state
        self.lastAction = action
        return action
