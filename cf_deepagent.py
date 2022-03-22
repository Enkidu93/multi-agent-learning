from mimetypes import init
from textwrap import indent

from sympy import Q
from cf_agent import CFAgent
from connect_four import ConnectFour
from value_aproximator import CFValueApproximator
import numpy as np
import random

class DeepCFAgent(CFAgent):

    def __init__(self, world: ConnectFour, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1):
        super().__init__(world, qvals_filename, epsilon, alpha, gamma, decay_epsilon, decay_alpha)
        self.value_approximator = CFValueApproximator()
        self.has_model = False

    def reset(self, reset_qvalues=False, reset_epsilon_to=0):
        return super().reset(reset_qvalues, reset_epsilon_to)
    
    def save(self, filename):
        return super().save(filename)

    def get_best_action(self) -> int:

        rand = random.random()

        best_action = None

        if len(self.world.actions) == 0:
            return None 

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            highest_q = -100
            tabular_actions = self.q_values.get(tuple(tuple(e) for e in self.world.state),None)
            if tabular_actions is not None:
                for action in self.world.actions:
                    cur_q = -100
                    tabular_q = tabular_actions.get(action,None)
                    if tabular_q is None and self.has_model:
                        cur_q = self.value_approximator.model.predict([np.array([self.world.state]),np.array([list(0 if n!=action else 1 for n in range(7))])])[0][0]
                    else:
                        cur_q = tabular_q
                    if cur_q is not None and cur_q > highest_q:
                        highest_q = cur_q
                        best_action = action
        

            # if not self.has_model:
            #     possible_actions = self.q_values.get(tuple(tuple(e) for e in self.world.state),{0:0}) #CHANGE\/

            #     highest_q = -100 # may need to rechoose appropriate value

            #     for action, q_value in possible_actions.items():
            #         if action not in self.world.actions:
            #             continue
            #         if q_value > highest_q:
            #             highest_q = q_value
            #             best_action = action
            
            # else:
            #     highest_q = -100

            #     for action in self.world.actions:
            #         cur_q = self.value_approximator.model.predict([np.array([self.world.state]),np.array([list(0 if n!=action else 1 for n in range(7))])])[0][0]
            #         if cur_q > highest_q:
            #             best_action = action
       
        self.epsilon *= self.decay_epsilon
        self.alpha *= self.decay_alpha

        if best_action is None:
            best_action = random.choice(self.world.actions)
        
        return best_action
    
    def take_action(self):
        return super().take_action()
    
    def refit_model(self):
        data = DeepCFAgent.convert(self.q_values)
        self.value_approximator.model.fit([data[0],data[1]], data[1],verbose=0) #[2]?
        self.has_model = True
    
    @staticmethod
    def convert(q_values):
        in_data_1 = []
        in_data_2 = []
        out_data = []
        for state, subd in q_values.items():
            for action, value in subd.items():
                in_data_1.append(state)
                in_data_2.append(list(0 if n!=action else 1 for n in range(7)))
                out_data.append(value)
        return np.array(in_data_1), np.array(in_data_2), np.array(out_data)

