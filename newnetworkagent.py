from agent import Agent
from world import World
from world import World
from function_approximator import FunctionApproximator
import random
import numpy as np

class NetworkAgent(Agent):
    def __init__(self, world:World, name:str, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1):
        super().__init__(world, qvals_filename, epsilon, alpha, gamma, decay_epsilon, decay_alpha)
        self.world = world
        self.name = name
        self.value_approximator = FunctionApproximator()
        self.has_model = False
        self.last_memory = []
        self.memories = [[],[]]
    
    def __str__(self):
        out = self.name + " "
        for ele in self.world.dictionary.get(self.name):
            out += str(round(ele,3)) + " "
        return out
    
    def refit_model(self):
        # in_data = [np.array(self.last_memory[0])]
        # out_data = [self.last_memory[1]]
        # x = np.array(in_data)
        # y = np.array(out_data)
        in_data = self.memories[0]
        out_data = self.memories[1]
        x = np.array(in_data)
        y = np.array(out_data)
        self.value_approximator.model.fit(x=x,y=y,verbose=0)
        self.memories = [[],[]]


    def get_best_action(self) -> int:
        
        rand = random.random()

        best_action = None

        if len(self.world.actions) == 0:
            return None 

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            highest_q = -100
            for action in self.world.actions:
                cur_q = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action]]))[0][0]
                if cur_q > highest_q:
                    highest_q = cur_q
                    best_action = action
       
        self.epsilon *= self.decay_epsilon
        self.alpha *= self.decay_alpha

        if best_action is None:
            best_action = random.choice(self.world.actions)
        
        return best_action

    def save(self, filename):
        return super().save(filename)

    def reset(self, reset_qvalues=False, reset_epsilon_to=0):
        self.reward = 0
        self.prev_action = None
        self.prev_state = None
        if reset_epsilon_to:
            self.epsilon = reset_epsilon_to
        if reset_qvalues:
            self.q_values = dict()

    def take_action(self) -> tuple[str,list]:
        self.prev_state = tuple(self.world.translateAbsoluteState(self))

        action = self.get_best_action()
        result = self.world.step(action,self)
        reward = result[0]
        self.reward += reward

        self.prev_action = action

        best_next_q = -100 # may need to rechoose appropriate value

        # find max for all a of Q(s_t+1, a)
        for action in self.world.actions:
            q_value = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action]]))[0][0]
            if q_value > best_next_q:
                best_next_q = q_value

        # Q-learning value adjustment
        old_q = self.value_approximator.model.predict(np.array([list(self.prev_state)  + [self.prev_action]]))[0][0]
        self.last_memory = [self.world.translateAbsoluteState(self) + [self.prev_action], old_q + self.alpha*(reward + self.gamma*(best_next_q) - old_q)]

        # self.refit_model()

        self.memories[0].append(np.array(self.last_memory[0]))
        self.memories[1].append(np.array(self.last_memory[1]))


        # return update
        return (self.name, self.world.dictionary.get(self.name))


