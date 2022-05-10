from agent import Agent
from world import World
# from battle_royale import BattleRoyale
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
    
    def __str__(self):
        out = self.name + " "
        for ele in self.world.dictionary.get(self.name):
            out += str(round(ele,3)) + " "
        return out
    
    def refit_model(self):
        in_data = []
        out_data = []
        for state, subd in self.q_values.items():
            for action, value in subd.items():
                curstate = []
                for e in state:
                    curstate.append(e) # I know it's bad...but it's worth a shot
                curstate.append(float(action))
                in_data.append(curstate)
                out_data.append([value])
        x = np.array(in_data)
        y = np.array(out_data)
        self.value_approximator.model.fit(x=[x],y=y,verbose=0)


    def get_best_action(self) -> int:
        
        rand = random.random()

        best_action = None

        if len(self.world.actions) == 0:
            return None 

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            highest_q = -1000
            tabular_actions = self.q_values.get(tuple(self.world.translateAbsoluteState(self)),None)
            if tabular_actions is not None:
                for action in self.world.actions:
                    cur_q = -1000
                    tabular_q = tabular_actions.get(action,None)
                    if tabular_q is None and self.has_model:
                        cur_q = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action]]))[0][0]
                    else:
                        cur_q = tabular_q
                    if cur_q is not None and cur_q > highest_q:
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
        new_state = tuple(self.world.translateAbsoluteState(self))

        self.prev_action = action

        randval = random.randint(0,len(self.world.actions)-1)

        best_next_q = -100 # may need to rechoose appropriate value
        possible_next_actions = self.q_values.get(new_state,{randval:0})

        # find max for all a of Q(s_t+1, a)
        for action, q_value in possible_next_actions.items():
            if q_value > best_next_q:
                best_next_q = q_value

        # Put in placeholder values for new states

        if self.q_values.get(self.prev_state) is None:
            self.q_values[self.prev_state] = dict()
        
        if self.q_values[self.prev_state].get(self.prev_action) is None:
            self.q_values[self.prev_state][self.prev_action] = 0

        # Q-learning value adjustment
        self.q_values[self.prev_state][self.prev_action] = self.q_values.get(self.prev_state, {randval:0}).get(self.prev_action,{randval:0}) + self.alpha*(reward + self.gamma*(best_next_q) - self.q_values[self.prev_state][self.prev_action])

        # return update
        return (self.name, self.world.dictionary.get(self.name))


