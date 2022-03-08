from agent import Agent
from world import World
import random

class NetworkAgent(Agent):
    def __init__(self, world: World, name:str, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1):
        super().__init__(world, qvals_filename, epsilon, alpha, gamma, decay_epsilon, decay_alpha)
        self.name = name
    def get_best_action(self) -> int:
        return super().get_best_action()
    def save(self, filename):
        return super().save(filename)
    def reset(self, reset_qvalues=False, reset_epsilon_to=0):
        return super().reset(reset_qvalues, reset_epsilon_to)
    

    def take_action(self):
        self.prev_state = self.world.state

        action = self.get_best_action()
        result = self.world.step(action)
        reward = result[0]
        self.reward += reward
        new_state = result[1]

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
        return new_state.get(self.name)
