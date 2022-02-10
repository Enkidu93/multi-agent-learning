from agent import Agent
from connect_four import ConnectFour
import random

class CFAgent(Agent):
    def __init__(self, world: ConnectFour, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1):
        super().__init__(world, qvals_filename, epsilon, alpha, gamma, decay_epsilon, decay_alpha)
        if self.world.players.get(1,None) is None: 
            self.world.players[1] = self
        else:
            self.world.players[-1] = self

    def reset(self, reset_qvalues=False, reset_epsilon_to=0):
        super().reset(reset_qvalues, reset_epsilon_to)
    
    def get_best_action(self) -> int:

        rand = random.random()

        best_action = None

        if len(self.world.actions) == 0:
            return None 

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            possible_actions = self.q_values.get(tuple(tuple(e) for e in self.world.state),{0:0})

            highest_q = -100 # may need to rechoose appropriate value

            for action, q_value in possible_actions.items():
                if action not in self.world.actions:
                    continue
                if q_value > highest_q:
                    highest_q = q_value
                    best_action = action
        
        self.epsilon *= self.decay_epsilon
        self.alpha *= self.decay_alpha

        if best_action is None:
            best_action = random.choice(self.world.actions)
        
        return best_action
    
    def take_action(self):
        self.prev_state = tuple(tuple(e) for e in self.world.state)

        action = self.get_best_action()
        if action is None:
            self.world.episode_complete = True
            return
        result = self.world.step(action)
        reward = result[0]
        self.reward += reward
        new_state = tuple(tuple(e) for e in result[1])

        self.prev_action = action

        best_next_q = 0 # may need to rechoose appropriate value
        possible_next_actions = self.q_values.get(new_state,None)


        # find max for all a of Q(s_t+1, a)
        if possible_next_actions is not None:
            for action, q_value in possible_next_actions.items():
                if q_value > best_next_q:
                    best_next_q = q_value

        # Put in placeholder values for new states

        if self.q_values.get(self.prev_state) is None:
            self.q_values[self.prev_state] = dict()
        
        if self.q_values[self.prev_state].get(self.prev_action) is None:
            self.q_values[self.prev_state][self.prev_action] = 0

        # Q-learning value adjustment
        self.q_values[self.prev_state][self.prev_action] = self.q_values.get(self.prev_state).get(self.prev_action) + self.alpha*(reward + self.gamma*(best_next_q) - self.q_values[self.prev_state][self.prev_action])
        
    def save(self, filename):
        return super().save(filename)

