from world import World
import random
import json

class Agent:

    def __init__(self, world:World,qvals_filename=None,epsilon=0.01,alpha=0.01,gamma=1,decay_epsilon=1,decay_alpha=1):
        self.world = world
        self.reward = 0
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.decay_epsilon = decay_epsilon
        self.decay_alpha = decay_alpha
        self.prev_action = None
        self.prev_state = None
        if qvals_filename is None:
            self.q_values:dict[tuple,dict[int,float]] = dict()
        else:
            self.q_values = json.load(qvals_filename)

    def reset(self,reset_qvalues=False,reset_epsilon_to=0):
        self.world.reset()
        self.reward = 0
        self.prev_action = None
        self.prev_state = None
        if reset_epsilon_to:
            self.epsilon = reset_epsilon_to
        if reset_qvalues:
            self.q_values = dict()

    def get_best_action(self) -> int:

        rand = random.random()

        best_action = 0

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            possible_actions = self.q_values.get(self.world.state,{0:0})

            highest_q = -100 # may need to rechoose appropriate value

            for action, q_value in possible_actions.items():
                if q_value > highest_q:
                    highest_q = q_value
                    best_action = action
        
        self.epsilon *= self.decay_epsilon
        self.alpha *= self.decay_alpha
        
        return best_action
    
    def take_action(self):
        self.prev_state = self.world.state

        action = self.get_best_action()
        result = self.world.step(action)
        reward = result[0]
        self.reward += reward
        new_state = result[1]

        self.prev_action = action

        best_next_q = -100 # may need to rechoose appropriate value
        possible_next_actions = self.q_values.get(new_state,{0:0})


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
        self.q_values[self.prev_state][self.prev_action] = self.q_values.get(self.prev_state, {2:0}).get(self.prev_action,{0:0}) + self.alpha*(reward + self.gamma*(best_next_q) - self.q_values[self.prev_state][self.prev_action])

    def save(self, filename):
        with open(filename, "w") as outfile:
            data = dict((str(k),v) for k,v in self.q_values.items())
            json.dump(data, outfile)
