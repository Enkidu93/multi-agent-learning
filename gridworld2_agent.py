from agent import Agent
from grid_world2 import GridWorld2
import random
import json
from ast import literal_eval
class GW2Agent(Agent):
    
    def __init__(self, world: GridWorld2, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1):
        self.world = world
        if self.world.players.get("A",None) is None:
            self.player_val = "A"
            self.world.players["A"] = self
        elif self.world.players.get("B",None) is None:
            self.player_val = "B" 
            self.world.players["B"] = self
        elif self.world.players.get("C",None) is None:
            self.player_val = "C"   
            self.world.players["C"] = self     
        elif self.world.players.get("D",None) is None:
            self.player_val = "D"  
            self.world.players["D"] = self    
        else:
            print("This game is full...no more agents can be added.")
            self.player_val = None
            return None
        self.state = (self.world.state.get(self.player_val), tuple(sorted((v for k,v in self.world.state.items() if k!=self.player_val))))
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
            with open(qvals_filename,'r') as q:
                data = json.loads(q.read())
                self.q_values = dict((literal_eval(k),v) for k,v in data.items())
    
    
    def reset(self, reset_qvalues=False, reset_epsilon_to=0):
        return super().reset(reset_qvalues, reset_epsilon_to)

    def get_best_action(self) -> int:

        rand = random.random()

        best_action = 0

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            possible_actions = self.q_values.get(self.state,{0:0})

            highest_q = -100 # may need to rechoose appropriate value

            for action, q_value in possible_actions.items():
                if q_value > highest_q:
                    highest_q = q_value
                    best_action = action
        
        self.epsilon *= self.decay_epsilon
        self.alpha *= self.decay_alpha
        
        return best_action

    def take_action(self):
        self.prev_state = self.state

        action = self.get_best_action()
        result = self.world.step(action)
        reward = result[0]
        self.reward += reward
        self.updateState()
        new_state = self.state

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
        return super().save(filename)
    
    def updateState(self):
        self.state = (self.world.state.get(self.player_val), tuple(sorted((v for k,v in self.world.state.items() if k!=self.player_val))))
