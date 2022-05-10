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
        self.q_values:dict[tuple,dict[int,float]] = dict()
    
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

        state = tuple(self.world.translateAbsoluteState(self))

        if len(self.world.actions) == 0:
            return None 

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
        else:
            highest_q = -100
            for action in self.world.actions:
                state_dict = self.q_values.get(state, None)
                cached_q = state_dict.get(action, None) if state_dict is not None else None
                if cached_q is not None:
                    cur_q = self.q_values.get(state).get(action)
                else:
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

    def reset(self, reset_qvalues=True, reset_epsilon_to=0):
        self.reward = 0
        self.prev_action = None
        self.prev_state = None
        if reset_epsilon_to:
            self.epsilon = reset_epsilon_to
        if reset_qvalues:
            self.q_values:dict[tuple,dict[int,float]] = dict()

    def take_action(self) -> tuple[str,list]:
        self.prev_state = tuple(self.world.translateAbsoluteState(self))

        action = self.get_best_action()
        result = self.world.step(action,self)
        reward = result[0]
        self.reward += reward

        new_state = tuple(self.world.translateAbsoluteState(self))


        self.prev_action = action

        best_next_q = -1000 # may need to rechoose appropriate value

        # printAction(action)
        # print(self.epsilon)

        # find max for all a of Q(s_t+1, a)
        for action in self.world.actions:
            q_value = best_next_q
            state_dict = self.q_values.get(new_state, None)
            cached_q = state_dict.get(action, None) if state_dict is not None else None
            if cached_q is not None:
                q_value = self.q_values.get(new_state).get(action)
            else:
                q_value = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action]]))[0][0]
                self.q_values[new_state] = dict()
                self.q_values[new_state][action] = q_value
            if q_value > best_next_q:
                best_next_q = q_value

        old_q = self.value_approximator.model.predict(np.array([list(self.prev_state)  + [self.prev_action]]))[0][0]
        if self.q_values.get(self.prev_state, None) is not None and self.q_values.get(self.prev_state).get(self.prev_action, None) is not None:
            # print("CACHED")
            old_q = self.q_values.get(self.prev_state).get(self.prev_action)
        else:
            old_q = self.value_approximator.model.predict(np.array([list(self.prev_state)  + [self.prev_action]]))[0][0]

        # Q-learning value adjustment
        self.last_memory = [self.world.translateAbsoluteState(self) + [self.prev_action], old_q + self.alpha*(reward + self.gamma*(best_next_q) - old_q)]
        # self.q_values[new_state][self.prev_action] = old_q + self.alpha*(reward + self.gamma*(best_next_q) - old_q)

        # self.refit_model()

        self.memories[0].append(np.array(self.last_memory[0]))
        self.memories[1].append(np.array(self.last_memory[1]))


        # return update
        return (self.name, self.world.dictionary.get(self.name))

def printAction(action):
    if action == 0:
        print("CCW")
    if action == 1:
        print("CW")
    if action == 2:
        print("STEP")
    if action == 3:
        print("ATTACK")

