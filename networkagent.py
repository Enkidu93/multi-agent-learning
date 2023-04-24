from agent import Agent
from world import World
# from battle_royale import BattleRoyale
from world import World
from function_approximator import FunctionApproximator
import random
import numpy as np
import time
from machine import Machine
import json
from ast import literal_eval


class NetworkAgent(Agent):
    def __init__(self, world:World, name:str, qvals_filename=None, epsilon=0.01, alpha=0.01, gamma=1, decay_epsilon=1, decay_alpha=1, is_heuristic=False, replay_df=None, is_tab=False, use_ping=False, is_switching=False, high_med_low_models=None):
        super().__init__(world, qvals_filename, epsilon, alpha, gamma, decay_epsilon, decay_alpha)
        self.world = world
        self.name = name
        self.value_approximator = FunctionApproximator()
        self.has_model = False
        self.machine = None
        # self.action_count = {0:0,1:0,2:0,3:0}
        self.time_in_inference = 0
        self.times_used_cached = 0
        self.times_used_model = 0
        self.is_heuristic = is_heuristic
        self.replay_df = replay_df
        self.replay_df_index = 0
        self.is_tabular = is_tab
        self.new_states = 0
        self.total_states = 0
        self.use_ping = use_ping
        self.ys = list()
        self.xs = list()
        self.is_switching = is_switching
        if(is_switching):
            model_lowd = dict()
            model_highd = dict()
            model_medd = dict()
            self.models = {"high": model_highd, "low": model_lowd, "med": model_medd}
            if(high_med_low_models):
                with open(high_med_low_models[0],'r') as q:
                    data = json.loads(q.read())
                    temp_dict = dict((literal_eval(k),v) for k,v in data.items())
                    self.model_highd:dict[tuple,dict[int,float]] = dict()
                    for k, v in temp_dict.items():
                        for subk, subv in v.items():
                            self.model_highd[k][int(subk)] = float(subv)
                with open(high_med_low_models[1],'r') as q:
                    data = json.loads(q.read())
                    temp_dict = dict((literal_eval(k),v) for k,v in data.items())
                    self.model_medd:dict[tuple,dict[int,float]] = dict()
                    for k, v in temp_dict.items():
                        self.model_medd[k] = dict()
                        for subk, subv in v.items():
                            self.model_medd[k][int(subk)] = float(subv)
                with open(high_med_low_models[2],'r') as q:
                    data = json.loads(q.read())
                    temp_dict = dict((literal_eval(k),v) for k,v in data.items())
                    self.model_lowd:dict[tuple,dict[int,float]] = dict()
                    for k, v in temp_dict.items():
                        self.model_lowd[k] = dict()
                        for subk, subv in v.items():
                            self.model_lowd[k][int(subk)] = float(subv)
    
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
                curstate.append(float(action)==0)
                curstate.append(float(action)==1)
                curstate.append(float(action)==2)
                curstate.append(float(action)==3)
                in_data.append(curstate)
                out_data.append([value])
        x = np.array(in_data)
        y = np.array(out_data)
        return self.value_approximator.model.fit(x=[x],y=y,verbose=1)
    
    def save_memories(self):
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
        self.xs.append(x)
        self.ys += y
    
    def refit_based_on_memories(self):
        return self.value_approximator.model.fit(x=self.xs, y=self.ys, verbose=1)


    def get_best_action(self) -> int:

        if(self.is_switching):
            self.switch_models()
        
        rand = random.random()

        best_action = None

        if len(self.world.actions) == 0:
            return None 

        if self.replay_df:
            #TODO
            pass
        
        if self.is_heuristic:
            return self.world.getHeuristicBestActionFor(self)

        if rand <= self.epsilon:
            best_action = random.choice(self.world.actions)
            return best_action
        else:
            highest_q = -10_000
            tabular_actions = dict()
            if not self.is_tabular:
                tabular_actions = self.q_values.get(tuple(self.world.translateAbsoluteState(self)), None)
            else:
                tabular_actions = self.q_values.get(tuple(self.world.translateAbsoluteState(self)), None)
                if tabular_actions is None:
                    tabular_actions = {random.choice(self.world.actions):0}
                    # self.new_states += 1
            if (tabular_actions is None):
                action = random.choice(self.world.actions)
                value = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action==0, action==1, action==2, action==3]]), verbose=0)[0][0]
                self.q_values[tuple(self.world.translateAbsoluteState(self))] = {action : value}
                tabular_actions = self.q_values.get(tuple(self.world.translateAbsoluteState(self)))
            for action in self.world.actions:
                cur_q = -10_000
                tabular_q = 0
                if not self.is_tabular:
                    tabular_q = tabular_actions.get(action,None)
                else:
                    tabular_q = tabular_actions.get(action,0)
                if tabular_q is None and self.has_model:
                    start = time.time()
                    cur_q = self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action==0, action==1, action==2, action==3]]), verbose=0)[0][0]
                    self.time_in_inference += (time.time() - start)
                    self.times_used_model += 1
                else:
                    cur_q = tabular_q
                    self.times_used_cached += 1
                if cur_q is not None and cur_q > highest_q:
                    highest_q = cur_q
                    best_action = action

        if best_action is None:
            print("This shouldn't happen...")
            best_action = random.choice(self.world.actions)
        
        return best_action

    def save(self, filename):
        return super().save(filename)

    def reset(self, reset_qvalues=False, reset_epsilon_to=0, reset_state_count=False):
        # print("RESETTING", self.name)
        self.reward = 0
        self.prev_action = None
        self.prev_state = None
        if reset_epsilon_to:
            self.epsilon = reset_epsilon_to
        if reset_qvalues:
            self.q_values = dict()
        if reset_state_count:
            self.total_states = 0
            self.new_states = 0

    def take_action(self) -> tuple:
        self.prev_state = tuple(self.world.translateAbsoluteState(self))

        action = self.get_best_action()
        result = self.world.step(action,self)
        reward = result[0]
        self.reward += reward
        new_state = tuple(self.world.translateAbsoluteState(self))

        self.prev_action = action            

        if(self.epsilon == 2.0) or self.is_heuristic:
            return (self.name, self.world.dictionary.get(self.name), action, reward)
        

        randval = random.randint(0,len(self.world.actions)-1)

        best_next_q = -100_000 # may need to rechoose appropriate value
        # possible_next_actions = self.q_values.get(new_state,{randval:0}) #how often do I happen?
        # self.value_approximator.model.predict(np.array([self.world.translateAbsoluteState(self) + [action]]))[0][0]
        start = time.time()
        if not self.is_tabular:
            possible_next_actions = self.q_values.get(new_state,{randval:self.value_approximator.model.predict(np.array([list(new_state) + [randval==0, randval==1, randval==2, randval==3]]), verbose=0)[0][0]})
        else:
            possible_next_actions = self.q_values.get(new_state, None)
            if possible_next_actions is None:
                possible_next_actions = {randval:0}
                # self.new_states += 1

        self.time_in_inference += (time.time() - start)
        # find max for all a of Q(s_t+1, a)
        for a, q_value in possible_next_actions.items():
            if q_value > best_next_q:
                best_next_q = q_value

        # Put in placeholder values for new states

        if self.q_values.get(self.prev_state) is None:
            self.q_values[self.prev_state] = dict()
            self.new_states += 1
        
        if self.q_values[self.prev_state].get(self.prev_action) is None:
            start = time.time()
            if not self.is_tabular:
                self.q_values[self.prev_state][self.prev_action] = self.value_approximator.model.predict(np.array([list(self.prev_state) + [self.prev_action==0,self.prev_action==1,self.prev_action==2,self.prev_action==3]]), verbose=0)[0][0]
            else:
                self.q_values[self.prev_state][self.prev_action] = 0
                
            self.time_in_inference += (time.time() - start)
    
        # Q-learning value adjustment
        # self.q_values[self.prev_state][self.prev_action] = self.q_values.get(self.prev_state, {randval:0}).get(self.prev_action,{randval:0}) + self.alpha*(reward + self.gamma*(best_next_q) - self.q_values[self.prev_state][self.prev_action])
        self.q_values[self.prev_state][self.prev_action] = self.q_values.get(self.prev_state).get(self.prev_action) + self.alpha*(reward + self.gamma*(best_next_q) - self.q_values[self.prev_state][self.prev_action])

        self.total_states += 1

        # return update
        return (self.name, self.world.dictionary.get(self.name), action, reward)

    def get_ping_time_to(self, other_agent):
        conn = self.machine.connections.get(other_agent.machine.name, None)
        assert(conn is not None)
        return 2*conn.generate_delay()
    
    def test_delay(self):
        count = 0
        delay = 0
        for conn in self.machine.connections.values():
            count += 1
            delay += conn.generate_delay()
        return delay/count
    
    def switch_models(self):
        assert(self.is_switching)
        avg_delay_measure = self.test_delay()
        # print("Delay was", avg_delay_measure)
        name = "low"
        if avg_delay_measure > 35:
            name = "med"
        if avg_delay_measure > 75:
            name = "high"
        # print(name)
        self.q_values = self.models.get(name)


