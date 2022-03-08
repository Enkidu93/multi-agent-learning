from world import World
from networkagent import NetworkAgent
from math import cos, sin, pi

R_CCW = 0
R_CW = 1
STEP = 2

class PredatorWorld(World):
    def __init__(self,agents:list[NetworkAgent]):
        self.dictionary = dict()
        self.agents = agents
        start_locs = [[20,20,45],[20,40,45],[40,20,45],[40,40,45]]
        i = 0
        for agent in self.agents:
            if i < 4:
                self.dictionary[agent.name] = start_locs[i]
                i+=1
            else:
                print("Too many agents...")
                break
        self.dictionary["Prey"] = [80,80]
        self.actions = [R_CCW, R_CW, STEP]
        self.episode_complete = False

    def reset(self):
        start_locs = [[20,20,45],[20,40,45],[40,20,45],[40,40,45]]
        i = 0
        for agent in self.agents:
            if i < 4:
                self.dictionary[agent.name] = start_locs[i]
                i+=1
            else:
                print("Too many agents...")
                break
        self.dictionary["Prey"] = [80,80]
        self.episode_complete = False
    
    def step(self, action:int, agent:NetworkAgent):
        if action == R_CW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] + 15)%360
            self.dictionary[agent.name] = new_state
        if action == R_CCW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] - 15)%360
            self.dictionary[agent.name] = new_state        
        if action == STEP:
            distance_stepped = 1 #TODO
            new_state = self.dictionary.get(agent.name)
            new_state[0] += distance_stepped*cos(new_state[2]*pi/180)
            new_state[1] += distance_stepped*sin(new_state[2]*pi/180)
        # MOVING PREY???  SEPARATE METHOD???
    
    def process(self, message_content):
        pass #TODO
