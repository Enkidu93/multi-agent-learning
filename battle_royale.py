from world import World
from networkagent import NetworkAgent
from math import cos, sin, pi, sqrt, atan2
import numpy as np

R_CCW = 0
R_CW = 1
STEP = 2
ATTACK = 3

class BattleRoyale(World):
    def __init__(self,agents:list[NetworkAgent]):
        self.dictionary = dict()
        self.agents = agents
        start_locs = [[sqrt(30)/2,0,pi],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),5*pi/3],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),2*pi/3]]
        i = 0
        for agent in self.agents:
            if i < 4:
                self.dictionary[agent.name] = start_locs[i]
                i+=1
            else:
                print("Too many agents...")
                break
        self.actions = [R_CCW, R_CW, STEP, ATTACK]

    def reset(self):
        start_locs = [[sqrt(30)/2,0,pi],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),5*pi/3],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),2*pi/3]]
        i = 0
        for agent in self.agents:
            if i < 3:
                self.dictionary[agent.name] = start_locs[i]
                i+=1
            else:
                print("Too many agents...")
                break
    
    def step(self, action:int, agent:NetworkAgent):
        reward = -0.1
        if action == R_CW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] + pi/12)%(2*pi)
            self.dictionary[agent.name] = new_state
        if action == R_CCW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] - pi/12)%(2*pi)
            self.dictionary[agent.name] = new_state        
        if action == STEP:
            distance_stepped = 1 #Keep constant for now
            new_state = self.dictionary.get(agent.name)
            new_state[0] += distance_stepped*cos(new_state[2])
            new_state[1] += distance_stepped*sin(new_state[2])
            self.dictionary[agent.name] = new_state        
        if action == ATTACK:
            # start with no negative reward for dying
            self_pos = self.dictionary.get(agent.name)
            self_x = self_pos[0]
            self_y = self_pos[1]
            self_theta = self_pos[2]

            hit = False
            
            for a in self.agents:
                a_abs_pos = self.dictionary.get(a.name)
                a_x = a_abs_pos[0]
                a_y = a_abs_pos[1]
                if a != agent and sqrt((a_x - self_x)**2 + (a_y - self_y)**2)<=2 and ((atan2((a_y-self_y),(a_x-self_x)))<=(self_theta + pi/8) or (atan2((a_y-self_y),(a_x-self_x)))>=(self_theta - pi/8)):
                    start_locs = [[sqrt(30)/2,0,pi],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),5*pi/3],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*cos(2*pi/3),2*pi/3]]
                    self.dictionary[a.name] = start_locs[self.agents.index(a)]
                    reward = 5 if not hit else 10
                    hit = True

            if not hit:
                reward = -0.05

        return (reward, self.dictionary)
    
    def process(self, message_content:tuple[str,list]):
        agent_name = message_content[0]
        agent_state = message_content[1]
        self.dictionary[agent_name] = agent_state
    
    @staticmethod
    def isOutOfBounds(x,y):
        if x**2 + y**2 > 30:
            return False
        return True
    
    def translateAbsoluteState(self,agent:NetworkAgent)->list:
        agent_abs_position = self.dictionary.get(agent.name)
        agent_x = agent_abs_position[0]
        agent_y = agent_abs_position[1]
        agent_theta = agent_abs_position[2]

        wall_r = sqrt(30) - sqrt(agent_x**2 + agent_y**2)
        wall_dtheta = atan2(agent_y/agent_x) - agent_theta #not sure if this is exactly right because of how atan2 is set up but it shouldn't matter since it's consistent

        foe1_abs_position = None
        foe2_abs_position = None

        for a in self.agents:
            if a != agent and foe1_abs_position is None:
                foe1_abs_position = self.dictionary.get(a.name)
            elif foe2_abs_position is None:
                foe2_abs_position = self.dictionary.get(a.name)
        
        foe1_x = foe1_abs_position[0]
        foe1_y = foe1_abs_position[1]
        foe1_theta = foe1_abs_position[2]

        foe1_r = sqrt((foe1_x - agent_x)**2 + (foe1_y - agent_y)**2)
        foe1_dtheta = atan2((foe1_y-agent_y)/(foe1_x-agent_x))
        foe1_reltheta = foe1_theta - agent_theta

        foe2_x = foe2_abs_position[0]
        foe2_y = foe2_abs_position[1]
        foe2_theta = foe2_abs_position[2]

        foe2_r = sqrt((foe2_x - agent_x)**2 + (foe2_y - agent_y)**2)
        foe2_dtheta = atan2((foe2_y-agent_y),(foe2_x-agent_x))
        foe2_reltheta = foe2_theta - agent_theta

        return [wall_r,wall_dtheta,foe1_r,foe1_dtheta,foe1_reltheta,foe2_r,foe2_dtheta,foe2_reltheta]







