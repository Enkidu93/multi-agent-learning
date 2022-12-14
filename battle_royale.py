from world import World
from newnetworkagent import NetworkAgent
from math import cos, sin, pi, sqrt, atan, atan2
import numpy as np
from message import Message
import random as r

R_CCW = 0
R_CW = 1
STEP = 2
ATTACK = 3

SCALE_DISTANCE = 1.5 # should be between 0 and sqrt(WORLD_SIZE)/2

WORLD_SIZE = 30

def generateStartLocs():
    distance1 = (1 + r.random())*sqrt(WORLD_SIZE)/3 
    distance2 = (1+ r.random())*sqrt(WORLD_SIZE)/3
    distance3 = (1 + r.random())*sqrt(WORLD_SIZE)/3

    orientation1 = r.random()*2*pi
    orientation2 = r.random()*2*pi
    orientation3 = r.random()*2*pi

    angle1 = pi/6 - r.random()*pi/3
    angle2 = 2*(pi/3) + pi/6 - r.random()*pi/3
    angle3 = 5*pi/3 + pi/6 - r.random()*pi/3
    
    return [
        [cos(angle1)*distance1,sin(angle1)*distance1,orientation1],
        [cos(angle2)*distance2,sin(angle2)*distance2,orientation2],
        [cos(angle3)*distance3,sin(angle3)*distance3,orientation3]
        ]

class BattleRoyale(World):
    def __init__(self,agents:list):
        self.dictionary = dict()
        self.agents = agents
        # start_locs = [[sqrt(WORLD_SIZE)/2*SCALE_DISTANCE,0*SCALE_DISTANCE,pi],[(sqrt(WORLD_SIZE)/2)*cos(2*pi/3)*SCALE_DISTANCE,(sqrt(WORLD_SIZE)/2)*sin(2*pi/3)*SCALE_DISTANCE,5*pi/3],[-(sqrt(WORLD_SIZE)/2)*cos(5*pi/3)*SCALE_DISTANCE,(sqrt(WORLD_SIZE)/2)*sin(5*pi/3)*SCALE_DISTANCE,2*pi/3 - pi/2]]
        start_locs = generateStartLocs()
        i = 0
        for agent in self.agents:
            if i < 4:
                self.dictionary[agent.name] = start_locs[i]
                i+=1
            else:
                print("Too many agents...")
                break
        self.actions = [R_CCW, R_CW, STEP, ATTACK]
        self.episode_complete = False
        self.suspect_episode_complete = False
        self.action_count = {0:0,1:0,2:0,3:0}
    

    def reset(self, reset_qvalues=False, reset_state_count=False):
        # print("RESETTING WORLD")
        # start_locs = [[sqrt(WORLD_SIZE)/2*SCALE_DISTANCE,0*SCALE_DISTANCE,pi],[(sqrt(WORLD_SIZE)/2)*cos(2*pi/3)*SCALE_DISTANCE,(sqrt(WORLD_SIZE)/2)*sin(2*pi/3)*SCALE_DISTANCE,5*pi/3],[-(sqrt(WORLD_SIZE)/2)*cos(5*pi/3)*SCALE_DISTANCE,(sqrt(WORLD_SIZE)/2)*sin(5*pi/3)*SCALE_DISTANCE,2*pi/3 - pi/2]]
        start_locs = generateStartLocs()
        count = 0
        for agent in self.agents:
            agent.reset(reset_qvalues=reset_qvalues, reset_state_count=reset_state_count)
            if count < 3:
                self.dictionary[agent.name] = start_locs[count]
                count+=1
            else:
                print("Too many agents...")
                break
        self.episode_complete = False
        self.suspect_episode_complete = False
    
    def step(self, action:int, agent:NetworkAgent):
        reward = -1
        self.action_count[action] += 1
        if action == R_CW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] + pi/12)%(2*pi)
            self.dictionary[agent.name] = new_state
        if action == R_CCW:
            new_state = self.dictionary.get(agent.name)
            new_state[2] = (new_state[2] - pi/12)%(2*pi)
            self.dictionary[agent.name] = new_state        
        if action == STEP:
            distance_stepped = 0.5 #Keep constant for now
            new_state = self.dictionary.get(agent.name)
            temp_state = new_state.copy()
            temp_state[0] += distance_stepped*cos(new_state[2])
            temp_state[1] += distance_stepped*sin(new_state[2])
            if not BattleRoyale.isOutOfBounds(temp_state[0],temp_state[1]):
                new_state[0] += distance_stepped*cos(new_state[2])
                new_state[1] += distance_stepped*sin(new_state[2])
                self.dictionary[agent.name] = new_state        
        if action == ATTACK:
            self_pos = self.dictionary.get(agent.name)
            self_x = self_pos[0]
            self_y = self_pos[1]
            self_theta = self_pos[2]

            hit = False
            
            for a in self.agents:
                a_abs_pos = self.dictionary.get(a.name)
                a_x = a_abs_pos[0]
                a_y = a_abs_pos[1]
                # print(sqrt((a_x - self_x)**2 + (a_y - self_y)**2)<=1,(atan2((a_y-self_y),(a_x-self_x)))<=((self_theta + pi/8)%(2*pi)),atan2((a_y-self_y),(a_x-self_x))>=((self_theta - pi/8)%2*pi))
                if a != agent and sqrt((a_x - self_x)**2 + (a_y - self_y)**2)<=2 and (atan2((a_y-self_y),(a_x-self_x))-pi/5)%(2*pi)<=self_theta%(2*pi) and (atan2((a_y-self_y),(a_x-self_x))+pi/5)%(2*pi)>=self_theta%(2*pi):
                    # start_locs = [[sqrt(WORLD_SIZE)/2,0,pi],[(sqrt(WORLD_SIZE)/2)*cos(2*pi/3),(sqrt(WORLD_SIZE)/2)*sin(2*pi/3),5*pi/3],[-(sqrt(WORLD_SIZE)/2)*cos(5*pi/3),(sqrt(WORLD_SIZE)/2)*sin(5*pi/3),2*pi/3 - pi/2]]
                    # self.dictionary[a.name] >= start_locs[self.agents.index(a)]
                    reward = 500
                    # reward = 10 if not hit else 20 #previously 5 and 10
                    hit = True

            if not hit:
                reward = -25
            else:
                # self.episode_complete = True
                self.suspect_episode_complete = True
            
            # print(agent.name, "took action", action)
        # agent.epsilon *= agent.decay_epsilon
        # agent.alpha *= agent.decay_alpha
        return (reward, self.dictionary)
    
    # def process(self, message_content:tuple[str,tuple]):
    def process(self, message:Message):
        agent_name = message.content[0]
        agent_state = message.content[1]
        self.dictionary[agent_name] = agent_state
    
    def translateAbsoluteState(self,agent:NetworkAgent):
        agent_abs_position = self.dictionary.get(agent.name)
        agent_x = agent_abs_position[0]
        agent_y = agent_abs_position[1]
        agent_theta = agent_abs_position[2]

        wall_r = sqrt(WORLD_SIZE) - sqrt(agent_x**2 + agent_y**2)
        wall_dtheta = atan2(agent_y,agent_x) - agent_theta #not sure if this is exactly right because of how atan2 is set up but it shouldn't matter since it's consistent

        foe1_abs_position = None
        foe2_abs_position = None

        for a in self.agents:
            if a != agent and foe1_abs_position is None:
                foe1_abs_position = self.dictionary.get(a.name)
            elif a != agent and foe2_abs_position is None:
                foe2_abs_position = self.dictionary.get(a.name)
        
        foe1_x = foe1_abs_position[0]
        foe1_y = foe1_abs_position[1]
        foe1_theta = foe1_abs_position[2]

        foe1_r = sqrt((foe1_x - agent_x)**2 + (foe1_y - agent_y)**2)
        foe1_dtheta = atan2((foe1_y-agent_y),(foe1_x-agent_x)) - agent_theta
        foe1_reltheta = foe1_theta - agent_theta

        foe2_x = foe2_abs_position[0]
        foe2_y = foe2_abs_position[1]
        foe2_theta = foe2_abs_position[2]

        foe2_r = sqrt((foe2_x - agent_x)**2 + (foe2_y - agent_y)**2)
        foe2_dtheta = atan2((foe2_y-agent_y),(foe2_x-agent_x)) - agent_theta
        foe2_reltheta = foe2_theta - agent_theta

        return [round(wall_r,0),round(wall_dtheta,0),round(foe1_r,0),round(foe1_dtheta,0),round(foe1_reltheta,0),round(foe2_r,0),round(foe2_dtheta,0),round(foe2_reltheta,0)]
        # return [round(wall_r*2,0)/2,round(wall_dtheta*2,0)/2,round(foe1_r*2,0)/2,round(foe1_dtheta*2,0)/2,round(foe1_reltheta*2,0)/2,round(foe2_r*2,0)/2,round(foe2_dtheta*2,0)/2,round(foe2_reltheta*2,0)/2]
    
    def visualize(self):
        return super().visualize()

    @staticmethod
    def isOutOfBounds(x,y):
        return ((x**2 + y**2) > WORLD_SIZE)

    def getHeuristicBestActionFor(self, agent):
        flip = r.random() >= 0.75
        # if(r.random() >= 0.85):
        #     return r.choice(self.actions)
        closest_agent = None
        # other_agent = None
        distance = 100 #would need to change
        agent_x = self.dictionary.get(agent.name)[0]
        agent_y = self.dictionary.get(agent.name)[1]
        agent_theta = self.dictionary.get(agent.name)[2]
        for a in self.agents:
            if a.name == agent.name:
                continue
            curdistance = sqrt((self.dictionary.get(a.name)[0] - agent_x)**2 + (self.dictionary.get(a.name)[1] - agent_y)**2)
            if curdistance < distance:
                distance = curdistance
                # other_agent = closest_agent
                closest_agent = a
            # else:
            #     other_agent = a
        # if(r.random() >= 0.5):
        #     closest_agent = other_agent
        closest_agent_x = self.dictionary.get(closest_agent.name)[0]
        closest_agent_y = self.dictionary.get(closest_agent.name)[1]
        closest_agent_theta = self.dictionary.get(closest_agent.name)[2]
        alpha = max((agent_theta - closest_agent_theta), (closest_agent_theta - agent_theta))
        theta = (atan2((closest_agent_y - agent_y),(closest_agent_x - agent_x)) - agent_theta)%(2*pi)
        # print(alpha, theta)
        # print()

        will_hit_wall = False

        if (agent_x + 0.5*cos(agent_theta))**2 + (agent_y + 0.5*sin(agent_theta))**2 >= WORLD_SIZE:
            will_hit_wall = True

        can_kill = sqrt((closest_agent_x - agent_x)**2 + (closest_agent_y - agent_y)**2)<=2 and (atan2((closest_agent_y-agent_y),(closest_agent_x-agent_x)) - pi/5)%(2*pi) <= agent_theta%(2*pi) and (atan2((closest_agent_y-agent_y),(closest_agent_x-agent_x)) + pi/5)%(2*pi) >=agent_theta%(2*pi)
        # print("CAN KILL", can_kill, agent.name)
        # print(self.dictionary.get(agent.name), self.dictionary.get(closest_agent.name))
        if can_kill:
            return ATTACK
        # print(sqrt((closest_agent_x - agent_x)**2 + (closest_agent_y - agent_y)**2))
        # print((atan2((closest_agent_y-agent_y),(closest_agent_x-agent_x)) - pi/5))
        # print((atan2((closest_agent_y-agent_y),(closest_agent_x-agent_x)) + pi/5))
        # print(will_hit_wall)
        # print()



        want_to_kill = alpha + pi >= 2*theta
        # print("WANT TO KILL", want_to_kill, agent.name)

        # if in_range:
        if want_to_kill:
            if (theta%(2*pi)) > pi/2 and (theta%(2*pi)) < 3*pi/2:
                turn_ccw = (theta%(2*pi)) < pi
                # print("HERE",turn_ccw)
                if turn_ccw:
                    return R_CCW if not flip else R_CW
                else:
                    return R_CW if not flip else R_CCW
            else:
                return STEP if not will_hit_wall and not flip else R_CW
        else:
            if (theta%(2*pi)) <= pi/2 or (theta%(2*pi)) >= 3*pi/2:
                turn_cw = (theta%(2*pi)) < pi
                # print("THERE",turn_cw)
                if turn_cw:
                    return R_CW if not flip else R_CCW
                else:
                    return R_CCW if not flip else R_CW
            else:
                return STEP if not will_hit_wall and not flip else R_CW
        

# sqrt(WORLD_SIZE)/2,0,pi ... sqrt(WORLD_SIZE)/2 - 0.5, 0, 0

# a = NetworkAgent(None, None)
# a.name = 'fred'
# b = NetworkAgent(None, None)
# b.name = 'bill'
# world = BattleRoyale(agents=[a,b])
# world.dictionary[(b.name)] = [sqrt(WORLD_SIZE)/2 - 0.5, 0, pi]
# print(world.getHeuristicBestActionFor(a))
# print(world.getHeuristicBestActionFor(b))










