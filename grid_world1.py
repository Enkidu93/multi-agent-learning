import random
from world import World

class GridWorld1(World):
    def __init__(self):
        self.state = (2,0)
        self.episode_complete = False
        DOWN = 0
        LEFT = 1
        UP = 2
        RIGHT = 3
        self.actions = [DOWN, LEFT, UP, RIGHT]  
        self.action_vectors = [(1,0),(0,-1),(-1,0),(0,1)]     
    
    def reset(self):
        self.state = (2,0)
        self.episode_complete = False
    
    def step(self, action) -> tuple: # returns tuple where tuple[0] is reward, tuple[1] is new state, tuple[2] is episode_completed

        ACTIONS = self.actions
        ACTION_VECTORS = self.action_vectors
        
        reward = -0.04 # defaults to -0.04 reward per move

        rand = random.random()

        veers_right = True if rand <= 0.1 else False
        veers_left = True if rand <= 0.2 and not veers_right else False

        true_action = action

        if veers_right:
            true_action = ACTIONS[(action+1)%len(ACTIONS)]
        elif veers_left:
            true_action = ACTIONS[(action-1)%len(ACTIONS)]
        
        new_state = (self.state[0] + ACTION_VECTORS[true_action][0], self.state[1] + ACTION_VECTORS[true_action][1])

        if World.isInBounds(new_state):
            self.state = new_state
        
        if self.state == (0,3):
            reward = 1
            self.episode_complete = True
        elif self.state == (1,3):
            reward = -1
            self.episode_complete = True
        
        return (reward, self.state, self.episode_complete)
    
    def visualize(self):
        out = ".___.___.___.___.\n"
        for row in range(3):
            out += "|"
            for col in range(4):
                if self.state == (row,col):
                    out += " A |"
                elif row == 0 and col == 3:
                    out += " 1 |"
                elif row == 1 and col == 3:
                    out += "-1 |"
                elif row == 1 and col == 1:
                    out += "XXX|"
                else:
                    out += "   |"

            out += "\n.___.___.___.___.\n"
        print(out)

    @staticmethod
    def isInBounds(state):
        return state[0] > -1 and state[0] < 3 and state[1] > -1 and state[1] < 4 and not (state[0] == 1 and state[1] == 1)
