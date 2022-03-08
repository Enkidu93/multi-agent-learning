import random
from world import World

class GridWorld2(World):
    def __init__(self):
        self.state = {"A":(1,1),"B":(1,7),"C":(7,1),"D":(7,7)}
        self.episode_complete = False
        DOWN = 0
        LEFT = 1
        UP = 2
        RIGHT = 3
        self.actions = [DOWN, LEFT, UP, RIGHT]  
        self.action_vectors = [(1,0),(0,-1),(-1,0),(0,1)] 
        self.players = {"A":None,"B":None,"C":None,"D":None} 
        self.cur_player = "A"   
    
    def reset(self):
        self.state = {"A":(0,0),"B":(0,8),"C":(8,0),"D":(8,8)}
        self.episode_complete = False
    
    def step(self, action) -> tuple: # returns tuple where tuple[0] is reward, tuple[1] is new state, tuple[2] is episode_complete

        action = int(action) #takes care of JSON loading issue

        ACTIONS = self.actions
        ACTION_VECTORS = self.action_vectors
        
        reward = -0.01 # defaults to -0.01 reward per move

        rand = random.random()

        veers_right = True if rand <= 0.05 else False
        veers_left = True if rand <= 0.1 and not veers_right else False

        true_action = action

        if veers_right:
            true_action = ACTIONS[(action+1)%len(ACTIONS)]
        elif veers_left:
            true_action = ACTIONS[(action-1)%len(ACTIONS)]
        
        new_state =  self.state[self.cur_player]
        new_state = (new_state[0] + ACTION_VECTORS[true_action][0], new_state[1] + ACTION_VECTORS[true_action][1])

        if GridWorld2.isInBounds(new_state,self.state):
            self.state[self.cur_player] = new_state
        
        if new_state == (4,4):
            reward = 1
            self.episode_complete = True
        elif new_state == (4,2) or new_state == (4,6):
            reward = -1
            self.episode_complete = True
        
        if self.cur_player == "A":
            self.cur_player = "B"
        elif self.cur_player == "B":
            self.cur_player = "C"
        elif self.cur_player == "C":
            self.cur_player = "D"   
        else:
            self.cur_player = "A"        
        
        return (reward, self.state, self.episode_complete)
    
    def visualize(self):
        out = ".___.___.___.___.___.___.___.___.___.\n"
        for row in range(9):
            out += "|"
            for col in range(9):
                if self.state.get("A") == (row,col):
                    out += " A |"
                elif self.state.get("B") == (row,col):
                    out += " B |"
                elif self.state.get("C") == (row,col):
                    out += " C |"
                elif self.state.get("D") == (row,col):
                    out += " D |"                                                            
                elif row == 4 and col == 4:
                    out += " 1 |"
                elif (row,col) in ((4,2),(4,6)):
                    out += "-1 |"
                elif (row,col) in ((2,2),(2,6),(6,2),(6,6)):
                    out += "XXX|"
                else:
                    out += "   |"

            out += "\n.___.___.___.___.___.___.___.___.___.\n"
        print(out)

    @staticmethod
    def isInBounds(state,boardstate):
        return (state[0] > -1 and state[0] < 9 and state[1] > -1 and state[1] < 9 
            and not (state in ((2,2),(2,6),(6,2),(6,6)))
            and not (state in boardstate.values()))
