from world import World
import random

class ConnectFour(World):
    def __init__(self):
        self.state = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]] # create empty 6x7 'matrix' 
        self.free_rows_per_column = [6,6,6,6,6,6,6]
        self.episode_complete = False
        self.actions = [3,4,2,5,1,6,0] # each column where a piece can be dropped
        self.cur_player = random.choice((-1,1))
        self.players = {1:None,-1:None}

    def reset(self):
        self.state = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]] # create empty 6x7 'matrix' 
        self.free_rows_per_column = [6,6,6,6,6,6,6]
        self.episode_complete = False
        self.actions = [3,4,2,5,1,6,0]
        self.cur_player = random.choice((-1,1))

    def step(self, action:int) -> tuple:
        reward = 0

        if self.winningMove(action, self.cur_player):
            reward = 1
            self.players.get(-1*self.cur_player).reward += -1
            self.episode_complete = True
        
        if self.free_rows_per_column[action] <= 0 and action in self.actions:
            self.actions.remove(action)


        # print(action, self.cur_player)

        # print("FREE ROWS PER COL:",self.free_rows_per_column[action])

        self.state[self.free_rows_per_column[action]-1][action] = self.cur_player
        self.free_rows_per_column[action] -= 1
        
        self.cur_player *= -1

        # print(reward, self.state, self.episode_complete, self.cur_player)
        
        return (reward, self.state, self.episode_complete)
        
    
    @staticmethod
    def translate(n):
        if n==1:
            return "A"
        if n==-1:
            return "B"
        return "X"

    def visualize(self):
        print("| ",0, " ",1, " ",2, " ",3, " ",4, " ",5, " ",6," |")
        for row in self.state:
            print("| ",ConnectFour.translate(row[0]),
                " ", ConnectFour.translate(row[1]),
                " ", ConnectFour.translate(row[2]), 
                " ",  ConnectFour.translate(row[3]), 
                " ",  ConnectFour.translate(row[4]), 
                " ",  ConnectFour.translate(row[5]), 
                " ",  ConnectFour.translate(row[6])," |")
        print()
    
    def winningMove(self, action:int, player:int) -> bool:
        board = self.state
        row = self.free_rows_per_column[action] - 1 # number of free rows minus one to find position of row up from "bottom"
        col = action
        num_in_row = 1
        steps_out = 1
        for i in (0,1):
            for j in (0,-1):

                if row+i in range(6) and col+j in range(7) and (i!=0 or j!=0):
                    while(board[row+steps_out*i][col+steps_out*j]==player):
                        num_in_row += 1
                        if num_in_row >= 4:
                            return True                     
                        steps_out += 1
                        if row+steps_out*i not in range(6) or col+steps_out*j not in range(7) or (i==0 and j==0):
                            break
                
                steps_out = 1

                if row-i in range(6) and col-j in range(7) and (i!=0 or j!=0):
                    while(board[row-steps_out*i][col-steps_out*j]==player):
                        num_in_row += 1
                        if num_in_row >= 4:
                            return True 
                        steps_out += 1
                        if row-steps_out*i not in range(6) or col-steps_out*j not in range(7) or (i==0 and j==0):
                            break 
                    
                steps_out = 1

                num_in_row = 1  

        return False              

# cf = ConnectFour()
# cf.state = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,-1,0,0,0],[1,0,1,1,0,0,0],[1,1,1,-1,0,-1,-1]]
# print(tuple(cf.state))
# cf.free_rows_per_column[1] = 5
# cf.visualize()
# print(cf.winningMove(4,-1))