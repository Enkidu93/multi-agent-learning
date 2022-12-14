import pandas as pd
import arcade
from math import sqrt, sin, pi, cos, degrees, floor
import battle_royale as b
import machine as m
import networkagent as n
from tensorflow.keras.models import load_model
from generate_delay import WeibullDelayGenerator

SCALE = 1
SCREEN_WIDTH = floor(SCALE*500)
SCREEN_HEIGHT = floor(SCALE*500)
ALPHA = 0.0
EPSILON = 0.0

gen = WeibullDelayGenerator(seed=10)


def no_delay():
    # return 0
    return gen.generate_weibulldist_delay()


class BattleRoyaleWindow(arcade.Window):
    def __init__(self,width,height):
        super().__init__(width,height,"Battle Royale")
        arcade.set_background_color(arcade.color.GRAY_BLUE)
    
    def setup(self):
        self.t = 0
        self.quit = False

        self.a = arcade.Sprite("sprite.png",SCALE*(0.15))
        self.a.center_x = SCALE*(10*sqrt(30)+250)
        self.a.center_y = SCALE*(10*0+250)
        self.a.angle = degrees(pi)
        self.a.change_angle = degrees(pi)
        self.a.change_x = self.a.center_x
        self.a.change_y = self.a.center_y

        self.b = arcade.Sprite("sprite.png",SCALE*0.15)
        self.b.center_x = SCALE*(10*sqrt(30)*cos(2*pi/3)+250)
        self.b.center_y = SCALE*(10*sqrt(30)*sin(2*pi/3)+250)
        self.b.angle = degrees(5*pi/3)
        self.b.change_angle = degrees(5*pi/3)
        self.b.change_x = self.b.center_x
        self.b.change_y = self.b.center_y

        self.c = arcade.Sprite("sprite.png",SCALE*0.15)
        self.c.center_x = SCALE*(-10*sqrt(30)*cos(5*pi/3)+250)
        self.c.center_y = SCALE*(10*sqrt(30)*sin(5*pi/3)+250)
        self.c.angle = degrees(2*pi/3 - pi/2)
        self.c.change_angle = degrees(2*pi/3 - pi/2)
        self.c.change_x = self.c.center_x
        self.c.change_y = self.c.center_y

        self.agent_reps = arcade.SpriteList()
        self.agent_reps.append(self.a)
        self.agent_reps.append(self.b)
        self.agent_reps.append(self.c)

        df = pd.read_csv('teststatsNOV14_101.csv')

        a1 = n.NetworkAgent(None,"A",epsilon=EPSILON,alpha=ALPHA)
        a2 = n.NetworkAgent(None,"B",epsilon=EPSILON,alpha=ALPHA,is_heuristic=True)
        a3 = n.NetworkAgent(None,"C",epsilon=EPSILON,alpha=ALPHA,is_heuristic=True)
        self.agents = [a1,a2,a3]

        w1 = b.BattleRoyale(self.agents)
        w2 = b.BattleRoyale(self.agents)
        w3 = b.BattleRoyale(self.agents)

        a1.world = w1
        a2.world = w2
        a3.world = w3

        m1 = m.Machine(a1,"A") #For now, these names have to match corresponding agents
        m2 = m.Machine(a2,"B")
        m3 = m.Machine(a3,"C")
        self.machines = [m1,m2,m3]

        c1_2 = m.Connection(m1,m2,no_delay)
        c1_3 = m.Connection(m1,m3,no_delay)
        m1.add_connection(m2,c1_2)
        m1.add_connection(m3,c1_3)

        c2_1 = m.Connection(m2,m1,no_delay)
        c2_3 = m.Connection(m2,m3,no_delay)
        m2.add_connection(m1,c2_1)
        m2.add_connection(m3,c2_3)

        c3_1 = m.Connection(m3,m1,no_delay)
        c3_2 = m.Connection(m3,m2,no_delay)
        m3.add_connection(m1,c3_1)
        m3.add_connection(m2,c3_2)

        # a1.value_approximator.model = load_model("model/OCT19TEST17")
        # a2.value_approximator.model = load_model("model\\WEANEDVM2")
        # a3.value_approximator.model = load_model("model\\WEANEDVM3")

        a1.has_model = True
        a2.has_model = True
        a3.has_model = True
    
    def on_draw(self):

        arcade.start_render()

        x = SCALE*250
        y = SCALE*250
        radius = SCALE*120
        arcade.draw_circle_outline(x, y, radius, arcade.color.ORIOLES_ORANGE, border_width=3)
        self.agent_reps.draw()
        if self.quit:
            arcade.draw_text("Game is complete...",SCALE*(SCREEN_WIDTH/2.5),SCALE*(3*SCREEN_HEIGHT/4),arcade.color.WHITE,SCALE*10,SCALE*20,'left')
            print(self.agents[0].reward)
            print(self.agents[1].reward)
            print(self.agents[2].reward)
        arcade.finish_render()

    def update(self,delta_time):
        for agent in self.agent_reps:
            agent.center_x = agent.change_x
            agent.center_y = agent.change_y
            agent.angle = agent.change_angle

    def on_key_press(self, key, modifiers):
        if (key == arcade.key.RIGHT or key == arcade.key.LEFT) and not self.quit:
            raw_pos = get_new_positions(self.machines,self.t)
            self.t += 30
            if raw_pos is None:
                self.quit = True
                # print(self.agents[0].q_values)
                return
            new_positions = [e.copy() for e in raw_pos.copy()]
            
            for i in range(len(new_positions)):
                for j in range(2):
                    new_positions[i][j] = SCALE*(new_positions[i][j]*20 + 250)
            i = 0
            for agent in self.agent_reps:
                agent.change_x = new_positions[i][0]
                agent.change_y = new_positions[i][1]
                agent.change_angle = degrees(new_positions[i][2])
                i += 1
    
    def on_key_release(self, key, modifiers):
        pass

def get_new_positions(machines, t) -> list:
    for machine in machines:
        machine.activate(t)
        if(machine.world.episode_complete):
            return None
    return list(machines[0].world.dictionary.values())



def main():
    game = BattleRoyaleWindow(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.setup()
    arcade.run()


if __name__ == "__main__":
    main()



