import arcade
from math import sqrt, sin, pi, cos

SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Test")

arcade.set_background_color(arcade.color.WHITE)

arcade.start_render()

x = 250
y = 250
radius = 120
arcade.draw_circle_outline(x, y, radius, arcade.color.BLACK)

start_locs = [[sqrt(30)/2,0,pi],[(sqrt(30)/2)*cos(2*pi/3),(sqrt(30)/2)*sin(2*pi/3),5*pi/3],[-(sqrt(30)/2)*cos(5*pi/3),(sqrt(30)/2)*sin(5*pi/3),-2*pi/3]]

arcade.draw_point(x=10*sqrt(30)+250,y=10*0+250,color=arcade.color.RED,size=5)

arcade.draw_point(x=10*sqrt(30)*cos(2*pi/3)+250,y=10*sqrt(30)*sin(2*pi/3)+250,color=arcade.color.BLUE,size=5)

arcade.draw_point(x=-10*sqrt(30)*cos(5*pi/3)+250,y=10*sqrt(30)*sin(5*pi/3)+250,color=arcade.color.GREEN,size=5)

arcade.finish_render()

arcade.run()