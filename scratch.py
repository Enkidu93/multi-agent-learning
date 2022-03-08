import random
from math import sqrt
def move(tups:tuple[tuple],curpos:tuple,scale=1,cap=5) -> tuple:
    SCALE = scale
    CAP = cap

    dy = 0
    dx = 0

    distance_to_closest = 1_000_000
    has_neighbor_x = False
    has_neighbor_y = True

    # "Pressure" movement
    for t in tups:

        dy += SCALE/(curpos[0] - t[0]) if (curpos[0] - t[0]) != 0 else 0
        dx += SCALE/(curpos[1] - t[1]) if (curpos[1] - t[1]) != 0 else 0
        
        # Keep track of closest for "buckling" movement below
        distance = sqrt((curpos[0] - t[0])**2 + (curpos[1] - t[1])**2)

    # "Buckling" movement
    if dy == 0 and dx == 0:
        dy = random.randint(-1,1)*SCALE/distance if distance != 0 else CAP
        dx = random.randint(-1,1)*SCALE/distance if distance != 0 else CAP

    return (min(CAP,dy),min(CAP,dx))

pos = ((2,0),(2,4),(4,2))
prey = (2,2)
dy,dx = move(pos, prey)
print(dy,dx)
# print(prey[0]+dy,prey[0]+dx)
# _ _ _ _ _
# _ _ _ _ _
# A _ P _ A
# _ _ _ _ _
# _ _ A _ _