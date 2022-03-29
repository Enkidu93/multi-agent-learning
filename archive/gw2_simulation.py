from grid_world2 import GridWorld2
import matplotlib.pyplot as plt
from gridworld2_agent import GW2Agent

world = GridWorld2()
a1 = GW2Agent(world,qvals_filename="qvals_a12000000.json")
a2 = GW2Agent(world,qvals_filename="qvals_a22000000.json")
a3 = GW2Agent(world,qvals_filename="qvals_a32000000.json")
a4 = GW2Agent(world,qvals_filename="qvals_a42000000.json")

VISUALIZE = True
N = 1

if VISUALIZE:
    world.visualize()
# world.visualize()
episodes = [i for i in range(N//10_000 - 1)]
steps_per_episode = []
rewards = []
# recent_average_steps = 0
# recent_average_reward = 0
num_steps = 0
avg_reward = 0

for i in range(N):
    epsiode_steps = 0
    if i == (N-1):
        VISUALIZE = True
    while not world.episode_complete:
        a1.take_action()
        num_steps += 1
        epsiode_steps+=1
        if VISUALIZE:
            world.visualize()

        a2.take_action()
        if world.episode_complete:
            break
        a3.take_action()
        if world.episode_complete:
            break
        a4.take_action()
        if world.episode_complete:
            break
        if epsiode_steps > 1_000:
            break
    
    if VISUALIZE:
        world.visualize()

    avg_reward += a1.reward
    if (i%10_000) == 0 and i !=0:
        steps_per_episode.append(num_steps/10_000)
        rewards.append(avg_reward/10_000)
        num_steps = 0
        avg_reward = 0
        print(i)


    # if i >= (9*N)/10:
    #     recent_average_reward += a1.reward
    #     recent_average_steps += num_steps

    # if (i%10) == 0:
    a1.reset()
    a2.reset()
    a3.reset()
    a4.reset()

# recent_average_steps /= (N/10)
# recent_average_reward /= (N/10)

plt.plot(episodes,steps_per_episode)
plt.xlabel("EPISODE")
plt.ylabel("AVERAGE NUM STEPS PER EPISODE")
plt.show()

plt.plot(episodes,rewards)
plt.xlabel("EPISODE")
plt.ylabel("AVERAGE REWARD PER EPISODE")
plt.show()

# print("AVERAGE NUM STEPS PER EPISODE LAST 10% RUNS FOR AGENT A:",recent_average_steps)
# print("AVERAGE REWARD PER EPISODE LAST 10% RUNS FOR AGENT A:",recent_average_reward)

answer = input("Would you like to save a1's qvalues? (Y/n)")
if answer == "Y":
    a1.save(f"qvals_a1{N}.json")
answer = input("Would you like to save a2's qvalues? (Y/n)")
if answer == "Y":
    a2.save(f"qvals_a2{N}.json")
answer = input("Would you like to save a3's qvalues? (Y/n)")
if answer == "Y":
    a3.save(f"qvals_a3{N}.json")
answer = input("Would you like to save a4's qvalues? (Y/n)")
if answer == "Y":
    a4.save(f"qvals_a4{N}.json")