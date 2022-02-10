from grid_world1 import GridWorld1
from agent import Agent
import matplotlib.pyplot as plt

world = GridWorld1()
agent = Agent(world,epsilon=0.99999,alpha=0.99999,gamma=1,decay_epsilon=0.99999,decay_alpha=.99999)

VISUALIZE = False
N = 100

if VISUALIZE:
    world.visualize()

episodes = []
steps_per_episode = []
rewards = []

recent_average_steps = 0
recent_average_reward = 0

for i in range(N):
    episodes.append(i)
    num_steps = 0
    while not agent.world.episode_complete:
        agent.take_action()
        num_steps += 1
        if VISUALIZE:
            world.visualize()
    steps_per_episode.append(num_steps)
    rewards.append(agent.reward)
    if i >= (9*N)/10:
        recent_average_reward += agent.reward
        recent_average_steps += num_steps
    agent.reset()

recent_average_steps /= (N/10)
recent_average_reward /= (N/10)

plt.plot(episodes,steps_per_episode)
plt.show()

plt.plot(episodes,rewards)
plt.show()

print("AVERAGE NUM STEPS PER EPISODE LAST 10% RUNS:",recent_average_steps)
print("AVERAGE REWARD PER EPISODE LAST 10% RUNS:",recent_average_reward)

answer = input("Would you like to save these qvalues? (Y/n)")
if answer == "Y":
    agent.save(f"qvals_{N}_{round(recent_average_steps,3)}_{round(recent_average_reward,3)}_{round(agent.epsilon,3)}_{round(agent.decay_epsilon,3)}_{round(agent.alpha,3)}_{round(agent.decay_alpha,3)}_{round(agent.gamma,3)}.json")