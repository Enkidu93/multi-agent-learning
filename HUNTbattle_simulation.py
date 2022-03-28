import battle_royale as b
import machine as m
import networkagent as n
import passiveBRagent as p
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def no_delay():
    return 0

N = 10_000
ALPHA = 0.6
EPSILON = 0.6
ALPHA_DECAY = 0.9995
EPSILON_DECAY = 0.9995

a1 = n.NetworkAgent(None,"A",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY)
a2 = p.PassiveBattleRoyaleAgent(None,"B",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY)
a3 = p.PassiveBattleRoyaleAgent(None,"C",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY)
agents = [a1,a2,a3]

w1 = b.BattleRoyale(agents)
w2 = b.BattleRoyale(agents)
w3 = b.BattleRoyale(agents)

a1.world = w1
a2.world = w2
a3.world = w3

m1 = m.Machine(a1,"VM1")
m2 = m.Machine(a2,"VM2")
m3 = m.Machine(a3,"VM3")
machines = [m1,m2,m3]

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

# a1.value_approximator.model = load_model(m1.name)
# a2.value_approximator.model = load_model(m2.name)
# a3.value_approximator.model = load_model(m3.name)

# a1.has_model = True
# a2.has_model = True
# a3.has_model = True

x = list()
y = list()
y_r =list()

avg_t = 0
avg_r = 0

for i in range(N):
    interval = 500

    quit = False
    t = 0
    while(t<10_000 and not quit):
        for machine in machines:
            machine.activate(t)
            if(machine.world.episode_complete):
                quit = True
        t+=30

    avg_t+=t

    avg_r+=a1.reward

    for machine in machines:
        machine.world.reset()

    if i%interval == 0 and i!=0:
        x.append(i)
        y.append(avg_t/interval)
        y_r.append(avg_r/interval)
        avg_t = 0
        avg_r = 0
        print(i)
        for machine in machines:
            machine.agent.refit_model()


plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Average length of match")
plt.show()

plt.plot(x,y_r)
plt.xlabel("Time")
plt.ylabel("Average reward for hunting agent")
plt.show()

for machine in machines:
    machine.agent.value_approximator.model.save("HUNT"+machine.name)

