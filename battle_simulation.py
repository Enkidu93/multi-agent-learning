import battle_royale as b
import machine as m
import networkagent as n
import pandas as pd
# import networkagent as n
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import clear_session
from generate_delay import WeibullDelayGenerator
import time

r_s = list()
x_s = list()
lt_wr = list()
lt_lr = list()
prev_alpha =  0.0
prev_epsilon = 0.0

learning_agent_positions = list()
enemy1_positions = list()
enemy2_positions = list()
actions = list()
fitting_losses = list()


M=0

for k in range(1, 2, 1): #[20,50,100,200,300,325,375,400,500]:
    # gen = WeibullDelayGenerator(seed=1,m=M,d=k)
    def delay():
        # return gen.generate_weibulldist_delay()
        return 0

    df = pd.DataFrame()

    N = 100
    ALPHA = prev_alpha
    EPSILON = prev_epsilon # = 2.0 disables q-updates for faster fully random
    ALPHA_DECAY = 0.999975 #0.999975
    EPSILON_DECAY = 0.999995
    INTERVAL = 20 # episodes

    start = time.time()

    a1 = n.NetworkAgent(None,"A",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY, is_tab=False)
    a2 = n.NetworkAgent(None,"B",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY, is_heuristic=True)
    a3 = n.NetworkAgent(None,"C",epsilon=EPSILON,alpha=ALPHA,decay_alpha=ALPHA_DECAY,decay_epsilon=EPSILON_DECAY, is_heuristic=True)
    agents = [a1,a2,a3]

    # print(time.time() - start, a1.time_in_inference)

    # print(a1.times_used_cached, a1.times_used_model)

    w1 = b.BattleRoyale(agents)
    w2 = b.BattleRoyale(agents)
    w3 = b.BattleRoyale(agents)

    a1.world = w1
    a2.world = w2
    a3.world = w3

    print(a1.world.action_count)

    m1 = m.Machine(a1,a1.name)
    m2 = m.Machine(a2,a2.name)
    m3 = m.Machine(a3,a3.name)
    machines = [m1,m2,m3]

    c1_2 = m.Connection(m1,m2,delay)
    c1_3 = m.Connection(m1,m3,delay)
    m1.add_connection(m2,c1_2)
    m1.add_connection(m3,c1_3)

    c2_1 = m.Connection(m2,m1,delay)
    c2_3 = m.Connection(m2,m3,delay)
    m2.add_connection(m1,c2_1)
    m2.add_connection(m3,c2_3)

    c3_1 = m.Connection(m3,m1,delay)
    c3_2 = m.Connection(m3,m2,delay)
    m3.add_connection(m1,c3_1)
    m3.add_connection(m2,c3_2)

    try:
        # a1.value_approximator.model = load_model(f"model/NOV17DISTANCED{k-1}")
        a1.value_approximator.model = load_model("model/DEC14")
    except:
        print("\n\n\n\n!!!Error loading model!!!\n\n\n\n")
    # # a2.value_approximator.model = load_model(f"model/F22SEP30SMALLBRAINROUND{66+k-1}FINALB")
    # a3.value_approximator.model = load_model(f"model/F22SEP30SMALLBRAINROUND{66+k-1}FINALC")

    a1.has_model = True
    a2.has_model = True
    a3.has_model = True

    master_model = a1.value_approximator.model
    a2.value_approximator.model = master_model
    a3.value_approximator.model = master_model

    master_cache = a1.q_values
    a2.q_values = master_cache
    a3.q_values = master_cache

    # print(len(master_cache))
    # print(master_model.get_weights())

    x = list()
    y = list()
    y_r =list()
    w_s = list()
    l_s = list()

    avg_t = 0
    avg_r = 0
    avg_wr = 0
    avg_lr = 0

    interval = INTERVAL 
    for i in range(1,N+1,1):
        # print(i)

        quit = False
        t = 0
        prev_reward = 0
        # while(t<(10*interval*(N+i)/N) and not quit):
        while(t<30*400 and not quit):
            for machine in machines:
                a, r = machine.activate(t)
                # print('REWARD:', r, 'AGENT:', machine.agent.name)
                if machine == m1 and r >= 500:
                    avg_wr += 1
                if machine == m1 and a1.reward - prev_reward <= -400:
                    avg_lr += 1
                prev_reward = a1.reward
                # if machine == m1:
                    # print("CURRENT POSITION/ACTION/REWARD:",m1.world.dictionary.get(m1.agent.name), a, m1.agent.reward)
                    # learning_agent_positions.append(m1.world.dictionary.get(m1.agent.name))
                    # enemy1_positions.append(m1.world.dictionary.get(m2.agent.name))
                    # enemy2_positions.append(m1.world.dictionary.get(m3.agent.name))
                    # actions.append(a)
                    # df = df.append({'action':a, 'mypos':m1.world.dictionary.get(m1.agent.name), 'enemy1pos':m1.world.dictionary.get(m2.agent.name), 'enemy2pos':m1.world.dictionary.get(m3.agent.name)}, ignore_index=True)
                if(machine.world.episode_complete):
                    quit = True
        # ############ STATIC ################
        #     m1.activate(t)
        #     if m1.world.episode_complete:
        #         quit = True
        # ####################################
            # print(t)
            t+=30
            

        avg_t+=t

        # for machine in machines:
            # avg_r+=machine.agent.reward
        avg_r += a1.reward

        if i%interval == 0 and i != 0:
            x.append(i)
            y.append(avg_t/interval)
            # y_r.append(avg_r/interval/3)
            y_r.append(avg_r/interval)
            w_s.append(avg_wr/interval)
            l_s.append(avg_lr/interval)
            print("Episode number/average match-length/average reward",i,avg_t/interval,avg_r/interval)
            print("CACHE LEN", len(master_cache))
            print("Epsilon/Alpha", a1.epsilon, a1.alpha)
            print("Percentage new states", (a1.new_states)/(a1.total_states))
            print("Win ratio:", avg_wr/interval)
            print("Loss ratio:", avg_lr/interval)
            avg_t = 0
            avg_r = 0
            avg_wr = 0
            avg_lr = 0
            # for machine in machines:
            #     machine.agent.refit_model()
            #     machine.world.reset(reset_qvalues=True)

            # history = m1.agent.refit_model()
            # fitting_losses.append(history.history['loss'])

            # a1.value_approximator.model.save(f'model/NOV18DISTANCED{k}_temp')
            # clear_session()
            # a1.value_approximator.model = load_model(f'model/NOV18DISTANCED{k}_temp')
        a1.reset(reset_epsilon_to=prev_epsilon if i <= N/4 else 0)
        # if (i <= N/5):
        #     a1.alpha = prev_alpha
        m1.world.reset(reset_state_count=True)
        m2.world.reset()
        m3.world.reset()
            
            # master_cache = a1.q_values
            # a2.q_values = master_cache
            # a3.q_values = master_cache

            # for machine in machines:
                # machine.agent.value_approximator.model.save(f"model/F22SEP30SBROUND{66+k}{i}"+machine.name)
                # clear_session()
                # machine.agent.value_approximator.model = load_model(f"model/F22SEP30SBROUND{66+k}{i}"+machine.name)
            # master_model = a1.value_approximator.model
            # a2.value_approximator.model = master_model
            # a3.value_approximator.model = master_model

        for machine in machines:
            a = machine.agent
            a.epsilon *= a.decay_epsilon
            a.alpha *= a.decay_alpha
            if(a.alpha < 0.01):
                a.alpha = 0.01
        
        if i==(N):
            prev_alpha = a1.alpha
            prev_epsilon = a1.epsilon

            # machine.agent.value_approximator.model.save(f"model/F22SEP29ROUND4{i}"+machine.name)
        # ############ STATIC ################
        # avg_r += m1.agent.reward
        # m1.world.reset()
        # ####################################

        # ############ STATIC ################
        #     m1.agent.refit_model()
        #     m1.agent.value_approximator.model.save("model\\TEST"+m1.name)
        # ####################################


    plt.plot(x,y)
    plt.xlabel("Time")
    plt.ylabel("Average length of match")
    plt.show()
    plt.cla()

    plt.plot(x,y_r)
    plt.xlabel("Time")
    plt.ylabel("Average reward for a1")
    plt.show()

    plt.plot(x, w_s)
    plt.xlabel("Time")
    plt.ylabel("Average win percent for a1")
    plt.show()
    plt.cla()

    plt.plot(x, l_s)
    plt.xlabel("Time")
    plt.ylabel("Average loss percent for a1")
    plt.show()
    plt.cla()
    
    # plt.savefig(f"round{66+k}.png")

    # df = pd.DataFrame(list(zip(actions, learning_agent_positions, enemy1_positions, enemy2_positions)), columns=['action', 'agent_pos', 'enemy1_pos', 'enemy2_pos'])
    # df.to_csv(f'ACTIONS_NOV17DISTANCED_{k}.csv', index=False)

    print()
    print("EXPERIMENT COMPLETE")

    print("Iteration", k)

    # print("Percentage new states", a1.new_states/a1.total_states)

    print("Overall time versus time spent in inference", time.time() - start, a1.time_in_inference)

    print("Action counts", a1.world.action_count)

    print("Times used cache/used model", a1.times_used_cached, a1.times_used_model)

    print("Win ratio", sum(w_s)/len(w_s))

    print("Loss ratio", sum(l_s)/len(l_s))

    # print("Win ratio", sum(y_r)/len(y_r))

    # a1.save_memories()
    # a1.refit_based_on_memories()
    # a1.refit_model()
    # a1.value_approximator.model.save("model/DEC14")

    lt_wr.append(sum(w_s)/len(w_s))
    lt_lr.append(sum(l_s)/len(l_s))
    r_s.append(sum(y_r)/len(y_r))
    x_s.append(k)

    # # print(master_model.get_weights())

    # for machine in machines:
    #     machine.agent.value_approximator.model.save(f"model/F22SEP30SMALLBRAINROUND{66+k}FINAL"+machine.name)

    # machine.agent.value_approximator.model.save(f"model/NOV17DISTANCED{k}")



plt.cla()
plt.plot(x_s,r_s)
# print(sum(r_s)/len(r_s))
plt.xlabel("Round")
# plt.xlabel(f"d (m={M})")
plt.ylabel("Average reward for a1")
plt.show()

plt.cla()
plt.plot(x_s,lt_wr)
plt.xlabel("Round")
# plt.xlabel(f"d (m={M})")
plt.ylabel("Average win-ratio for a1")
plt.show()

plt.cla()
plt.plot(x_s,lt_lr)
plt.xlabel("Round")
# plt.xlabel(f"d (m={M})")
plt.ylabel("Average loss-ratio for a1")
plt.show()

plt.cla()
plt.plot([i for i in range(len(fitting_losses))],fitting_losses)
plt.xlabel("Round")
# plt.xlabel(f"d (m={M})")
plt.ylabel("Model fitting loss for a1")
plt.show()



