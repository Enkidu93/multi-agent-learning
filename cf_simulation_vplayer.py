from spacy import load
from connect_four import ConnectFour
from cf_agent import CFAgent
from cf_deepagent import DeepCFAgent
import matplotlib.pyplot as plt
from math import sqrt
from keras.models import load_model

cf = ConnectFour()
a1 = DeepCFAgent(cf,gamma=0.9,epsilon=0.0)

a1.value_approximator.model = load_model('model2')

N=5

num_steps=0
a1_wins = 0
player_wins = 0
for n in range(N):
    cf.visualize()
    print(a1.world.players.get(-1))
    while(not cf.episode_complete):

        a1.take_action()
        if cf.episode_complete:
            print("Phil wins!")
            a1_wins+=1
            break

        num_steps+=1
        
        cf.visualize()
        
        answer = -1
        while(answer == -1):
            answer = int(input("Where do you want to go?..."))
            if answer not in cf.actions:
                print("Invalid move...please try again.")
                answer = -1

        cf.step(answer)        

        if cf.episode_complete:
            print("You win!")
            player_wins+=1

        num_steps+=1

        cf.visualize()
    
    a1.reset()

print(a1_wins/N)
print(player_wins/N)

# plt.plot(x,y)
# plt.show()

# answer = input("Would you like to save a1's model?")
# if answer == 'Y':
#     filename = input("Enter a filename...")
#     a1.value_approximator.model.save(filename)
#     a1.save("qvals_a1")

# answer = input("Would you like to save a2's model?")
# if answer == 'Y':
#     filename = input("Enter a filename...")
#     a2.value_approximator.model.save(filename)
