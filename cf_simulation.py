from connect_four import ConnectFour
from cf_agent import CFAgent
from cf_deepagent import DeepCFAgent
import matplotlib.pyplot as plt
from math import sqrt
from keras.models import load_model

cf = ConnectFour()
a1 = DeepCFAgent(cf,gamma=1,alpha=0.2,epsilon=0.75)
a2 = DeepCFAgent(cf,gamma=1,alpha=0.2,epsilon=0.75)
# a1 = DeepCFAgent(cf,gamma=1.0,epsilon=0.0)
# a2 = DeepCFAgent(cf,gamma=1.0,epsilon=0.0)

# a1.value_approximator.model = load_model('trythis')
# a2.value_approximator.model = load_model('trythat')

VISUALIZE = False
N=10_000

x = [i for i in range(1,N//100)]
y = []
num_steps=0
a1_wins = 0
a2_wins = 0

if VISUALIZE:
    cf.visualize()
for n in range(N):
    while(not cf.episode_complete):

        a1.take_action()
        if cf.episode_complete:
            if VISUALIZE:
                print("A1 wins!")
            a1_wins+=1
            break

        num_steps+=1
        
        if VISUALIZE:
            cf.visualize()
        
        a2.take_action()
        if cf.episode_complete:
            if VISUALIZE:
                print("A2 wins!")
            a2_wins+=1

        num_steps+=1

        if VISUALIZE:
            cf.visualize()

    if n%100==0 and n!=0:
        print(n)
        y.append(num_steps/100)
        num_steps = 0
    
    # if (sqrt(n)//1)%5==0 and n!=0:
    #     a1.refit_model()
    #     a2.refit_model()

    if n%100==0 and n!=0:
        a1.refit_model()
        a2.refit_model()
    
    if n==(N-2):
        VISUALIZE = True

    a1.reset() 
    a2.reset()

print(a1_wins/N)
print(a2_wins/N)

plt.plot(x,y)
plt.show()

answer = input("Would you like to save a1's model?")
if answer == 'Y':
    filename = input("Enter a filename...")
    a1.value_approximator.model.save(filename)
    a1.save("qvals_a1")

answer = input("Would you like to save a2's model?")
if answer == 'Y':
    filename = input("Enter a filename...")
    a2.value_approximator.model.save(filename)
