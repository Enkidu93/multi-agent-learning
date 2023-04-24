from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np

# x = [-396,-396,-396,-396,-396,-396,-396,-396,-396,10.666,7.75,-200.75,-397,-391.25,-201.25,-200.0,10.0,6.667]
# y = [i for i in range(len(x))]
# z = [2.6025]*len(x)
# r = [13.025]*len(x)
# plt.plot(y,r)
# plt.plot(y,z)
# plt.plot(y,x)
# plt.show()


x = []
y = []
for k in range(1,250,1):
    model = load_model(f"model/NOV3TEST{k}")
    y.append(np.abs(np.array(model.get_weights()[0])).mean())
    x.append(k)
plt.plot(x,y)
plt.show()