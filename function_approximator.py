from keras import Sequential, Model, Input
from keras.layers import Dense, Conv2D
import numpy as np

class FunctionApproximator():
    def __init__(self) -> None:
        input = Input(shape=(9)) # (wt, wr, at, ar, adt, bt, br, bdt, action)
        d1 = Dense(50,activation='relu')(input)
        d2 = Dense(50,activation='relu')(d1)
        output = Dense(1,activation="linear")(d2)

        self.model = Model(inputs=[input],outputs=output)
        self.model.compile(optimizer='adam',loss='mse')

# test = FunctionApproximator()
# x_test:np.ndarray = np.array([[1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0,1],[0,0,1,1,0,0,1,1,0]])
# y_test:np.ndarray = np.array([[10],[10],[10],[-10]])
# print(x_test.shape)
# print(y_test.shape)

# test.model.fit(
#     x=[x_test],
#     y=y_test
# )
# xhat = np.array([[1,0,1,0,1,0,1,0,1]])
# print(test.model.predict(xhat)[0][0])
# xhat = np.array([[0,0,1,1,0,0,1,1,0]])
# print(test.model.predict(xhat)[0][0])