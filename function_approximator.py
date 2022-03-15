from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Conv2D
import numpy as np

class FunctionApproximator():
    def __init__(self) -> None:
        input = Input(shape=(8,)) # (wt, wr, at, ar, adt, bt, br, bdt)
        d1 = Dense(50,activation='relu')(input)
        d2 = Dense(50,activation='relu')(d1)
        output = Dense(1,activation="linear")(d2)

        self.model = Model(inputs=[input],outputs=output)
        self.model.compile(optimizer='adam',loss='mse')