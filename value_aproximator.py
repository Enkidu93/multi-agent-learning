from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Concatenate
import numpy as np

class CFValueApproximator():
    def __init__(self):
        x_in = Input(shape=(6,7,1))
        x_1 = Conv2D(12,(4,4),activation='tanh',kernel_initializer='he_uniform')(x_in)
        x_3 = Flatten()(x_1)
        x_4 = Dense(10,activation='relu')(x_3)
        x = Model(inputs=x_in,outputs=x_4)

        y_in = Input(shape=(7,))

        c = Concatenate()([x.output,y_in])

        y_1 = Dense(6,activation='tanh')(c)
        y_2 = Dense(1,activation="linear")(y_1)

        self.model = Model(inputs=[x_in,y_in], outputs=y_2)
        
        self.model.compile(optimizer='adam',loss='mse')

        # self.data:tuple[np.array,np.array] = None

# cfva = CFValueApproximator()
# cfva.model.fit(
#         [
#             np.array([[[1,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,1,0]]]),
#             np.array([[0,0,0,0,0,0,0]])
#         ],
#         np.array([[0.89]])
#     )
# print(cfva.model.predict(        (
#             np.array([[[1,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,1,0]]]),
#             np.array([[0,0,0,0,0,0,0]])
#         ))) 


