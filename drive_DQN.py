from tensorflow.keras.models import load_model
from gym_duckietown.envs import DuckietownEnv

from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, Input
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam

import numpy as np
import cv2


INPUT_SHAPE = (76,76, 1)

index_action_map ={0 : [0.11, 0],
                   1 : [0.11, 1],
                   2 : [0.11, -1],
                   3 : [-0.11, 0],
                   4 : [-0.11, 1],
                   5 : [-0.11, -1],
                   6 : [0, 1],
                   7 : [0, -1],
                   8 : [0, 0]}


def build_model(output_shape = 9):
    model = Sequential()
    model.add(Input(shape=INPUT_SHAPE))
    model.add(Conv2D(32,(3,3), activation="relu"))
    model.add(Conv2D(32,(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16,(3,3), activation="relu"))
    model.add(Conv2D(16,(3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))#10

    model.add(Flatten()) #this converts our 3D feature maps to 1D array

    model.add(Dense(32, activation= "relu"))
    model.add(Dense(16, activation= "relu"))


    model.add(Dense(output_shape, activation='sigmoid'))  
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def convert_predict_to_action(action):
    
    wheel_distance = 0.102
    min_rad = 0.08
    
    action = index_action_map[action]
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        #v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2
    return action

def preprocess(state):
    state = cv2.cvtColor(state, 0)
    state = cv2.resize(state[200::],(76,76))
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY).reshape(1,76,76, 1)
    return state / 255

env = DuckietownEnv(
    seed=1,
    map_name="small_loop",
    draw_curve=True,
    draw_bbox=False,
    domain_rand=True,
    frame_skip=1,
    distortion=False,
    camera_rand=True,
    dynamics_rand=True,
)

current_state = env.reset()
env.render()

model = build_model()

model.load_weights("episode_30_step_36.h5")

done = False

while (True):

   
    
    
    action = np.argmax(model.predict(preprocess(current_state)))

    action_input_to_env = convert_predict_to_action(action)
    env.render()
    print(action_input_to_env)
    current_state, reward, done, info = env.step(action_input_to_env)
    if(done):
        env.reset()


print("\t\tThe")
print("\t\t\t\tEnd")