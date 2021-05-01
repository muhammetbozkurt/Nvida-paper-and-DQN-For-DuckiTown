from tensorflow.keras.models import load_model
from gym_duckietown.envs import DuckietownEnv

from tensorflow.keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam

import numpy as np
import cv2


SHAPE = (160, 320, 3)


def build_model():
    model=Sequential()
    model.add(Lambda(lambda x: x/255.,input_shape=(160,320,3)))
    #ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    model.add(Conv2D(24,(5,5),activation="elu",strides=(2,2)))
    model.add(Conv2D(36,(5,5),activation="elu",strides=(2,2)))
    model.add(Conv2D(48,(5,5),activation="elu",strides=(2,2)))
    model.add(Conv2D(64,(5,5),activation="elu"))
    model.add(Conv2D(64,(5,5),activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100,activation="elu"))
    model.add(Dense(50,activation="elu"))
    model.add(Dense(10,activation="elu"))
    model.add(Dense(3,activation="softmax"))


    model.summary()
    return model

def crop_resize(image, result_shape = (160, 320)):
    preprocessed_image = image[200::]
    preprocessed_image = cv2.resize(preprocessed_image, result_shape, cv2.INTER_AREA)
    return preprocessed_image.reshape(1, result_shape[1], result_shape[0], 3)

def stay_in_bounds(v2):
    res = v2 if(1>v2) else 0.9
    res = v2 if(-1<v2) else -0.9
    return res

def action_creator(v1, v2):
    wheel_distance = 0.102
    min_rad = 0.08

    
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        #v1 += delta_v
        v2 -= delta_v


    return np.array([v1, v2])

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

model.load_weights("trio_weights_180_2.h5")

done = False

while (not done):

    current_state = crop_resize(current_state)

    v2 = model.predict_classes(current_state)[0] -1
    env.render()
    action = action_creator(0.11, v2)
    print(action)
    current_state, reward, done, info = env.step(action)



print("\t\tThe")
print("\t\t\t\tEnd")