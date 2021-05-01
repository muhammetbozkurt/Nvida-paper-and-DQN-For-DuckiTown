from PIL import Image
import sys
import pandas as pd

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
import time

record_flag = False
current_state = None
mem_counter = 0

memory = {"x": [], "y": [], "x_button": [], "y_button": [], "image_name": []}
#olur da yanlışlıkla birden ffazla yaratılırsa diye
mem_counter = 0

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

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global record_flag
    global current_state
    global mem_counter

    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    act_copy = np.array(action)

    if key_handler[key.UP]:
        action += np.array([0.11, 0.0])
        act_copy += np.array([1, 0])
    if key_handler[key.DOWN]:
        act_copy -= np.array([1, 0])
        action -= np.array([0.11, 0])
    if key_handler[key.LEFT]:
        act_copy += np.array([0, 1])
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        act_copy -= np.array([0, 1])
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])


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

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5



    if(record_flag):
        im_name = f"{time.time()}".replace(".","_")+".png"
        im = Image.fromarray(current_state)

        im.save(f"images/{im_name}")
        memory["x"].append(v1)
        memory["y"].append(v2)
        memory["x_button"].append(act_copy[0])
        memory["y_button"].append(act_copy[1])
        memory["image_name"].append(im_name)

    current_state, reward, done, info = env.step(action)
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    if(key_handler[key.R]):
    	record_flag = True

    if(key_handler[key.S]):
    	record_flag = False
    	df = pd.DataFrame(memory)

    	df.to_csv(f"data/{mem_counter}.csv")
    	#memory yi temizle
    	for header in memory:
    		memory[header] = []
    	mem_counter += 1

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()