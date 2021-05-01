import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
from random import sample, randint
import cv2
#https://ai.stackexchange.com/questio



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


from gym_duckietown.envs import DuckietownEnv


"""
Initialize replay memory capacity.
Initialize the policy network with random weights.
Clone the policy network, and call it the target network.
For each episode:

    Initialize the starting state.
    For each time step:
        Select an action.
            Via exploration or exploitation
        Execute selected action in an emulator.
        Observe reward and next state.
        Store experience in replay memory.
        Sample random batch from replay memory.
        Preprocess states from batch.
        Pass batch of preprocessed states to policy network.
        Calculate loss between output Q-values and target Q-values.
            Requires a pass to the target network for the next state
        Gradient descent updates weights in the policy network to minimize loss.
            After ceratin time steps, weights in the target network are updated to the weights in the policy network

"""

MIN_EPSILON = 0.05
#Initialize replay memory capacity.
MEMORY_SIZE_LIMIT = 1000
DISCOUNT = 0.99
MINIBATCH_SIZE = 16
TARGET_UPDATE = 1
EPISODES = 500
MAX_STEP_LIMIT = 500


#number of actions
#1 , 0
#1 , 1
#1 , -1
#etc
ACTIONS = 9

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


class DQN:

    def __init__(self):
        #Initialize the policy network with random weights.
        self.policy = self.build_model(INPUT_SHAPE)
        self.target = self.build_model(INPUT_SHAPE)
        #Clone the policy network, and call it the target network.

        self.policy.load_weights("episode_28_step_32.h5")

        self.target.set_weights(self.policy.get_weights())
        self.epsilon = 1
        self.replay_memory = []
        self.clone_models_counter = 0

    def build_model(self, input_shape, output_shape = 9):
        model = Sequential()
        model.add(Input(shape=input_shape))
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

    def epsilon_decay(self):
        if(self.epsilon > MIN_EPSILON):
            self.epsilon *= 0.9

    def append_to_replay_memory(self, experience):
        self.replay_memory.append(experience)
        if(len(self.replay_memory) > MEMORY_SIZE_LIMIT):
            self.replay_memory.pop(0)

    def preprocess(self, state):
        state = cv2.resize(state[200::],(76,76))
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY).reshape(76,76, 1)
        return state / 255

    def load_policy_model(self, path):
        self.policy.load_weights(path)
        self.target.load_weights(path)

    def get_action_from_policy(self, state):
        return self.policy.predict(state.reshape(-1, 76,76, 1))
    
    def _get_action_from_target(self, state):
        return self.target.predict(state.reshape(-1, 76,76, 1))

    def train(self, terminal_step, episode, step):

        if(len(self.replay_memory) < MINIBATCH_SIZE):
            return
        #Sample random batch from replay memory.
        minibatch = sample(self.replay_memory, MINIBATCH_SIZE)

        #Preprocess states from batch.
        #experience = (current_state, action, reward, done, next_state)
        current_states = np.array([self.preprocess(batch[0]) for batch in minibatch])
        next_states = np.array([self.preprocess(batch[4]) for batch in minibatch])

        #Pass batch of preprocessed states to policy network.
        #   Requires a pass to the target network for the next state
        current_q_values = self.get_action_from_policy(current_states)
        next_q_values = self._get_action_from_target(next_states)

        #X and y will be used to keep states and new q values for fitting policy model
        X = list()
        y = list()

        for index, (current_state, action, reward, done, new_current_state) in enumerate(minibatch):

            #Bellman Equation
            if not done:
                max_next_q = np.max(next_q_values[index])
                new_q = reward + DISCOUNT * max_next_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_q_values[index]
            current_qs[action] = new_q

            #i appended current_states[index] instead of current_state because it was preprocessed aldready (line 144)
            X.append(current_states[index])
            y.append(current_qs)

        #Gradient descent updates weights in the policy network to minimize loss.
        self.policy.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)
        
        if(terminal_step):
            self.clone_models_counter += 1

        #After ceratin time steps, weights in the target network are updated to the weights in the policy network
        if(self.clone_models_counter >= TARGET_UPDATE):
            self.target.set_weights(self.policy.get_weights())
            self.target.save_weights(f"models/episode_{episode}_step_{step}.h5")
            print(f"_{step}_"*3)
            print(f"_{episode}_"*3)
            print("*"*25)
            self.clone_models_counter = 0

if(__name__ == "__main__"):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    tf.device('/gpu:1')

    env = DuckietownEnv(
            seed=1,
            map_name="small_loop",
            draw_curve=True,
            draw_bbox=False,
            domain_rand=True,
            frame_skip=1,
            distortion=False,
            camera_rand=False,
            dynamics_rand=True,
        )
    env.reset()
    """
    For each episode:

        Initialize the starting state.
        For each time step:
            Select an action.
                Via exploration or exploitation
            Execute selected action in an emulator.
            Observe reward and next state.
            Store experience in replay memory.
            ---agent.train
            Sample random batch frmo replay memory.
            Preprocess states from batch.
            Pass batch of preprocessed states to policy network.
            Calculate loss between output Q-values and target Q-values.
                Requires a pass to the target network for the next state
            Gradient descent updates weights in the policy network to minimize loss.
                After ceratin time steps, weights in the target network are updated to the weights in the policy network
    """
    agent = DQN()

    for episode in range(EPISODES):
        current_state = env.reset()
        
        #current_state = np.array(env.state)
        done = False
        step = 0
        while (not done and step < MAX_STEP_LIMIT):
            #Select an action.
            #   Via exploration or exploitation
            if(np.random.random() < agent.epsilon):#exploration
                action = np.random.randint(0, ACTIONS)
            else:#exploitation
                curr_state = agent.preprocess(current_state)
                action = np.argmax(agent.get_action_from_policy(curr_state))

            action_input_to_env = convert_predict_to_action(action)

            #if(episode % 10 == 0):
            #    env.render()

            #Execute selected action in an emulator.
            #Observe reward and next state.
            next_state, reward, done, info =  env.step(action_input_to_env)

            #Store experience in replay memory.
            experience = (current_state, action, reward, done, next_state)
            agent.append_to_replay_memory(experience)
            
            terminal_step = done or (step == MAX_STEP_LIMIT - 1)
            
            agent.train(terminal_step, episode, step)
            
            current_state = next_state

            step += 1
        
        agent.epsilon_decay()

        print("*"*25)
        print("*"*25)
        print("*"*25)
        print("*"*25)
