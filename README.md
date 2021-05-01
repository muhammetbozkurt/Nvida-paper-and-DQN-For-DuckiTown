# Nvida-paper-and-DQN-For-DuckiTown

## Deep Q Learning

Deep Q learning is a synthesis of deep learning and Qlearning.
Moreover, Q-learning is a reinforcement learning
algorithm. To clarify these sentences, reinforcement learning
must be understood. Reinforcement learning mainly uses
agent-environment analogy to imitate how intelligent creatures
should take actions in an environment.
One of the most crucial ideas behind reinforcement learning
is Markov Decision Processes. Markov Decision process is
a discrete time stochastic control process which examines
the decision making mathematically, randomly and under
the control of the decision maker. That process is used in
optimization problems solved with dynamic programming and
reinforcement learning. After implementation of tasks in an
Markov decision process, agent can take actions, get rewards
regarding that action and learn from actions and rewards in
an environment which is defined with the markov decision
process.
Our model learns within episodes and each time steps in
an episode. Before the learning phase, the policy network
initialized with random weights. At each episode, agent start
with the initial state. After that the agent takes an action in
each time step. Agent observer reward and next state with
the taken action. That experience is stored in the replay
memory. Sample batch taken from replay memory randomly.
That state preprocessed and batch regards to state passed to
policy network. After that process, loss value calculated with
output Q values and target Q values. Gradient descent updates
weights in the policy network to minimize loss. After certain
timesteps, weight in the target network is updated to the weight
in the policy network.
We implement the Bellman equation to update the Q table
after each step. The agent updates the current Q value with the
predicted future reward. That assumes the agent has taken the
highest Q value action. For that, agent look at every action for
a specific state and pick a state-action pair with the highest Q
value. Then gradient descent used to minimize loss between
current Q value and target Q value.
While training the DQN model we had problems about
insufficient hardware systems. Due to our environment not
compatible with colab, we have to use our computers and
the most powerful gpu we have is nvidia 950m which is
not sufficient for running simulation and training deep neural
network models simultaneously.



## NVidia's End-To-End Steering Model


![figure](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/overall.PNG)

Nvidia
published an article explaining its deep learning model, seen
in figure 2, that uses convolutional neural networks to map
raw pixels directly to steering commands from a single front
camera. Related to the nvidia’s research paper, convolutional
neural networks provide high accuracy on recognizing patterns
in visual data. As it is known that deep neural networks
require huge amounts of data to learn the relationship between
input data and output. Therefore, we applied image
augmentation techniques to produce more data from collected
original data. One of the advantages of image augmentation
is that it helps model generalize relation between data and
its label. Moreover, it helps to prevent overfitting. We used
techniques such as flipping image, changing its brightness,
adding shadows in image augmentation phase. Model has 5
convolutional and 4 dense layers. These layers use ELU as
an activation function to handle vanishing gradient problems.
Our model gets an input which has a shape of 160,320,3.
While the first 2 dimensions are sizes of images as pixels,
the third dimension is command of drive. Output shape is
1 and output of model is steering angle. This model gives
1,922,129 trainable parameters in total. At the train process,
Mean-squared was used as loss function because our output is
a scalar value between -1 and 1. Moreover, Adam optimizer
function as an optimization function. We trained our model in
10 epochs with 10000 steps for every epoch. In every epoch, 320000 data points were created using our image generator
function which is our custom image augmentation function.



## Modified NVidia's End-To-End Steering Model

This section’s works were done by Muhammet Bozkurt.
In this section, we will explain the model that we developed
by making some minor changes in the deep learning model
suggested by Nvidia in their article as seen in figure 3. We
tried to convert Nvidia’s model to a classifier instead of a
regressor.
We can list the changes we made to Nvidia’s model as
follows:
* Unlike Nvidia’s model which has one neuron at its
output layer, our model’s output layer has three neurons
representing moving left or right and not moving. Assum-
Fig. 3. Customized Nvidia Model.
ing that our agent always moves forward with constant
velocity.
* A Sigmoid function was added to the output of our custom
function as activation function because the addition
of the Sigmoid function ensures that the output of our
model is a probability distribution. In other words, the
outputs of the 3 neurons in the last layer will be between
0 and 1 and their aggregation will be adjusted to 1.
* Last chance we made is changing loss function to categorical
cross-entropy which is for multi class classification
tasks since we have 3 classes.

## Results

__Nvidia end-to-end Result__:

![Figure](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/nvidia.gif)

__Modified Nvidia end-to-end Result__:

![Figure](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/custom.gif)

__DQN Result__:

As stated in DQN section, our hardware was not capable to
train DQN model. Thus, sufficient
results for DQN model could not be achieved to get .

![Figure](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/dqn.gif)
