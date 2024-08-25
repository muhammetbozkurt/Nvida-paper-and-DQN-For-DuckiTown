# NVidia Paper and DQN For DuckiTown

## Deep Q Learning

Deep Q Learning combines deep learning with Q-learning, a reinforcement learning algorithm. To grasp this, one must first understand reinforcement learning, which mimics how intelligent beings act in an environment using an agent-environment analogy. A key concept in reinforcement learning is the Markov Decision Process (MDP), a mathematical framework used in dynamic programming and reinforcement learning to model decision-making under uncertainty.

In an MDP, an agent interacts with an environment by taking actions, receiving rewards, and learning from the outcomes. Our model learns through episodes, with each episode comprising multiple time steps. Initially, the policy network is randomly initialized. During each episode, the agent begins in an initial state, takes actions, and receives rewards, storing the experiences in replay memory. A sample batch from this memory is used to update the policy network using gradient descent, which minimizes the loss between current and target Q-values.

We apply the Bellman equation to update the Q-table, allowing the agent to refine its decisions by considering future rewards. Due to hardware limitations, including an NVIDIA 950M GPU, we faced challenges running simulations and training deep neural networks simultaneously, which affected the DQN model's performance.

## NVidia's End-To-End Steering Model

![NVidia Model](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/overall.PNG)

NVIDIA introduced a deep learning model that uses convolutional neural networks (CNNs) to map raw pixels directly to steering commands from a front camera. CNNs excel at recognizing patterns in visual data but require large datasets for effective learning. To address this, we applied image augmentation techniques, such as flipping, brightness adjustment, and shadow addition, to expand our dataset and prevent overfitting. The model consists of five convolutional layers and four dense layers, using ELU activation to mitigate vanishing gradient issues. The output, a steering angle between -1 and 1, is optimized using the Adam optimizer and mean-squared error as the loss function.

## Modified NVidia's End-To-End Steering Model

We modified NVIDIA’s model to function as a classifier instead of a regressor. The key changes include:
- Replacing the single output neuron with three neurons to represent left, right, and no movement, assuming constant forward velocity.
- Adding a softmax function to produce a probability distribution for the three output neurons.
- Changing the loss function to categorical cross-entropy to suit the multi-class classification task.

## Results

**Nvidia End-To-End Result:**

![NVidia Result](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/nvidia.gif)

**Modified Nvidia End-To-End Result:**

![Modified NVidia Result](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/custom.gif)

**DQN Result:**

Due to hardware limitations, we were unable to achieve satisfactory results with the DQN model.

![DQN Result](https://github.com/muhammetbozkurt/Nvida-paper-and-DQN-For-DuckiTown/blob/main/results/dqn.gif)
