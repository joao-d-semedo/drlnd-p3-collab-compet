[//]: # (Image References)

[image1]: Performance.png "Performance"

# Project 3: Collaboration and Competition

## Learning Algorithm

The training procedure leverages the DDPG architecture. As suggested in the project description, to adapt DDPG to train multiple agents, we first noted that each agent receives its own, local observation. Thus, we can easily adapt the code to simultaneously train both agents through self-play. In our case, each agent used the same actor network to select actions, and the experience was added to a shared replay buffer.

At a high level, the training loop involves:

1. Environment interaction: the agents interact with the environment using the current actor network (with an additive exploration noise process) and store the outcomes of their interactions in memory (the shared experience replay buffer).

2. Network update: Every few interactions (controlled by the hyperparameter `UPDATE_EVERY` in `ddpg_agent.py`) the agent draws experiences from memory (uniformly at random) and trains the actor and critic networks using the corresponding target networks.

For full details, see [original publication](https://arxiv.org/abs/1509.02971).

### Architectures

The function approximator network employed in this solution is a simple multi-layer perceptron (MLP). The implementation provides options for the activation types, as well as the number and size of the hidden layers. Furthermore, this implementation allows the user to specify an input normalization layer, as well as a custom distribution of the weights of the final linear layer of the network. The usage of input normalization only applies to the environment state (since no detailed description of the state was provided, input normalization provides an effective way to control the input distributions, which is crutial for stable training).

The default parameters for the actor network are:
- `sizes = [state_size, 128, 128, action_space_size]`, which defines a MLP with two hidden layers with 128 and 128 units respectively (number of input and output units are matched to the environment, 24 and 2 in this case).
- `activation = torch.nn.ReLU()`, which sets the activation functions between layers as rectified linear units (ReLU).
- `input_norm` was set to use Batch Normalization
- `output_layer_init` was set to `1e-3`, to stabilize early training
- `output_activation` was set to `nn.Tanh`.

The default parameters for the critic network are:
- `sizes = [state_size+action_size, 128, 128, 1]`, which defines a MLP with two hidden layers with 128 and 128 units respectively (number of input units is matched to the environment, 26 in this case).
- `activation = torch.nn.ReLU()`, which sets the activation functions between layers as rectified linear units (ReLU).
- `input_norm` was set to use Batch Normalization (only applied to the states)
- `output_layer_init` was set to `1e-3`, to stabilize early training
- `output_activation` was not set.

### Agents

The agent object contains both actor and critic MLP networks (`agent.actor_local` and `agent.critic_local`), the target actor and critic MLP networks (`agent.actor_target` and `agent.critic_target`), and the memory (the experience replay buffer, `agent.memory`). All these components are shared between the two agents.

The agent initialization receives the following inputs:
- `num_agents` is the number of agents present in the environment
- `state_size` is the size of the state space
- `action_size` is the size of the action space
- `seed` is the random seed to be used
- `noise_process_std=0.2` controls the magnitude of the added exploration noise
- `noise_process_theta=0.15` controls the autocorrelation of the added exploration noise

The following hyperparameters in `ddpg_agent.py` control the two steps of the training procedure outlined above.
- `BUFFER_SIZE = 1e6` controls the size of the experience replay buffer
- `BATCH_SIZE = 1024` controls the number of experiences drawn from the replay buffer whenever a DQN update is triggered. This large batch size was helpful in conjunction with the use of Batch Norm
- `ACTOR_LR = 1e-3` is the learning rate used when updating the weights of the local actor network
- `CRITIC_LR = 1e-3` is the learning rate used when updating the weights of the local critic network
- `UPDATE_EVERY = 20` controls how frequently to trigger a network update step (1 means the network weights are updated after every interaction with the environment, 20 means the weights are updated every 20 interactions, etc)
- `NUM_UPDATES = 5` controls the number of network update steps executed everytime network updating is triggered

### Training

- `env` is the environment to be used (the Tennis environment in this case)
- `brain_name` is the name of the default environment brain to be used
- `agent` is the DDPG agent to be trained
- `gamma = .99` is the discont factor used when computing returns (between 0 and 1)
- `tau = 1e-3` controls the rate at which the target DQN is updated (between 0 and 1: 0 means the target DQN is never updated, 1 means the target network is always equal to the main network)
- `noise_decay` controls the decay of the amount of exploration noise added during training
- `n_episodes = 1000` controls the total number of episodes used for training
- `max_t = 1000` controls the maximum duration of an episode

The exploration noise process used here is an Ornstein-Uhlenbeck Process and the implementation was adapted from [this repository](https://github.com/ShangtongZhang/DeepRL).

## Learning Performance

Bellow is the result of one training run, showing the average total rewards over 100 episodes (in orange) and the per episode total rewards (in blue) over the course of 3500 episodes. In this instance, the agent needed 2996 episodes to solve the environment. It would eventually reach a maximum in-training performance above 0.9. Note that these performance results are obtained during training, which means exploration noise is still being added to the agents' actions. When we loaded the best performing agents and ran them for 100 episodes we obtained an average score of 1.35. This also means that the agent likely solved the task before episode 2996, but needed to improve above that level to reach the performance target while exploration noise was still being added.

![Learning Performance][image1]

The network weights were saved in `checkpoint.pth` and can be loaded using `Tennis_trained_agent_viz.ipynb`.

## Ideas for Future Work

While the approach employed here was fairly successful, it represents a fairly straightforward implementation that can be improved in a number of ways. While these improvements might have a limited impact in solving this simple environment, they might prove much more important when tackling more complex environments (for example, learning using pixel-level representation of states):
- Further hyper-parameter tuning - There is still a lot more to explore here and while simple, this approach already includes about a dozen hyperparameters. One potential direction would be to explore changing some of these hyperparameters during training, potentially speeding up early trainig while allowing the training to remain stable later on.
- Prioritized Experience Replay
- Try more recent state of the art approaches to multi-agent learning (here we only adapted DDPG)

