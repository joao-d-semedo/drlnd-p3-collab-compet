import numpy as np
import random
from collections import namedtuple, deque

from network import Actor, Critic
from random_process import OrnsteinUhlenbeckProcess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024       # minibatch size
ACTOR_LR = 1e-3         # actor network learning rate
CRITIC_LR = 1e-3        # critic network learning rate
UPDATE_EVERY = 20       # how often to update the network
NUM_UPDATES = 5         # num network updates

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, seed, noise_process_std=0.2, noise_process_theta=0.15):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor
        self.actor_local = Actor(
            sizes=[state_size, 128, 128, action_size],
            input_norm=nn.BatchNorm1d(state_size),
            output_layer_init=1e-3,
            output_activation=nn.Tanh,
            seed=seed
         ).to(device)
        self.actor_target = Actor(
            sizes=[state_size, 128, 128, action_size],
            input_norm=nn.BatchNorm1d(state_size),
            output_layer_init=1e-3,
            output_activation=nn.Tanh,
            seed=seed
         ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)
        
        # Critic
        self.critic_local = Critic(
            sizes=[state_size + action_size, 128, 128, 1],
            input_norm=nn.BatchNorm1d(state_size),
            output_layer_init=1e-3,
            seed=seed
         ).to(device)
        self.critic_target = Critic(
            sizes=[state_size + action_size, 128, 128, 1],
            input_norm=nn.BatchNorm1d(state_size),
            output_layer_init=1e-3,
            seed=seed
         ).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)

        self.exploration_noise = OrnsteinUhlenbeckProcess(size=(num_agents, action_size), std=noise_process_std, theta=noise_process_theta)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, states, actions, rewards, next_states, dones, gamma, tau):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 10*BATCH_SIZE:
                for _ in range(NUM_UPDATES):
                    experiences = self.memory.sample()
                    self.learn(experiences, gamma, tau)

    def act(self, states, exploring=False, noise_scale=1.0):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """

        states = torch.from_numpy(states).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states)
        self.actor_local.train()

        if exploring:
            actions += noise_scale*torch.from_numpy(self.exploration_noise.sample()).float().to(device)

        return np.clip(actions, -1, 1)

    def learn(self, experiences, gamma, tau):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        n_exp = states.shape[0]

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # next_Q = self.critic_target(torch.cat( (next_states, next_actions), dim=1 ))
            next_Q = self.critic_target(next_states, next_actions)

        Q_targets = rewards + gamma*next_Q*(1 - dones)
        
        Q_predictions = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(Q_predictions, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()

        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local,  self.actor_target,  tau)
        self.soft_update(self.critic_local, self.critic_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def reset(self):
        self.exploration_noise.reset_states()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def ddpg(env, brain_name, agent, n_episodes=2000, max_t=1000, gamma=0.99, tau=1e-3, noise_decay=1.0):
    """DDPG.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        std_start (float): starting value of std, for exploration noise process
        std_end (float): minimum value of std
        std_decay (float): multiplicative factor (per episode) for decreasing std
    """

    ep_scores = []                     # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    best_mean_score = -np.Inf
    solved = False
    noise_scale = 1.0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(agent.num_agents)
        for _ in range(max_t):
            actions = agent.act(states, exploring=True, noise_scale=noise_scale).numpy()
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            agent.step(states, actions, rewards, next_states, dones, gamma, tau)
            states = next_states
            scores += rewards
            if any(dones):
                break
        ep_score = scores.max()
        scores_window.append(ep_score) # save most recent score
        ep_scores.append(ep_score)     # save most recent score

        agent.reset()

        noise_scale *= noise_decay

        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.5f}'.format(i_episode, mean_score), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.5f}'.format(i_episode, mean_score))
        if not solved and mean_score>=0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode, mean_score))
            solved=True
        if i_episode >= 100 and mean_score > best_mean_score:
            torch.save(agent.actor_local.state_dict(), 'checkpoint.pth')
            best_mean_score = mean_score
    return ep_scores

