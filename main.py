import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import random
from collections import namedtuple
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Assuming the action space is between -1 and 1
        return x

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
input_size = 4
hidden_size = 128
output_size = 2
tau = 0.005  # for soft update of target parameters
gamma = 0.99  # discount factor
buffer_size = 1000000
minibatch_size = 64

# Main model and target model
policy_net = PolicyNetwork(input_size, hidden_size, output_size)
target_policy_net = copy.deepcopy(policy_net)
q_net = QNetwork(input_size + output_size, hidden_size, 1)
target_q_net = copy.deepcopy(q_net)

# Optimizers
policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
q_optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Soft update function
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# Assume we have a function to add noise to the actions for exploration purposes
def add_exploration_noise(action):
    # Add exploration noise here (e.g., Ornstein-Uhlenbeck process)
    return action

# Assume we have a function to interact with the environment and return a new state and reward
def env_step(state, action):
    # Implement environment interaction here
    return next_state, reward, done


### updates ###

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
num_episodes = 1000  # Set the number of episodes you want to train for
max_steps_per_episode = 200  # Set the maximum number of steps allowed per episode

def add_exploration_noise(action, noise_std=0.2):
    noise = np.random.normal(0, noise_std, size=action.shape)
    return np.clip(action + noise, -1.0, 1.0)  # Assuming the action space is between -1 and 1

env = gym.make('Humanoid-v5')

# Training loop would go here, including:
# - Interacting with the environment
# - Storing transitions in the replay buffer
# - Sampling from the buffer to update the Q-network and policy network
# - Periodically updating the target networks using soft_update

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_steps_per_episode):
        # Select action according to policy and add exploration noise
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = policy_net(state_tensor).detach().numpy()[0]
        action = add_exploration_noise(action)

        # Execute action in the environment
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # Store transition in the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization if the replay buffer is large enough
        if len(replay_buffer) > minibatch_size:
            transitions = replay_buffer.sample(minibatch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.FloatTensor(batch.state)
            action_batch = torch.FloatTensor(batch.action)
            reward_batch = torch.FloatTensor(batch.reward)
            next_state_batch = torch.FloatTensor(batch.next_state)
            done_batch = torch.FloatTensor(batch.done)

            # Compute the target Q value
            target_actions = target_policy_net(next_state_batch)
            target_Q_values = target_q_net(next_state_batch, target_actions).detach()
            expected_Q_values = reward_batch + (gamma * target_Q_values * (1 - done_batch))

            # Get current Q values
            current_Q_values = q_net(state_batch, action_batch)

            # Compute Q network loss
            q_loss = F.mse_loss(current_Q_values, expected_Q_values.unsqueeze(1))

            # Optimize the Q network
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Compute policy loss
            policy_loss = -q_net(state_batch, policy_net(state_batch)).mean()

            # Optimize the policy network
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Update the target networks
            soft_update(target_policy_net, policy_net, tau)
            soft_update(target_q_net, q_net, tau)

        if done:
            break

    print(f"Episode {episode}: Total reward = {episode_reward}")