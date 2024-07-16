import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import cv2
from matplotlib.animation import FuncAnimation

# Verificar se a GPU está disponível
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hiperparâmetros
num_episodes = 500
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
target_update = 10
max_memory = 10000
batch_size = 64

env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
n_actions = env.action_space.n
state_dim = env.observation_space.shape

# Adicionando uma linha para verificar o tamanho do estado
print(f"Dimensões do estado: {state_dim}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def feature_size(self, input_shape):
        x = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        x = self.conv1(x)
        print(f"After conv1: {x.size()}")
        x = self.conv2(x)
        print(f"After conv2: {x.size()}")
        x = self.conv3(x)
        print(f"After conv3: {x.size()}")
        return x.view(1, -1).size(1)

policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=max_memory)
loss_fn = nn.MSELoss()

def preprocess_state(state):
    state = np.transpose(state, (2, 0, 1))  # Transpor para C x H x W
    state = state / 255.0  # Normalizar
    return state

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()

def optimize_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = loss_fn(current_q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(env, num_episodes):
    epsilon = epsilon_start
    rewards = []
    
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")

    def update(frame):
        ax.clear()
        sns.lineplot(x=range(len(rewards)), y=rewards, ax=ax)
        ax.set(xlabel='Episode', ylabel='Total Reward', title='Rewards over Episodes')
        plt.draw()

    ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show(block=False)
    
    for episode in range(num_episodes):
        state = preprocess_state(env.reset()[0])
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state
            optimize_model()

            frame = env.render()
            cv2.imshow("River Raid", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    cv2.destroyAllWindows()
    env.close()
    return rewards

if __name__ == "__main__":
    train(env, num_episodes)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import cv2
from matplotlib.animation import FuncAnimation

# Verificar se a GPU está disponível
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Hiperparâmetros
num_episodes = 500
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
target_update = 10
max_memory = 10000
batch_size = 64

env = gym.make("ALE/Riverraid-v5", render_mode="rgb_array")
n_actions = env.action_space.n
state_dim = env.observation_space.shape

# Adicionando uma linha para verificar o tamanho do estado
print(f"Dimensões do estado: {state_dim}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def feature_size(self, input_shape):
        x = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
        x = self.conv1(x)
        print(f"After conv1: {x.size()}")
        x = self.conv2(x)
        print(f"After conv2: {x.size()}")
        x = self.conv3(x)
        print(f"After conv3: {x.size()}")
        return x.view(1, -1).size(1)

policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=max_memory)
loss_fn = nn.MSELoss()

def preprocess_state(state):
    state = np.transpose(state, (2, 0, 1))  # Transpor para C x H x W
    state = state / 255.0  # Normalizar
    return state

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()

def optimize_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = loss_fn(current_q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(env, num_episodes):
    epsilon = epsilon_start
    rewards = []
    
    fig, ax = plt.subplots()
    sns.set(style="whitegrid")

    def update(frame):
        ax.clear()
        sns.lineplot(x=range(len(rewards)), y=rewards, ax=ax)
        ax.set(xlabel='Episode', ylabel='Total Reward', title='Rewards over Episodes')
        plt.draw()

    ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show(block=False)
    
    for episode in range(num_episodes):
        state = preprocess_state(env.reset()[0])
        total_reward = 0
        done = False
        
        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state
            optimize_model()

            frame = env.render()
            cv2.imshow("River Raid", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    cv2.destroyAllWindows()
    env.close()
    return rewards

if __name__ == "__main__":
    train(env, num_episodes)

