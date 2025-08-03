import pandas as pd
import numpy as np
import json
import random
import torch
from uuid import uuid4
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


# dias = ['LUN', 'MAR', 'MIE', 'JUE', 'VIE']
# franjas = ['07:00 - 08:30', '08:45 - 10:15', '10:30 - 13:00']
#
# with open('obs.json', 'r') as file:
#     obs_consol = json.load(file)
#

class Agent:
    def __init__(self):
        pass

class Env:
    def __init__(self):
        self.data = pd.read_excel('data.xlsx', dtype='string')
        # States
        self.states= ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110']
        # Actions
        self.actions = ['up', 'down']
        # self.data.CANT_ALUMN = self.data.CANT_ALUMN.astype('int32')
        # self.data.AFORO = self.data.AFORO.astype('int32')
        # self.agents = list(self.data['AGENT'])
        self.data_prime = self.data.copy()
        self.aulas = ['101', '102', '103', '104', '105']
        self.aula_aforo = {'101': 15, '102': 15, '103': 20, '104': 25, '105': 25}
        self.nivel = {'BAS': 1, 'INT': 1}

    def get_reward(self):
        diff = self.data['AFORO'] - self.data['CANT_ALUMN']
        # diff_prime = self.data_prime['AFORO'] - self.data_prime['CANT_ALUMN']
        negative_diff = np.where(diff < 0, diff, 0)
        # negative_diff_prime = np.where(diff_prime < 0, diff_prime, 0)
        # negative_diff_prime.sum()
        return negative_diff.sum()

    def get_obs(self):
        data = self.data.copy()
        data['DIFF'] = data['AFORO'] - data['CANT_ALUMN']
        data['IDX_NIVEL'] = data['NIVEL'].map(self.nivel)
        obs = []
        for agent in self.agents:
            data_agent = np.array(data.loc[data.AGENT == agent, ['DIFF', 'IDX_NIVEL']])
            obs.append(data_agent.flatten())
        return obs

    def get_next_obs(self, aulas: list):

        for idx_agent, idx_aula in enumerate(aulas):
            agent = self.agents[idx_agent]
            aula = self.aulas[idx_aula]
            aforo = self.aula_aforo[aula]
            self.data_prime.loc[
                self.data_prime.AGENT == agent,
                ['AULA', 'AFORO']] = [aula, aforo]

        data_prime = self.data_prime.copy()
        data_prime['DIFF'] = data_prime['AFORO'] - data_prime['CANT_ALUMN']
        data_prime['IDX_NIVEL'] = data_prime['NIVEL'].map(self.nivel)
        obs = []
        for agent in self.agents:
            data_prime_agent = np.array(data_prime.loc[data_prime.AGENT == agent, ['DIFF', 'IDX_NIVEL']])
            obs.append(data_prime_agent.flatten())
        return obs

    def get_state(self, data: pd.DataFrame):

        data_01 = data.copy()
        data_01['DIFF'] = data_01['AFORO'] - data_01['CANT_ALUMN']
        agent_state = []

        for agent in self.agents:
            data_agent = np.array(data_01.loc[data_01.AGENT == agent, 'DIFF'])
            agent_state.append(data_agent[0])

        id_class = data_01.groupby(
            ['ID_CLASS'], as_index=False
        ).agg(UNIQUE=pd.NamedAgg(
            column='AULA',
            aggfunc=lambda x: 1 if len(np.unique(x)) == 1 else 0))

        return list(id_class.UNIQUE) + agent_state

    def update_data(self):
        self.data = self.data_prime.copy()


# Define individual Q-network for each agent
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the mixing network
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        # Hypernetwork for generating weights
        self.hyper_w1 = nn.Linear(state_dim, 64 * n_agents)
        self.hyper_w2 = nn.Linear(state_dim, 64 * 1)
        self.hyper_b1 = nn.Linear(state_dim, 64)
        self.hyper_b2 = nn.Linear(state_dim, 1)

    def forward(self, q_values, state):
        # q_values: (batch_size, n_agents)
        # state: (batch_size, state_dim)
        batch_size = q_values.size(0)

        # Generate weights and biases using hypernetworks
        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, 64)
        b1 = self.hyper_b1(state).view(batch_size, 1, 64)
        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, 64, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # First layer of mixing
        x = torch.relu(torch.bmm(q_values.unsqueeze(1), w1) + b1)
        # Second layer of mixing
        q_tot = torch.bmm(x, w2) + b2
        return q_tot.squeeze(2)  # (batch_size, 1)


# QMIX Agent
class QMIX:
    def __init__(self, n_agents, obs_dim, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.n_agents = n_agents
        self.gamma = gamma
        self.q_networks = [QNetwork(obs_dim, action_dim) for _ in range(n_agents)]
        self.target_q_networks = [QNetwork(obs_dim, action_dim) for _ in range(n_agents)]
        self.mixing_network = MixingNetwork(n_agents, state_dim)
        self.target_mixing_network = MixingNetwork(n_agents, state_dim)

        # Copy weights to target networks
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            target_q_net.load_state_dict(q_net.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        self.q_optimizers = [optim.Adam(q_net.parameters(), lr=lr) for q_net in self.q_networks]
        self.mixing_optimizer = optim.Adam(self.mixing_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=10000)
        self.avg_reward = 0.0  # Running estimate of average reward
        self.avg_reward_alpha = 0.01  # Step size for updating average reward

    def act(self, observations, epsilon):
        actions = []
        for i, obs in enumerate(observations):
            if random.random() < epsilon:
                action = random.randint(0, self.q_networks[i].fc3.out_features - 1)
            else:
                # unsqueeze(0) to add one dimension. [] => [[]]
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                q_values = self.q_networks[i](obs_tensor)
                action = q_values.argmax().item()
            actions.append(action)
        return actions

    def store(self, state, observations, actions, reward, next_state, next_observations):
        self.replay_buffer.append((state, observations, actions, reward, next_state, next_observations))

    def update_avg_reward(self, reward):
        self.avg_reward = self.avg_reward + self.avg_reward_alpha * (reward - self.avg_reward)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, obs, actions, rewards, next_states, next_obs = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)

        # unsqueeze(1) add one dimension at dim=1
        # [1.2, 3.4] => [[1.2], [3.4]]
        # (batch_size,) => (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        # dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute current Q-values
        q_values = []
        for i, q_net in enumerate(self.q_networks):
            obs_i = torch.FloatTensor([o[i] for o in obs])
            q_i = q_net(obs_i)
            q_values.append(q_i.gather(1, actions[:, i].unsqueeze(1)))
        q_values = torch.cat(q_values, dim=1)  # (batch_size, n_agents)
        q_tot = self.mixing_network(q_values, states)

        # Compute target Q-values
        next_q_values = []
        for i, target_q_net in enumerate(self.target_q_networks):
            next_obs_i = torch.FloatTensor([o[i] for o in next_obs])
            next_q_i = target_q_net(next_obs_i)
            next_q_values.append(next_q_i.max(dim=1)[0].unsqueeze(1))
        next_q_values = torch.cat(next_q_values, dim=1)
        target_q_tot = self.target_mixing_network(next_q_values, next_states)

        target = (rewards + self.avg_reward) * target_q_tot

        # Compute loss
        loss = ((q_tot - target.detach()) ** 2).mean()

        # Update networks
        for opt in self.q_optimizers:
            opt.zero_grad()
        self.mixing_optimizer.zero_grad()
        loss.backward()
        for opt in self.q_optimizers:
            opt.step()
        self.mixing_optimizer.step()

    def update_target_networks(self):
        for q_net, target_q_net in zip(self.q_networks, self.target_q_networks):
            target_q_net.load_state_dict(q_net.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())


# Example usage
if __name__ == "__main__":
    env = Env()

    n_agents = len(env.agents)
    obs_dim = 2  # Example observation dimension
    state_dim = 6 + len(env.agents)  # Example state dimension
    action_dim = len(env.aulas)  # Example action dimension
    qmix = QMIX(n_agents, obs_dim, state_dim, action_dim)
    n_episodes = 1000
    # Simulate an episode
    for episode in range(n_episodes):
        iter_episodes = iter(range(n_episodes))
        episode = next(iter_episodes)
        state = env.get_state(env.data)
        # state = np.random.rand(state_dim)
        observations = env.get_obs()
        # observations = [np.random.rand(obs_dim) for _ in range(n_agents)]
        epsilon = max(0.1, 1.0 - episode / 800)

        actions = qmix.act(observations, epsilon)

        reward = env.get_reward()  # Simulated reward
        # reward = random.random()  # Simulated reward
        next_observations = env.get_next_obs(actions)
        # next_observations = [np.random.rand(obs_dim) for _ in range(n_agents)]
        next_state = env.get_state(env.data_prime)
        # next_state = np.random.rand(state_dim)
        # done = random.random() < 0.1
        qmix.store(state, observations, actions, reward, next_state, next_observations)

        qmix.update_avg_reward(reward)
        # print(qmix.replay_buffer)
        # print(observations, actions, reward, done)
        qmix.train(batch_size=32)
        #
        if episode % 100 == 0:
            qmix.update_target_networks()
            print(f"Episode {episode}")

        env.update_data()
        ww = env.data

        qmix.q_networks[0](torch.from_numpy(np.array([])))