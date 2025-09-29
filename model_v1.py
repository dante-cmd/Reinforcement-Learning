import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback


class DataSet:
    def __init__(self, path):
        self.path = path
        self.dim_aulas = None
        self.dim_periodo_franja = None
        self.dim_frecuencia = None
        self.dim_horario = None
        self.items = None
        self.items_bimestral = None
        self.load_all()

    def read_json(self, name: str):
        with open(self.path / name, "r") as f:
            return json.load(f)

    def dim_aulas_loader(self):
        self.dim_aulas = self.read_json("dim_aulas.json")

    def dim_periodo_franja_loader(self):
        self.dim_periodo_franja = self.read_json("dim_periodo_franja.json")

    def dim_frecuencia_loader(self):
        self.dim_frecuencia = self.read_json("dim_frecuencia.json")

    def dim_horario_loader(self):
        self.dim_horario = self.read_json("dim_horario.json")

    def items_loader(self):
        self.items = self.read_json("items.json")

    def items_bimestral_loader(self):
        self.items_bimestral = self.read_json("items_bimestral.json")

    def load_all(self):
        self.dim_horario_loader()
        self.dim_aulas_loader()
        self.dim_frecuencia_loader()
        self.dim_periodo_franja_loader()
        self.items_loader()
        self.items_bimestral_loader()


class Environment:
    def __init__(self, dataset, sede: str):
        self.sede = sede
        self.dim_aulas = dataset.dim_aulas[sede].copy()
        self.dim_periodo_franja = dataset.dim_periodo_franja.copy()
        self.dim_frecuencia = dataset.dim_frecuencia.copy()
        self.dim_horario = dataset.dim_horario.copy()
        self.items = dataset.items[sede].copy()
        self.items_bimestral = dataset.items_bimestral[sede].copy()
        self.roomlog = self.get_roomlog()
        self.state = None
        self.idx_item = 0
        self.n_franjas = self.get_n_franjas()
        self.n_aulas = len(self.roomlog.keys())

    def get_n_franjas(self):
        n_franjas = 0
        for periodo_franja in self.dim_periodo_franja:
            n_franjas += len(self.dim_periodo_franja[periodo_franja]['FRANJAS'])
        return n_franjas

    def get_roomlog(self):
        # self = env_01
        aulas = self.dim_aulas['AULA']
        room_log = {}
        for aula in aulas:
            room_log[aula] = {}
            # data = {}
            for periodo_franja in self.dim_periodo_franja.keys():
                franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
                dias = self.dim_periodo_franja[periodo_franja]['DIAS']
                for dia in dias:
                    room_log[aula][dia] = {}
                    # data[dia] = {}
                    for franja in franjas:
                        room_log[aula][dia][franja] = 0
                        # data[dia][franja] = 0
            # room_log[aula] = data

        for item in self.items_bimestral:
            # conflict = 0
            dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
            franjas = self.dim_horario[item['HORARIO']]
            for dia in dias:
                for franja in franjas:
                    room_log[item['AULA']][dia][franja] = 1
        return room_log

    @staticmethod
    def sample(items):
        return np.random.choice(
            items,
            replace=False,
            size=len(items)).tolist()

    def reset(self):
        self.idx_item = 0
        self.items_bimestral = self.sample(self.items_bimestral)
        self.items = self.sample(self.items)
        self.roomlog = self.get_roomlog()
        self.state = self.get_state()
        return self.state

    def get_utilization(self):
        aulas = self.dim_aulas['AULA']
        roomlog = self.roomlog.copy()

        aulas_utilization = []
        for aula in aulas:
            utilization = []
            for periodo_franja in self.dim_periodo_franja.keys():
                franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
                dias = self.dim_periodo_franja[periodo_franja]['DIAS']
                for franja in franjas:
                    utilization.append(
                        sum([roomlog[aula][dia][franja] for dia in dias]) / len(dias)
                    )
            aulas_utilization.append(utilization)
        return np.array(aulas_utilization)

    def get_capacity(self, alumn: float):
        aforos = self.dim_aulas['AFORO']
        capacity = []
        for aforo in aforos:
            capacity.append(alumn / aforo)
        return np.array(capacity)

    def get_state(self):
        if self.idx_item >= len(self.items):
            return None

        item = self.items[self.idx_item]
        capacity = self.get_capacity(item['ALUMN'])
        utilization = self.get_utilization()
        return utilization, capacity

    def get_available_actions(self):

        item = self.items[self.idx_item]
        aulas = self.dim_aulas['AULA']
        roomlog = self.roomlog.copy()
        dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = self.dim_horario[item['HORARIO']]

        available = []
        for aula in aulas:
            conflict = []
            for dia in dias:
                for franja in franjas:
                    conflict.append(True if roomlog[aula][dia][franja] == 1 else False)
            available.append(0 if any(conflict) else 1)
        return available

    def step(self, action: int):
        if self.idx_item >= len(self.items):
            return None, None, True

        item = self.items[self.idx_item].copy()

        conflict = 0
        dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = self.dim_horario[item['HORARIO']]
        aula = self.dim_aulas['AULA'][action]
        aforo = self.dim_aulas['AFORO'][action]
        roomlog = self.roomlog.copy()
        # for periodo_franja in self.dim_periodo_franja.keys():
        for dia in dias:
            for franja in franjas:
                conflict = + roomlog[aula][dia][franja]

        # reward = 0
        # response = ''
        # if action == self.n_aulas:
        if conflict > 0:
            reward = -10
            response = '[Conflict]'
            aula = np.NAN
            aforo = np.NAN
        else:
            if (aforo - item['ALUMN']) < 0:
                reward = (aforo - item['ALUMN']) - 2
                response = '[Aforo-Alum<0]'
            elif ((aforo - item['ALUMN']) >= 0) and ((aforo - item['ALUMN']) <= 2):
                reward = 2
                response = '[Aforo=Alumn]|[Aforo-Alumn<2]'
            else:
                # (aforo - item['ALUMN']) > 2:
                reward = 0
                response = '[Aforo-Alumn>2]'
            dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
            franjas = self.dim_horario[item['HORARIO']]
            # for periodo_franja in self.dim_periodo_franja.keys():
            for dia in dias:
                for franja in franjas:
                    roomlog[aula][dia][franja] = 1
            self.roomlog = roomlog.copy()
        done = False
        self.idx_item += 1
        next_state = self.get_state()
        if next_state is None:
            done = True

        info = item.copy()
        info['AULA'] = aula
        info['AFORO'] = aforo
        info['RESPONSE'] = response

        return reward, next_state, done, info


class SchedulingEnv(gym.Env):
    def __init__(self, dataset, sede: str):
        # self = env
        super(SchedulingEnv, self).__init__()
        self.sede = sede
        self.dataset = dataset
        self.env_core = Environment(dataset, sede)

        # Define action and observation space
        self.n_aulas = self.env_core.n_aulas
        self.n_franjas = self.env_core.n_franjas

        # Action space: choose an aula (0 to n_aulas-1) or reject (n_aulas)
        # self.action_space = spaces.Discrete(self.n_aulas + 1)
        self.action_space = spaces.Discrete(self.n_aulas)

        # Observation space: tuple (utilization [n_aulas x n_franjas], capacity [n_aulas])
        self.observation_space = spaces.Dict({
            "utilization": spaces.Box(
                low=0, high=1, shape=(self.n_aulas, self.n_franjas), dtype=np.float32),
            "capacity": spaces.Box(
                low=0, high=100, shape=(self.n_aulas,), dtype=np.float32)
        })
        self.state = None
        self.reset()

    def reset(self, seed=None, options=None):
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

        self.env_core.idx_item = 0
        self.env_core.items_bimestral = self.env_core.sample(self.env_core.items_bimestral)
        self.env_core.items = self.env_core.sample(self.env_core.items)
        self.env_core.roomlog = self.env_core.get_roomlog()
        self.state = self.env_core.get_state()

        return self._convert_state(self.state), {}

    def step(self, action):
        reward, next_state, done, info = self.env_core.step(action)
        self.state = next_state

        # Convert response to reward shaping if needed
        # (you can later refine this)
        truncated = False  # No time limit here
        # info = {"response": response}

        obs = self._convert_state(next_state) if not done else None
        return obs, reward, done, truncated, info

    @staticmethod
    def _convert_state(state):
        if state is None:
            return None
        utilization, capacity = state
        utilization = np.array(utilization, dtype=np.float32)
        capacity = np.array(capacity, dtype=np.float32)
        return {
            "utilization": utilization,
            "capacity": capacity
        }

    def render(self, mode="human"):
        print(f"Current item index: {self.env_core.idx_item}")


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_output_dim=128, mlp_output_dim=64):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        # Image-like input: utilization [n_aulas, n_franjas]
        n_aulas, n_franjas = observation_space["utilization"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_aulas * n_franjas * 20, cnn_output_dim),
            nn.ReLU(),
        )

        # Vector input: capacity [n_aulas]
        self.mlp = nn.Sequential(
            nn.Linear(n_aulas, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, mlp_output_dim),
            nn.ReLU(),
        )

        # Compute combined features dim
        self._features_dim = cnn_output_dim + mlp_output_dim

    def forward(self, observations):
        utilization = observations["utilization"].unsqueeze(1)  # Add channel dim
        capacity = observations["capacity"]

        cnn_features = self.cnn(utilization)
        mlp_features = self.mlp(capacity)
        combined = torch.cat([cnn_features, mlp_features], dim=1)
        return combined


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, out_channels=20):
        super(ConvNet, self).__init__()
        self.stack = nn.Sequential(
            # initial input shape : (64, 1, 12, 19)
            nn.Conv2d(
                in_channels=1, out_channels=12,
                kernel_size=3, stride=1, padding=1),
            # output shape : (64, 12, 12, 19)
            # Apply normalization by channel (delta and beta are learning)
            # (x - mean)/std * delta + beta
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            # input shape : (64, 12, 12, 19)
            nn.Conv2d(
                in_channels=12, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1),
            # output shape : (64, 20, 12, 19)
            nn.ReLU(),
            # input shape : (64, 20, 12, 19)
            nn.BatchNorm2d(num_features=out_channels),
            # output shape : (64, 32, 12, 19)
            nn.ReLU(),
        )

        # self.fc = nn.Linear(in_features=75 * 75 * 32,
        #                     out_features=num_classes)
        self.dropout = nn.Dropout(0.3)

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

    def forward(self, input):
        output = self.stack(input)
        output = self.dropout(output)
        # Above output will be in matrix form, with shape (256,32,75,75)
        # output = output.view(-1, 32*75*75)
        # Resize the array. -1 a comodin to fit the dimention of the rows given
        # the dimentions of the columns, 32*75*75.
        # output = self.fc(output)

        return output


# 2. DQN Model (Improved)
class FeedNet(nn.Module):
    def __init__(self, n_aulas, out_features=64):
        super(FeedNet, self).__init__()
        self.fc1 = nn.Linear(n_aulas, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Combined network that concatenates both branches
class MultimodalNet(nn.Module):
    def __init__(self, cnn_branch, ff_branch, input_size, output_size):
        super(MultimodalNet, self).__init__()
        self.cnn_branch = cnn_branch
        self.ff_branch = ff_branch

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_size)
        )

    def forward(self, image_data, feature_data):
        # Process through CNN branch
        cnn_features = self.cnn_branch(image_data)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)  # Flatten

        # Process through feedforward branch
        ff_features = self.ff_branch(feature_data)

        # Concatenate features from both branches
        combined_features = torch.cat((cnn_features, ff_features), dim=1)

        # Final classification
        output = self.classifier(combined_features)
        return output


# 3. DQN Agent (Fixed)
class DQNAgent:
    def __init__(self, env):
        self.env = env
        # self.state_size = len(env.get_state())
        # self.action_size = env.n_aulas
        # len(env_01.roomlog.keys())
        out_channels = 20
        conv_net = ConvNet(out_channels=out_channels)

        out_feed = 64
        feed_net = FeedNet(env.n_aulas, out_features=out_feed)

        input_multimodal = out_feed + (env.n_franjas * env.n_aulas * out_channels)
        self.model = MultimodalNet(conv_net,
                                   feed_net,
                                   input_multimodal, env.n_aulas + 1)
        self.target_model = MultimodalNet(conv_net,
                                          feed_net,
                                          input_multimodal, env.n_aulas + 1)

        # self.model = DQN(self.state_size, self.action_size)
        # self.target_model = DQN(self.state_size, self.action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # 0.001
        self.epsilon_decay = 0.995  # 0.995
        self.update_target_every = 100
        self.steps = 0

    def act(self, state):
        # self = dqn_agent
        mask_actions = self.env.get_available_actions()
        if sum(mask_actions) == 0:
            return self.env.n_aulas
        else:
            if random.random() <= self.epsilon:
                return random.randrange(self.env.n_aulas)

            states_conv = torch.FloatTensor(state[0]).unsqueeze(0).unsqueeze(0)
            states_feed = torch.FloatTensor(state[1]).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(states_conv,
                                      states_feed)
                q_values = q_values.squeeze(0)[:-1]
            action = np.argmax([-float('inf') if mask == 0 else q_values[idx].item()
                                for idx, mask in enumerate(mask_actions)])
            return action  # torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        if (state is not None) and (next_state is not None):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        states_conv = torch.FloatTensor(np.array([np.expand_dims(t[0][0], 0) for t in minibatch]))
        states_feed = torch.FloatTensor(np.array([t[0][1] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states_conv = torch.FloatTensor(np.array([np.expand_dims(t[3][0], 0) for t in minibatch]))
        next_states_feed = torch.FloatTensor(np.array([t[3][1] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        current_q = self.model(states_conv, states_feed).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_model(next_states_conv, next_states_feed).max(1)[0]

        target = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(current_q.squeeze(), target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, episodes):
        # self = dqn_agent
        # episodes = 100
        # e=0

        for e in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            responses = []

            while not done:
                action = self.act(state)
                # self.env.items[self.env.idx_item]
                # self.env.items[1]
                reward, next_state, done, info = self.env.step(action)
                # import pandas as pd
                # ll_0 = pd.DataFrame(state[0])
                # ll_1 = pd.DataFrame(next_state[0])

                # if next_state is not None:
                self.remember(state, action, reward, next_state, done)
                responses.append(info['RESPONSE'])

                state = next_state
                total_reward += reward
                self.steps += 1

                loss = self.replay()

                if self.steps % self.update_target_every == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if e % 10 == 0:
                # unique_response, count_response = np.unique(responses, return_counts=True)
                print(f"🧪 Episode: {e}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

                # for idx, (xx, yy) in enumerate(zip(unique_response, count_response / count_response.sum())):
                #     if idx == 0:
                #         print(f'  {xx}:{yy:0.2%}', end=' | ')
                #     elif idx == (len(unique_response) - 1):
                #         print(f'{xx}:{yy:0.2%}')
                #     else:
                #         print(f'{xx}:{yy:0.2%}', end=' | ')

    def get_assignments(self):
        # self = dqn_agent
        """Get optimal assignments using current policy"""
        assignments = []
        state = self.env.reset()
        done = False

        while not done:
            item = self.env.items[self.env.idx_item].copy()
            mask_actions = self.env.get_available_actions()
            if sum(mask_actions) == 0:
                action = self.env.n_aulas
            else:
                with torch.no_grad():
                    states_conv = torch.FloatTensor(state[0]).unsqueeze(0).unsqueeze(0)
                    states_feed = torch.FloatTensor(state[1]).unsqueeze(0)
                    q_values = self.model(states_conv, states_feed)
                    q_values = q_values.squeeze(0)[:-1]
                    action = np.argmax([-float('inf') if mask == 0 else q_values[idx].item()
                                        for idx, mask in enumerate(mask_actions)])

            reward, next_state, done, info = self.env.step(action)
            assignments.append(info)
            # if info['valid']:
            #     course = info['course']
            #     room = info['room']
            #     room_data = self.env.temp_classrooms[room]
            #     assignments.append({
            #         'course': course['code'],
            #         'room': room,
            #         'capacity': room_data['capacity'],
            #         'students': course['students'],
            #         'utilization': course['students'] / room_data['capacity'],
            #         'schedule': course['schedule'],
            #         'frequency': course['frequency'],
            #         'days': self.env.frequency_mapping[course['frequency']],
            #         'slots': self.env.composite_slots[course['schedule']]
            #     })

            state = next_state

        return assignments


if __name__ == '__main__':
    PATH = Path("project")
    dataset_01 = DataSet(PATH)

    # Create wrapped gym environment
    env = SchedulingEnv(dataset_01, 'Ica')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=100)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="./checkpoints/",
                                             name_prefix="dqn_scheduling")

    # Custom policy with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128, mlp_output_dim=64),
        net_arch=[256, 256],  # Q-network top layers
    )

    # Create DQN agent
    model = DQN(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=2000,
        batch_size=64,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=100,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/"
    )

    # Train
    model.learn(total_timesteps=70000, callback=checkpoint_callback)

    # Save model
    model.save("dqn_scheduling_final")

    # To load and run later:
    model = DQN.load(
        "dqn_scheduling_final",
        env=env,
        custom_objects={"policy_kwargs": policy_kwargs},
    )

    # Run inference
    env = SchedulingEnv(dataset_01, 'Ica')
    obs, _ = env.reset()
    done = False
    assignments = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        assignments.append(info)
    df = pd.DataFrame(assignments)
    # df.to_clipboard()
        # print(f"Action: {action}, Reward: {reward}, Info: {info}")

# PATH = Path("project")
# SEDE = 'Ica'
# dataset_01 = DataSet(PATH)
# env_01 = Environment(dataset_01, SEDE)
# dqn_agent = DQNAgent(env_01)
# dqn_agent.train(5000)
# ww = dqn_agent.get_assignments()
# hh = pd.DataFrame(ww)
# # Define the path to save the model
# PATH = f"model_{SEDE}.pt"
# # Save the model's state_dict
# torch.save(dqn_agent.model.state_dict(), PATH)
# # ll = dqn_agent.get_assignments()
