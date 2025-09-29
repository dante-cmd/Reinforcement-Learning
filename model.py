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
        self.dataset = dataset
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
        # self.items_bimestral = self.sample(self.items_bimestral)
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

    def get_availability(self, item):
        # self=env_01
        # item=self.items[1].copy()
        aulas = self.dim_aulas['AULA']
        aforos = self.dim_aulas['AFORO']
        roomlog = self.roomlog.copy()
        dias_expected = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas_expected = self.dim_horario[item['HORARIO']]

        aulas_availability = []
        for aula, aforo in zip(aulas, aforos):
            availability = []
            for periodo_franja in self.dim_periodo_franja.keys():
                franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
                dias = self.dim_periodo_franja[periodo_franja]['DIAS']
                for franja in franjas:
                    availability_dias = []
                    for dia in dias:
                        if (dia in dias_expected) and (franja in franjas_expected):
                            availability_dias.append(True if roomlog[aula][dia][franja] == 0 else False)
                    if franja in franjas_expected:
                        availability.append(item['ALUMN'] / aforo if all(availability_dias) else 0)
                    else:
                        availability.append(0)
            aulas_availability.append(availability)
        # df = pd.DataFrame(aulas_availability)
        return np.array(aulas_availability)

    def get_utilization_expected(self, item:dict):

        dias_expected = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas_expected = self.dim_horario[item['HORARIO']]

        utilization_expected = []
        for periodo_franja in self.dim_periodo_franja.keys():
            franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
            dias = self.dim_periodo_franja[periodo_franja]['DIAS']
            for franja in franjas:
                utilization_franjas = []
                for dia in dias:
                    if (franja in franjas_expected) and (dia in dias_expected):
                        utilization_franjas.append(1)
                    else:
                        utilization_franjas.append(0)
                utilization_expected.append(sum(utilization_franjas)/len(utilization_franjas))
        return np.array(utilization_expected)

    def get_capacity(self, item: dict):
        aforos = self.dim_aulas['AFORO']
        capacity = []
        for aforo in aforos:
            capacity.append(item['ALUMN'] / aforo)
        return np.array(capacity)

    def get_match_capacity(self, item: dict):
        aforos = self.dim_aulas['AFORO']
        capacity = []
        for aforo in aforos:
            capacity.append(1 if item['ALUMN'] == aforo else 0)
        return np.array(capacity)

    def get_state(self):
        if self.idx_item >= len(self.items):
            return None

        item = self.items[self.idx_item].copy()
        # capacity = self.get_capacity(item)
        availability = self.get_availability(item)
        match_capacity = self.get_match_capacity(item)
        # utilization_expected = self.get_utilization_expected(item)

        # obs = np.concatenate([capacity, match_capacity,
        #                       utilization_expected])

        utilization = self.get_utilization()
        utilization_01 = np.expand_dims(utilization, 0)
        availability_01 = np.expand_dims(availability, 0)

        obs = np.concatenate([utilization_01, availability_01], axis=0)
        return obs, match_capacity

    def get_available_actions(self):

        item = self.items[self.idx_item].copy()
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
        aula = self.dim_aulas['AULA'][action]
        aforo = self.dim_aulas['AFORO'][action]
        roomlog = self.roomlog.copy()

        # conflict = 0
        # dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        # franjas = self.dim_horario[item['HORARIO']]
        # # for periodo_franja in self.dim_periodo_franja.keys():
        # for dia in dias:
        #     for franja in franjas:
        #         conflict = + roomlog[aula][dia][franja]

        # reward = 0
        # response = ''
        if action == self.n_aulas:
            # conflict > 0:
            reward = -10
            response = '[Conflict]'
        else:
            if (aforo - item['ALUMN']) < 0:
                reward = aforo - item['ALUMN'] - 2
                response = '[Aforo-Alum<0]'
            # else:
            elif ((aforo - item['ALUMN']) >= 0) and ((aforo - item['ALUMN']) <= 2):
                reward = 1 + (item['ALUMN']/aforo)
                # response = '[Aforo-Alumn>=0]'
                response = '[Aforo=Alumn]|[Aforo-Alumn<=2]'
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


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, out_channels=20):
        super(ConvNet, self).__init__()
        self.stack = nn.Sequential(
            # initial input shape : (64, 1, 12, 19)
            nn.Conv2d(
                in_channels=2, out_channels=12,
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
    def __init__(self, input_size, out_features=64):
        super(FeedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
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
    def __init__(self, env, assignation=False):
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

        if assignation:
            self.model.load_state_dict(
                torch.load(f"model_{self.env.sede}.pt", weights_only=True))

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
        # state = self.env.get_state()

        mask_actions = self.env.get_available_actions()
        if sum(mask_actions) == 0:
            return self.env.n_aulas
        else:
            if random.random() <= self.epsilon:
                return random.randrange(self.env.n_aulas)

            states_conv = torch.FloatTensor(state[0]).unsqueeze(0) # .unsqueeze(0)
            states_feed = torch.FloatTensor(state[1]).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(states_conv,
                                      states_feed)
                q_values = q_values.squeeze(0)[:-1]
            action = np.argmax([-float('inf') if mask == 0 else q_values[idx].item()
                                for idx, mask in enumerate(mask_actions)])
            # torch.argmax(q_values).item()
            return action

    def remember(self, state, action, reward, next_state, done):
        if (state is not None) and (next_state is not None):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # self = dqn_agent
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)

        states_conv = torch.FloatTensor(np.array([t[0][0] for t in minibatch]))
        states_feed = torch.FloatTensor(np.array([t[0][1] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states_conv = torch.FloatTensor(np.array([t[3][0] for t in minibatch]))
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
                # print(state[0])
                # print(state[1][:12])
                # print(state[1][12:24])
                # print(state[1][24:])
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

        torch.save(
            self.model.state_dict(),
            f"model_{self.env.sede}.pt"
        )

    def get_assignments(self, episodes):
        # self = dqn_agent
        # episodes = 100
        # e=0

        self.epsilon = 0
        best_reward = -float('inf')
        best_responses = []
        for e in range(episodes):
            state = self.env.reset()
            self.env.items = self.env.dataset.items[self.env.sede].copy()
            total_reward = 0
            done = False
            responses = []

            while not done:
                action = self.act(state)
                reward, next_state, done, info = self.env.step(action)

                # if next_state is not None:
                self.remember(state, action, reward, next_state, done)
                responses.append(info)

                state = next_state
                total_reward += reward
                self.steps += 1

                loss = self.replay()

            if total_reward > best_reward:
                best_reward = total_reward
                best_responses = responses.copy()
                print(f"⭐ Episode: {e}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")
            print(f"🧪 Episode: {e}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

                # if self.steps % self.update_target_every == 0:
                #     self.target_model.load_state_dict(self.model.state_dict())

            # Decay epsilon
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay

            # if e % 10 == 0:
                # unique_response, count_response = np.unique(responses, return_counts=True)
            #     print(f"🧪 Episode: {e}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

        return best_responses

    def get_assignments_v0(self):
        # self = dqn_agent
        """Get optimal assignments using current policy"""
        assignments = []
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            # item = self.env.items[self.env.idx_item].copy()
            mask_actions = self.env.get_available_actions()
            if sum(mask_actions) == 0:
                action = self.env.n_aulas
            else:
                with torch.no_grad():
                    states_conv = torch.FloatTensor(state[0]).unsqueeze(0)
                    states_feed = torch.FloatTensor(state[1]).unsqueeze(0)
                    q_values = self.model(states_conv, states_feed)
                    q_values = q_values.squeeze(0)[:-1]
                    action = np.argmax([-float('inf') if mask == 0 else q_values[idx].item()
                                        for idx, mask in enumerate(mask_actions)])

            reward, next_state, done, info = self.env.step(action)
            assignments.append(info)
            total_reward += reward
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

        print(f"🧪Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.2f}")

        return assignments


if __name__ == '__main__':
    PATH = Path("project")
    SEDE = 'Ica'
    dataset_01 = DataSet(PATH)

    env_01 = Environment(dataset_01, SEDE)
    dqn_agent = DQNAgent(env_01)
    dqn_agent.train(100)

    dqn_agent = DQNAgent(env_01, assignation=True)
    ll = dqn_agent.get_assignments(100)
    kk = pd.DataFrame(ll)
    kk = env_01.items[20]
    po = env_01.get_utilization_expected(kk)
    dqn_agent.train(5000)
    # Save the model's state_dict
    # torch.save(dqn_agent.model.state_dict(), PATH)
    # ll = dqn_agent.get_assignments(100)

    # dqn_agent.train(5000)
    # ww = dqn_agent.get_assignments()
    # hh = pd.DataFrame(ww)
    # Define the path to save the model
    # PATH_MODEL = f"model_{SEDE}.pt"
    # Save the model's state_dict
    # torch.save(dqn_agent.model.state_dict(), PATH)
    # ll_00 = dqn_agent.train(1000)
    # kk = pd.DataFrame(ll)
    # dqn_agent.model.load_state_dict(torch.load(PATH_MODEL))

    # ll = dqn_agent.get_assignments()




