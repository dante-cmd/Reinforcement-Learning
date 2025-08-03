import pandas as pd
import numpy as np
import json
import torch
from uuid import uuid4
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


class Agent:
    def __init__(self):
        pass

class Env:
    def __init__(self):
        # self.data = pd.read_excel('data.xlsx', dtype='string')
        # Course and quantity
        self.course = 'B01RP'
        self.quantity = 19
        # States
        self.states= np.array(['101', '102', '103', '104', '105', '106', '107', '108', '109', '110'])
        # capacities
        self.capacities = np.array([15, 15, 20, 25, 25, 20, 20, 14, 10, 19])
        # Actions
        self.actions = np.array(['up', 'down', 'keep'])
        # rewards
        self.diff = self.capacities - self.quantity
        self.rewards = np.where(self.diff < 0,
                                self.diff, np.where((self.diff>=0) & self.diff<=2, 2, 0))
        # discount factor
        self.gamma = 0.9
        # Exploration factor
        self.epsilon = 0.1
        # init
        self.state = np.random.choice(self.states)
        self.id_state = np.where(self.states == self.state)[0][0]
        self.target = np.where(self.rewards == np.max(self.rewards))[0]
        self.q_values = np.zeros((len(self.states), len(self.actions)))

    def reset(self):
        self.state = np.random.choice(self.states)
        self.id_state = np.where(self.states == self.state)[0][0]

    def step(self, id_action):
        x = int(self.id_state)
        factor = 0.7
        if id_action == 0:
            x += 1  # 'up'
        elif id_action == 1:
            x -= 1  # down
        elif id_action == 2:
            x += 0  # down
            factor = 1
        # The agent keeps in the same state if
        # the action inplies over the max or min values of the enviroment
        next_id_state = max(0, min(x, len(self.states)-1))

        # next_reward = 1 if next_state == self.goal else -0.01
        # reward = 1 if self.state == self.goal else -0.01
        # reward = self.rewards[self.id_state]
        reward = self.rewards[next_id_state]

        done = next_id_state in self.target
        return next_id_state, factor*reward, done

    def choose_action(self):
        if self.epsilon<np.random.random():
            action = np.random.choice(self.actions)
            id_action =np.where(self.actions == action)[0][0]
        else:
            id_action = np.argmax(self.q_values[self.id_state])

        return id_action

class TreeEnv(Env):
    def __init__(self):
        super().__init__()
        self.id_state = np.where(self.states == self.state)[0]
        self.tree_update_freq = 100
        self.experience_buffer = defaultdict(list)
        # self.trees = [DecisionTreeRegressor(random_state=1234, min_samples_split=5) for action in self.actions]
        self.trees = [LinearRegression() for action in self.actions]
        self.alpha = 0.9
        self.epsilon = 0.1
        self.step_count = 0
        self.quantity = 25

    def initialize_trees(self):
        # self = TreeEnv()
        sample_states = np.random.random((10, self.id_state.shape[0]))
        sample_actions = np.random.random((10, self.actions.shape[0]))

        for idx in range(len(self.actions)):
            tree = self.trees[idx]
            tree.fit(sample_states, sample_actions[:, idx])
            self.trees[idx] = tree

    def predict_q_value(self, id_state:np.array, id_action:int):
        return self.trees[id_action].predict(id_state.reshape(-1, 1))[0]

    def get_id_action(self, action) -> int:
        id_action = int(np.where(self.actions == action)[0][0])
        return id_action

    def choose_action(self) -> int:
        if self.epsilon<np.random.random():
            action = np.random.choice(self.actions)
            id_action = self.get_id_action(action)
        else:
            q_values= [self.predict_q_value(self.id_state.reshape(-1, 1),
                                            self.get_id_action(action)) for action in self.actions]

            id_action = np.argmax(q_values)

        return id_action

    def store_experience(self, id_state:np.array, id_action:int, reward:int,
                         id_next_state:np.array, done:int):
        """Store experience in buffer"""
        self.experience_buffer[id_action].append((id_state, reward, id_next_state, done))
        self.step_count += 1


    def update_trees(self):
        """Update all action trees using experience buffer"""
        for action in self.actions:
            id_action = self.get_id_action(action)
            if not self.experience_buffer[id_action]:
                continue

            # Prepare training data for this action's tree
            states, rewards, next_states, dones = zip(*self.experience_buffer[id_action])
            rewards = np.array(rewards)
            states = np.vstack(states)
            next_states = np.vstack(next_states)

            # Compute target Q-values using Bellman equation
            current_q = np.array([self.predict_q_value(s.reshape(-1, 1), id_action) for s in states])
            next_q = np.zeros(states.shape[0])

            for i, (ns, done) in enumerate(zip(next_states, dones)):
                if done:
                    next_q[i] = 0
                else:
                    next_q[i] = max([self.predict_q_value(ns, self.get_id_action(na)) for na in self.actions])

            # targets = rewards + self.gamma * next_q
            # print(current_q, rewards, next_q)
            # print(current_q.shape, rewards.shape, next_q.shape)
            targets = ((1 - self.alpha) * current_q +
                       self.alpha * (rewards + self.gamma * next_q))

            # Fit the tree to these targets
            self.trees[id_action].fit(states, targets)

        # Clear buffer after update
        self.experience_buffer = defaultdict(list)

    def get_id_state(self, state):
        return np.where(self.states == state)[0]

    def reset(self):
        self.state = np.random.choice(self.states)
        self.id_state = self.get_id_state(self.state)
        # x = np.array([1,6, 1, 1, 8, 8])

    def step(self, id_action:int):
        x = int(self.id_state[0])
        if id_action == 0:
            x += 1  # 'up'
        elif id_action == 1:
            x -= 1  # down
        elif id_action == 2:
            x += 0  # down
        # The agent keeps in the same state if
        # the action implies over the max or min values of the environment
        next_id_state = max(0, min(x, len(self.states)-1))

        # next_reward = 1 if next_state == self.goal else -0.01
        # reward = 1 if self.state == self.goal else -0.01
        # reward = self.rewards[self.id_state]
        reward = int(self.rewards[next_id_state])

        done = next_id_state in self.target
        return np.array([next_id_state]), reward, done


def q_learning():
    n_iter = 10000
    env = Env()
    while True:
        while True:
            env.reset()
            id_action = env.choose_action()
            next_id_state, reward, done = env.step(id_action)
            env.q_values[env.id_state,id_action] += env.gamma*(
                    reward + np.max(
                env.q_values[next_id_state, :]) - env.q_values[env.id_state, id_action])
            # print(env.id_state, id_action, env.q_values[env.id_state, id_action], next_id_state, reward, done)

            env.id_state = next_id_state

            if done:
                # print('done')
                break

        if n_iter <= 1:
            break
        n_iter-=1
    return  env

def get_tree_approx():

    """Train the agent in the given environment"""
    env = TreeEnv()
    env.initialize_trees()
    rewards = []

    for episode in range(1000):
        env.reset()
        total_reward = 0
        done = False

        while not done:
            id_action = env.choose_action()
            id_next_state, reward, done = env.step(id_action)

            env.store_experience(env.id_state, id_action, reward, id_next_state, done)
            # env.experience_buffer
            # Periodically update trees
            if env.step_count % env.tree_update_freq == 0:

                env.update_trees()

            env.id_state = id_next_state.copy()
            total_reward += reward

        rewards.append(total_reward)

    from collections import namedtuple

    Qvalues= namedtuple('QValues', ['state', 'action', 'q_value'])

    qw = []
    for state in env.states:
        for action in env.actions:
            q_value = env.predict_q_value(
                env.get_id_state(state), env.get_id_action(action))
            qw.append(Qvalues(state, action, q_value))

    df = pd.DataFrame(qw)
    er = df.pivot_table(
        index=['state'],
        columns=['action'],
        values='q_value',
        aggfunc='sum'
    )

    return rewards


def deep_q_learning():
    pass


# Q-learning agent with decision tree approximation
class QLearningTreeAgent:
    def __init__(self, state_dim, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1,
                 max_depth=5):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=5, random_state=42)
        self.replay_buffer = deque(maxlen=10000)
        self.X_train = []  # State-action pairs
        self.y_train = []  # Q-value targets

    def act(self, state:np.ndarray):
        assert state.shape == (1,)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self._predict_q_values(state)
        return np.argmax(q_values)

    def _predict_q_values(self, state:np.ndarray) -> np.ndarray:
        # Predict Q-values for all actions
        assert state.shape == (1,)
        q_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            state_action = np.append(state, action).reshape(1, -1)
            q_values[action] = self.q_tree.predict(state_action)[0] if len(self.X_train) > 0 else 0.0
        return q_values

    def store(self, state:np.ndarray, action:int, reward:int, next_state:np.ndarray, done:bool):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size:int=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)

        X_batch = []
        y_batch = []
        for state, action, reward, next_state, done in batch:
            # Compute target Q-value
            next_q_values = self._predict_q_values(next_state)
            target = reward if done else reward + self.gamma * np.max(next_q_values)
            current_q = self._predict_q_values(state)[action]
            updated_q = current_q + self.alpha * (target - current_q)

            # Add to training data
            state_action = np.append(state, action)
            X_batch.append(state_action)
            y_batch.append(updated_q)

        # Update training data
        self.X_train.extend(X_batch)
        self.y_train.extend(y_batch)

        # Retrain decision tree
        if len(self.X_train) > 0:
            self.q_tree.fit(np.array(self.X_train), np.array(self.y_train))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class Envs:
    def __init__(self):
        self.course = 'B01RP'
        self.quantity = 19
        # States
        self.states = np.array(['101', '102', '103', '104', '105',
                                '106', '107', '108', '109', '110'])
        # capacities
        self.capacities = np.array([15, 15, 20, 25, 25,
                                    20, 20, 14, 10, 19])
        # Actions
        self.actions = np.array(['up', 'down', 'keep'])
        # rewards
        self.diff = self.capacities - self.quantity
        self.rewards = np.where(self.diff < 0,
                                self.diff,
                                np.where((self.diff >= 0) & self.diff <= 2, 2, 0))

        self.target = np.where(
            self.rewards == np.max(self.rewards))[0]

    def step(self, state:np.ndarray, action:int):
        assert state.shape == (1,)

        x = int(state[0])
        if action == 0:
            x += 1  # 'up'
        elif action == 1:
            x -= 1  # down
        elif action == 2:
            x += 0  # down
        # The agent keeps in the same state if
        # the action implies over the max or min values of the environment
        next_state = max(0, min(x, len(self.states)-1))

        # next_reward = 1 if next_state == self.goal else -0.01
        # reward = 1 if self.state == self.goal else -0.01
        # reward = self.rewards[self.id_state]
        reward = int(self.rewards[next_state])

        done = next_state in self.target
        return np.array([next_state]), reward, done

    def reset(self):
        return np.array([np.random.choice(range(0,len(self.states)))])

# Training and evaluation
def main():
    env = Envs()
    # gym.make('CartPole-v1')
    state_dim = env.states.shape[0]
    #env.observation_space.shape[0]  # 4 for CartPole
    n_actions = env.actions.shape[0]
    #env.action_space.n  # 2 for CartPole
    agent = QLearningTreeAgent(state_dim=state_dim, n_actions=n_actions)

    n_episodes = 500
    max_steps = 500
    rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done  = env.step(state, action)
            # done = terminated or truncated
            agent.store(state, action, reward, next_state, done)

            agent.train(batch_size=32)
            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(rewards[-50:]) if rewards else 0
            print(f"Episode {episode}, Avg Reward (last 50): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    #  env.close()

    # Plot average rewards
    ww  = []
    for state in range(state_dim):
        ww.append(agent._predict_q_values(np.array([state])))

    kk = pd.DataFrame(index=env.states, columns=env.actions,data=ww)



    import matplotlib.pyplot as plt
    plt.plot(np.convolve(rewards, np.ones(50) / 50, mode='valid'))
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (50-episode moving avg)')
    plt.title('Q-Learning with Decision Tree on CartPole-v1')
    plt.show()


# Example usage
if __name__ == "__main__":
    env_q_learning = q_learning()
    data_q_learning = pd.DataFrame(env_q_learning.q_values,
                                   columns=env_q_learning.actions,
                                   index=env_q_learning.states)
    data_q_learning['AFORO'] = env_q_learning.capacities
    data_q_learning['quantity'] = env_q_learning.quantity
    data_q_learning.to_clipboard(index=False)