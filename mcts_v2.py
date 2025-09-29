# tictactoe.py
import copy
# model.py
import torch.nn as nn
import torch.nn.functional as F
# mcts.py
import math
import torch
import numpy as np
from collections import defaultdict
# train.py
import torch
import torch.optim as optim


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # X starts

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_valid_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def step(self, action: tuple):
        i, j = action
        if self.board[i, j] != 0:
            raise ValueError("Invalid move")
        self.board[i, j] = self.current_player
        done, winner = self.check_winner()
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1
            elif winner == -self.current_player:
                reward = -1
            else:  # draw
                reward = 0
        self.current_player *= -1
        return self.get_state(), reward, done

    def check_winner(self):
        # Check rows, cols, diagonals
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return True, self.board[i, 0]
            if abs(self.board[:, i].sum()) == 3:
                return True, self.board[0, i]
        if abs(self.board.diagonal().sum()) == 3:
            return True, self.board[1, 1]
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return True, self.board[1, 1]
        if len(self.get_valid_moves()) == 0:
            return True, 0  # draw (no winner)
        return False, 0  # (no winner)

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join([symbols[x] for x in row]))


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 9, 64)
        self.policy_head = nn.Linear(64, 9)  # 9 actions
        self.value_head = nn.Linear(64, 1)  # scalar value [-1, 1]

    def forward(self, x):
        # x: (batch, 3, 3) -> add channel dim
        x = x.unsqueeze(1)  # (batch, 1, 3, 3)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # from neural net policy

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_fully_expanded(self, env):
        return len(self.children) == len(env.get_valid_moves())

    def best_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            uct_score = child.value + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_child = child

        return best_action, best_child


def is_terminal(env):
    env_copy = copy.deepcopy(env)
    # env_copy.board = state
    done, _ = env_copy.check_winner()
    return done


def mcts_search(root_state, model, env, num_simulations=800, c_puct=1.0):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TicTacToeNet().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # num_simulations = 5000
    # c_puct = 1.0
    # env = TicTacToe()
    # root_state = env.reset()
    root_player = env.current_player
    root_board = env.board.copy()

    root = MCTSNode(root_state)

    for _ in range(num_simulations):
        node = root  # It is not a copy, all steps applied to node are saved in root as well
        search_path = [node]

        # 1. Selection
        while node.is_fully_expanded(env) and not is_terminal(env):
            action, node = node.best_child(c_puct)
            env.step(action)
            search_path.append(node)

        # 2. Expansion
        if not is_terminal(env):
            state_tensor = torch.FloatTensor(node.state).unsqueeze(0)  # add batch dim
            with torch.no_grad():
                policy, value = model(state_tensor)
                policy = policy.squeeze(0).numpy()
                value = value.item()

            valid_moves = env.get_valid_moves()
            for idx, move in enumerate(valid_moves):
                if move not in node.children:
                    prior = policy[move[0] * 3 + move[1]]  # flatten index
                    child_state = node.state.copy()
                    child_env = copy.deepcopy(env)

                    child_env.board = child_state  # It is not necessary a copy
                    child_env.step(move)  # This step changes child_state through child_env.board
                    node.children[move] = MCTSNode(child_state, parent=node, prior=prior)

            # Pick one child to simulate
            # if node.children:
            #     action, node = node.best_child(c_puct)
            #     search_path.append(node)

        # 3. Simulation (use NN value instead of random rollout)
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
        with torch.no_grad():
            _, value = model(state_tensor)
            value = value.item()

        # 4. Backpropagation
        for n in reversed(search_path):
            n.visit_count += 1
            # if n.parent:
            n.value_sum += value
            value = -value  # flip value for opponent

        env.board = root_board.copy()
        env.current_player=root_player

    # Return visit counts as policy
    visit_counts = np.zeros(9)
    for action, child in root.children.items():
        idx = action[0] * 3 + action[1]
        visit_counts[idx] = child.visit_count

    # Avoid division by zero
    total = visit_counts.sum()
    if total == 0:
        return np.ones(9) / 9
    return visit_counts / total


# from tictactoe import TicTacToe
# from model import TicTacToeNet
# from mcts import mcts_search

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TicTacToeNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def self_play_episode(model, num_simulations=100):
    env = TicTacToe()
    state = env.reset()
    states, mcts_policies, players = [], [], []

    while True:
        # Use MCTS to get improved policy
        policy = mcts_search(state, model, env, num_simulations=num_simulations)

        states.append(state.copy())
        mcts_policies.append(policy)
        players.append(env.current_player)

        # Sample action from MCTS policy
        action_idx = np.random.choice(9, p=policy)
        action = (action_idx // 3, action_idx % 3)
        state, reward, done = env.step(action)

        if done:
            # Assign final rewards to all states
            returns = []
            for player in players:
                if reward == 0:
                    returns.append(0)
                else:
                    returns.append(1 if player == reward else -1)
            return states, mcts_policies, returns


def train_step(states, target_policies, target_values, model, optimizer):
    states = torch.FloatTensor(states).to(device)
    target_policies = torch.FloatTensor(target_policies).to(device)
    target_values = torch.FloatTensor(target_values).unsqueeze(1).to(device)

    optimizer.zero_grad()
    pred_policy, pred_value = model(states)
    policy_loss = -(target_policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
    value_loss = torch.nn.MSELoss()(pred_value, target_values)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    return loss.item()


def train():
    # Training loop
    for episode in range(1000):
        states, policies, values = self_play_episode(model, num_simulations=50)
        loss = train_step(states, policies, values, model, optimizer)
        if episode % 50 == 0:
            print(f"Episode {episode}, Loss: {loss:.4f}")


def play():
    # play.py
    env = TicTacToe()
    state = env.reset()
    model.eval()

    while True:
        env.render()
        if env.current_player == 1:
            # Human move
            move = input("Enter move (row,col): ")
            i, j = map(int, move.split(','))
            action = (i, j)
        else:
            # AI move via MCTS
            policy = mcts_search(state, model, env, num_simulations=200)
            action_idx = np.argmax(policy)
            action = (action_idx // 3, action_idx % 3)
            print(f"AI plays: {action}")

        state, reward, done = env.step(action)
        if done:
            env.render()
            if reward == 1:
                print("X wins!")
            elif reward == -1:
                print("O wins!")
            else:
                print("Draw!")
            break


if __name__ == '__main__':
    train()
    play()
