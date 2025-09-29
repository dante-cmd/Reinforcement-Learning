import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for X, -1 for O

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
        self.current_player *= -1
        return self.get_state(), reward, done

    def check_winner(self):
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return True, self.board[i, 0]
            if abs(self.board[:, i].sum()) == 3:
                return True, self.board[0, i]
        if abs(self.board.diagonal().sum()) == 3:
            return True, self.board[0, 0]
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return True, self.board[0, 2]
        if len(self.get_valid_moves()) == 0:
            return True, 0
        return False, 0

    def render(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in self.board:
            print(' '.join([symbols[x] for x in row]))


class TicTacToeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.policy_head = nn.Linear(256, 9)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 3, 3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value


class MCTSNode:
    def __init__(self, state, parent=None, prior=1.0, player=1):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.player = player

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def is_fully_expanded(self, valid_moves):
        return len(self.children) == len(valid_moves)

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


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*samples)
        return np.array(states), np.array(policies), np.array(values)

    def __len__(self):
        return len(self.buffer)


def is_terminal(state, env):
    env_copy = copy.deepcopy(env)
    env_copy.board = state.copy()
    done, _ = env_copy.check_winner()
    return done


def add_dirichlet_noise(priors, valid_moves, epsilon=0.25, alpha=0.3):
    noise = np.random.dirichlet([alpha] * len(valid_moves))
    noisy_priors = (1 - epsilon) * priors + epsilon * noise
    return noisy_priors / noisy_priors.sum()


def mcts_search(root_state, model, env, num_simulations=800, c_puct=1.0, temperature=1.0):
    root = MCTSNode(root_state, player=env.current_player)
    root_board = root_state.copy()
    root_player = env.current_player

    # Add Dirichlet noise to root node priors
    state_tensor = torch.FloatTensor(root.state).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, _ = model(state_tensor)
        policy = policy.squeeze(0).cpu().numpy()
    valid_moves = env.get_valid_moves()
    valid_indices = [move[0] * 3 + move[1] for move in valid_moves]
    valid_priors = policy[valid_indices]
    if len(valid_moves) > 0:
        valid_priors = add_dirichlet_noise(valid_priors, valid_moves)
        for move, prior in zip(valid_moves, valid_priors):
            root.children[move] = MCTSNode(root.state, parent=root, prior=prior, player=-root.player)

    for _ in range(num_simulations):
        node = root
        search_path = [node]
        current_env = copy.deepcopy(env)
        current_env.board = node.state.copy()
        current_env.current_player = node.player

        # Selection
        while node.is_fully_expanded(current_env.get_valid_moves()) and not is_terminal(node.state, current_env):
            action, node = node.best_child(c_puct)
            current_env.step(action)
            search_path.append(node)

        # Expansion
        if not is_terminal(node.state, current_env):
            state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(device)
            with torch.no_grad():
                policy, value = model(state_tensor)
                policy = policy.squeeze(0).cpu().numpy()
                value = value.item()

            valid_moves = current_env.get_valid_moves()
            valid_indices = [move[0] * 3 + move[1] for move in valid_moves]
            valid_priors = policy[valid_indices]
            for move, prior in zip(valid_moves, valid_priors):
                if move not in node.children:
                    child_env = copy.deepcopy(current_env)
                    child_state, _, _ = child_env.step(move)
                    node.children[move] = MCTSNode(child_state, parent=node, prior=prior, player=-node.player)

            # Simulate one more step if possible
            if node.children:
                action = random.choice(list(node.children.keys()))
                node = node.children[action]
                current_env.step(action)
                search_path.append(node)

        # Simulation
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = model(state_tensor)
            value = value.item()

        # Backpropagation
        for n in reversed(search_path):
            n.visit_count += 1
            n.value_sum += value if n.player == root_player else -value

        current_env.board = root_board.copy()
        current_env.current_player = root_player

    visit_counts = np.zeros(9)
    for action, child in root.children.items():
        idx = action[0] * 3 + action[1]
        visit_counts[idx] = child.visit_count ** (1.0 / temperature)

    total = visit_counts.sum()
    return visit_counts / total if total > 0 else np.ones(9) / 9


def self_play_episode(model, num_simulations=800, temperature=1.0):
    env = TicTacToe()
    state = env.reset()
    states, mcts_policies, values = [], [], []

    while True:
        policy = mcts_search(state, model, env, num_simulations=num_simulations, temperature=temperature)
        states.append(state.copy())
        mcts_policies.append(policy)

        action_idx = np.random.choice(9, p=policy)
        action = (action_idx // 3, action_idx % 3)
        state, reward, done = env.step(action)

        if done:
            for i, player in enumerate([env.current_player * -1] * len(states)):
                if reward == 0:
                    values.append(0)
                else:
                    values.append(1 if player == reward else -1)
            return states, mcts_policies, values


def train_step(states, target_policies, target_values, model, optimizer):
    states = torch.FloatTensor(states).to(device)
    target_policies = torch.FloatTensor(target_policies).to(device)
    target_values = torch.FloatTensor(target_values).unsqueeze(1).to(device)

    optimizer.zero_grad()
    pred_policy, pred_value = model(states)
    policy_loss = -(target_policies * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_value, target_values)
    loss = policy_loss + value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def evaluate_model(model, num_games=100):
    model.eval()
    wins = 0
    draws = 0
    env = TicTacToe()

    for _ in range(num_games):
        state = env.reset()
        while True:
            policy = mcts_search(state, model, env, num_simulations=200, temperature=0.1)
            action_idx = np.argmax(policy)
            action = (action_idx // 3, action_idx % 3)
            state, reward, done = env.step(action)
            if done:
                if reward == 1:
                    wins += 1
                elif reward == 0:
                    draws += 1
                break
            valid_moves = env.get_valid_moves()
            if valid_moves:
                action = random.choice(valid_moves)
                state, _, done = env.step(action)
                if done:
                    break

    model.train()
    return wins / num_games, draws / num_games


def save_checkpoint(model, optimizer, episode, path="checkpoint.pth"):
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def train():
    replay_buffer = ReplayBuffer(capacity=100000)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    best_win_rate = 0.0

    for episode in range(1000):
        temperature = 1.0 if episode < 500 else 0.1  # Reduce exploration after 500 episodes
        states, policies, values = self_play_episode(model, num_simulations=800, temperature=temperature)
        for s, p, v in zip(states, policies, values):
            replay_buffer.add(s, p, v)

        if len(replay_buffer) >= 64:
            for _ in range(10):  # Multiple training steps per episode
                states, policies, values = replay_buffer.sample(64)
                loss = train_step(states, policies, values, model, optimizer)
            lr_scheduler.step()

        if episode % 50 == 0:
            win_rate, draw_rate = evaluate_model(model, num_games=100)
            print(f"Episode {episode}, Loss: {loss:.4f}, Win Rate: {win_rate:.3f}, Draw Rate: {draw_rate:.3f}")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_checkpoint(model, optimizer, episode, "best_tictactoe.pth")


def play():
    env = TicTacToe()
    state = env.reset()
    model.eval()

    while True:
        env.render()
        if env.current_player == 1:
            move = input("Enter move (row,col): ")
            try:
                i, j = map(int, move.split(','))
                action = (i, j)
            except:
                print("Invalid input. Please use format: row,col (e.g., 0,0)")
                continue
        else:
            policy = mcts_search(state, model, env, num_simulations=200, temperature=0.1)
            action_idx = np.argmax(policy)
            action = (action_idx // 3, action_idx % 3)
            print(f"AI plays: {action}")

        try:
            state, reward, done = env.step(action)
        except ValueError:
            print("Invalid move! Try again.")
            continue

        if done:
            env.render()
            if reward == 1:
                if env.current_player == 1:
                    print("X wins!")
                else:
                    print("O wins!")
            elif reward == -1:
                if env.current_player == -1:
                    print("O wins!")
                else:
                    print("X wins!")
            else:
                print("Draw!")
            break


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TicTacToeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train()
    play()