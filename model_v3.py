"""
AlphaZero-style MCTS + Neural Network for Gomoku (single-file).

Dependencies:
  - Python 3.8+
  - PyTorch
  - NumPy

This is a compact, educational implementation, not highly optimized for production.
Features:
  - Gomoku game (configurable board size and connect-k)
  - Residual ConvNet with policy + value heads (PyTorch)
  - PUCT MCTS that uses network priors and value (no rollouts)
  - Self-play generator and training loop with replay buffer
  - Dirichlet noise at root, temperature for policy targets

Usage (quick):
  - pip install torch torchvision numpy
  - python alphazero_gomoku.py --train

Adjust hyperparameters near the top of the file.
"""

import math
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------- Hyperparameters ---------------------------
BOARD_SIZE = 9           # Gomoku board (9x9 recommended for speed). Use 15 for standard.
CONNECT_N = 5            # How many in a row to win
N_RESIDUAL_BLOCKS = 6
CHANNELS = 128
SELFPLAY_EPISODES_PER_ITER = 30
MCTS_SIMULATIONS = 400   # number of MCTS simulations per move
CPUCT = 1.5
DIRICHLET_ALPHA = 0.3    # for root noise (depends on board size)
ROOT_NOISE_EPS = 0.25
REPLAY_BUFFER_SIZE = 20000
BATCH_SIZE = 64
TRAINING_STEPS_PER_ITER = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALUE_LOSS_WEIGHT = 1.0
POLICY_LOSS_WEIGHT = 1.0


# --------------------------- Gomoku Game ---------------------------
class Gomoku:
    def __init__(self, n=BOARD_SIZE, connect=CONNECT_N):
        self.n = n
        self.connect = connect
        # 0 empty, 1 black, 2 white
        self.board = np.zeros((n, n), dtype=np.int8)
        self.player_to_move = 1
        self.last_move = None

    def clone(self):
        g = Gomoku(self.n, self.connect)
        g.board = self.board.copy()
        g.player_to_move = self.player_to_move
        g.last_move = self.last_move
        return g

    def do_move(self, move):
        r, c = move
        assert self.board[r, c] == 0
        self.board[r, c] = self.player_to_move
        self.last_move = (r, c)
        self.player_to_move = 3 - self.player_to_move

    def legal_moves(self):
        empties = np.argwhere(self.board == 0)
        return [tuple(x) for x in empties]

    def is_on_board(self, r, c):
        return 0 <= r < self.n and 0 <= c < self.n

    def check_winner(self):
        # return 1/2 for winner, 0 for draw, None for not finished
        n, k = self.n, self.connect
        b = self.board
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for r in range(n):
            for c in range(n):
                if b[r,c] == 0:
                    continue
                player = b[r,c]
                for dr, dc in directions:
                    count = 1
                    rr, cc = r+dr, c+dc
                    while self.is_on_board(rr, cc) and b[rr, cc] == player:
                        count += 1
                        if count >= k:
                            return int(player)
                        rr += dr; cc += dc
        if np.all(b != 0):
            return 0
        return None

    def game_over(self):
        return self.check_winner() is not None

    def encode(self):
        # Return a tensor with shape (2, n, n): plane for current player and opponent
        current = (self.board == self.player_to_move).astype(np.float32)
        opp = (self.board == (3 - self.player_to_move)).astype(np.float32)
        return np.stack([current, opp], axis=0)

    def display(self):
        chars = '.XO'
        for r in range(self.n):
            print(' '.join(chars[self.board[r,c]] for c in range(self.n)))
        print()


# --------------------------- Neural Network ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)


class AlphaZeroNet(nn.Module):
    def __init__(self, board_size, channels=CHANNELS, n_blocks=N_RESIDUAL_BLOCKS):
        super().__init__()
        self.board_size = board_size
        self.conv_in = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        # x shape: (B, 2, n, n)
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return F.log_softmax(p, dim=1), v


# --------------------------- MCTS (PUCT) ---------------------------
class MCTS:
    def __init__(self, net, cpuct=CPUCT, n_sims=MCTS_SIMULATIONS, dirichlet_alpha=DIRICHLET_ALPHA):
        self.net = net
        self.cpuct = cpuct
        self.n_sims = n_sims
        self.dirichlet_alpha = dirichlet_alpha

    def run(self, state):
        # The tree will be composed of dict keyed by state_key mapping to Node info
        # Node stores: P (prior), N (visits), W (total value), Q (W/N)
        self.root = {}
        root_key = self._state_key(state)
        # use network to initialize root priors and value
        policy_log, value = self._net_predict(state)
        priors = np.exp(policy_log.cpu().numpy().squeeze())
        legal = state.legal_moves()
        legal_mask = np.zeros(state.n * state.n, dtype=np.float32)
        for (r,c) in legal:
            legal_mask[r*state.n + c] = 1.0
        priors = priors * legal_mask
        if priors.sum() == 0:
            priors = legal_mask
        priors = priors / priors.sum()

        node = {
            'P': priors,      # prior for all actions
            'N': np.zeros_like(priors, dtype=np.int32),
            'W': np.zeros_like(priors, dtype=np.float32),
            'Q': np.zeros_like(priors, dtype=np.float32),
            'is_expanded': True,
            'value': float(value.item()),
            'state': state.clone()
        }
        # add Dirichlet noise to root priors for exploration
        dir_noise = np.random.dirichlet([self.dirichlet_alpha] * (state.n * state.n))
        node['P'] = node['P'] * (1 - ROOT_NOISE_EPS) + dir_noise * ROOT_NOISE_EPS
        self.root[root_key] = node

        for _ in range(self.n_sims):
            self._simulate(state)

        # After sims, build policy vector proportional to visit counts
        root = self.root[root_key]
        visits = root['N']
        pi = visits / (visits.sum() + 1e-10)
        return pi.reshape(state.n, state.n), root

    def _simulate(self, root_state):
        path = []  # list of (node_key, action_index)
        state = root_state.clone()
        node_key = self._state_key(state)
        node = self.root.get(node_key)

        # selection
        while True:
            if node is None:
                # unknown node -> expand here
                policy_log, value = self._net_predict(state)
                priors = np.exp(policy_log.cpu().numpy().squeeze())
                legal = state.legal_moves()
                legal_mask = np.zeros(state.n * state.n, dtype=np.float32)
                for (r,c) in legal:
                    legal_mask[r*state.n + c] = 1.0
                priors = priors * legal_mask
                if priors.sum() == 0:
                    priors = legal_mask
                priors = priors / priors.sum()
                node = {
                    'P': priors,
                    'N': np.zeros_like(priors, dtype=np.int32),
                    'W': np.zeros_like(priors, dtype=np.float32),
                    'Q': np.zeros_like(priors, dtype=np.float32),
                    'is_expanded': True,
                    'value': float(value.item()),
                    'state': state.clone()
                }
                self.root[node_key] = node
                break

            # if terminal, backpropagate value
            winner = state.check_winner()
            if winner is not None:
                if winner == 0:
                    leaf_value = 0.0
                else:
                    # value is +1 for current player win in our perspective? we make value from current player's view
                    leaf_value = 1.0 if winner == (3 - state.player_to_move) else -1.0
                self._backpropagate(path, leaf_value)
                return

            # select action maximizing PUCT
            total_N = node['N'].sum()
            # U = c * P * sqrt(sumN) / (1 + N_a)
            uct = node['Q'] + self.cpuct * node['P'] * math.sqrt(max(1, total_N)) / (1 + node['N'])
            # mask illegal actions
            legal = state.legal_moves()
            legal_idx = [r*state.n + c for (r,c) in legal]
            # choose argmax among legal
            best_a = max(legal_idx, key=lambda a: float(uct[a]))

            path.append((node_key, best_a))
            # apply move
            r, c = divmod(best_a, state.n)
            state.do_move((r, c))
            node_key = self._state_key(state)
            node = self.root.get(node_key)

        # expansion happened above; leaf node created, backprop with network value
        leaf_value = node['value']
        # value returned by network is from the perspective of the current player in that node
        # we need to backpropagate the value from perspective of the player who just moved
        # path contains moves made from root -> leaf; when backpropagating we treat value sign accordingly
        self._backpropagate(path, leaf_value)

    def _backpropagate(self, path, leaf_value):
        # path is list of (node_key, action_index) from root downwards
        # leaf_value is network value from perspective of the player to move at leaf
        # As we go up the path, flip the sign because players alternate.
        value = leaf_value
        for node_key, a in reversed(path):
            node = self.root[node_key]
            # update statistics for action a
            node['N'][a] += 1
            node['W'][a] += value
            node['Q'][a] = node['W'][a] / node['N'][a]
            value = -value

    def _net_predict(self, state):
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(state.encode(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logp, v = self.net(x)
        return logp.detach().cpu(), v.detach().cpu()

    def _state_key(self, state):
        # compact state key for dict: bytes
        return state.board.tobytes() + bytes([state.player_to_move])


# --------------------------- Replay Buffer & Dataset ---------------------------
SPExample = namedtuple('SPExample', ['state', 'pi', 'value'])


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, example):
        self.buffer.append(example)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b.state for b in batch], axis=0)
        pis = np.stack([b.pi for b in batch], axis=0)
        values = np.array([b.value for b in batch], dtype=np.float32)
        return states, pis, values

    def __len__(self):
        return len(self.buffer)


class SelfPlayDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay = list(replay_buffer.buffer)

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        ex = self.replay[idx]
        return ex.state, ex.pi, ex.value


# --------------------------- Self-Play ---------------------------
class SelfPlay:
    def __init__(self, net, mcts_simulations=MCTS_SIMULATIONS, cpuct=CPUCT):
        self.net = net
        self.mcts = MCTS(net, cpuct=cpuct, n_sims=mcts_simulations)

    def execute_episode(self, temp=1.0):
        game = Gomoku()
        states = []
        mcts_pis = []
        current_players = []

        while True:
            pi_matrix, root = self.mcts.run(game)
            # temperature: convert pi_matrix to flattened distribution
            pi_flat = pi_matrix.reshape(-1)
            if temp == 0:
                # deterministic: choose argmax
                a = int(np.argmax(pi_flat))
                pi_target = np.zeros_like(pi_flat)
                pi_target[a] = 1.0
            else:
                # sample move proportional to (pi^(1/temp))
                pi_temp = np.power(pi_flat + 1e-10, 1.0 / temp)
                pi_temp = pi_temp / pi_temp.sum()
                a = np.random.choice(len(pi_temp), p=pi_temp)
                pi_target = pi_temp

            r, c = divmod(a, game.n)
            states.append(game.encode())
            mcts_pis.append(pi_target.reshape(game.n, game.n))
            current_players.append(game.player_to_move)

            game.do_move((r, c))
            winner = game.check_winner()
            if winner is not None:
                # compute z values (from perspective of player who played at that state)
                z = []
                for player in current_players:
                    if winner == 0:
                        z.append(0.0)
                    else:
                        z.append(1.0 if winner == player else -1.0)
                return [SPExample(s, p, v) for s, p, v in zip(states, mcts_pis, z)]


# --------------------------- Training ---------------------------
class Trainer:
    def __init__(self, net):
        self.net = net.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.replay = ReplayBuffer()

    def self_play_and_fill(self, n_episodes=SELFPLAY_EPISODES_PER_ITER):
        sp = SelfPlay(self.net)
        for _ in range(n_episodes):
            examples = sp.execute_episode(temp=1.0)
            for ex in examples:
                self.replay.push(ex)

    def train_step(self, batch_size=BATCH_SIZE, steps=TRAINING_STEPS_PER_ITER):
        if len(self.replay) < batch_size:
            return
        dataset = SelfPlayDataset(self.replay)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.net.train()
        for step, batch in enumerate(loader):
            if step >= steps:
                break
            states, pis, values = batch
            states = torch.tensor(states, dtype=torch.float32).to(DEVICE)
            pis = torch.tensor(pis.reshape(pis.size(0), -1), dtype=torch.float32).to(DEVICE)
            values = torch.tensor(values, dtype=torch.float32).to(DEVICE)

            log_policy, pred_values = self.net(states)
            policy_loss = -torch.mean(torch.sum(pis * log_policy, dim=1))
            value_loss = F.mse_loss(pred_values, values)
            loss = POLICY_LOSS_WEIGHT * policy_loss + VALUE_LOSS_WEIGHT * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
            self.optimizer.step()

    def save_model(self, path='az_gomoku.pth'):
        torch.save(self.net.state_dict(), path)


# --------------------------- Main / CLI ---------------------------
def main(args):
    net = AlphaZeroNet(BOARD_SIZE).to(DEVICE)
    trainer = Trainer(net)

    if args.train:
        iters = args.iters
        for it in range(iters):
            t0 = time.time()
            print(f"Iteration {it+1}/{iters}: Self-play {SELFPLAY_EPISODES_PER_ITER} games...")
            trainer.self_play_and_fill(SELFPLAY_EPISODES_PER_ITER)
            print(f"Replay buffer size: {len(trainer.replay)}")
            print("Training network...")
            trainer.train_step()
            t1 = time.time()
            print(f"Iter {it+1} finished in {t1-t0:.1f}s")
            trainer.save_model()
    elif args.selfplay:
        net.load_state_dict(torch.load(args.model, map_location=DEVICE))
        sp = SelfPlay(net)
        for _ in range(args.games):
            ex = sp.execute_episode()
            print(f"Generated {len(ex)} examples from one episode")
    elif args.play:

        # play against a human
        net.load_state_dict(torch.load(args.model, map_location=DEVICE))
        player = args.player
        mcts = MCTS(net)
        game = Gomoku()
        print("Gomoku: you are X (player 1) if chosen, board indices are 0..n-1 for rows and cols")
        game.display()
        while not game.game_over():
            if game.player_to_move == player:
                move_txt = input("Your move (row col): ")
                r,c = map(int, move_txt.split())
                game.do_move((r,c))
            else:
                pi, root = mcts.run(game)
                a = int(np.argmax(pi.reshape(-1)))
                r,c = divmod(a, game.n)
                print(f"AI moves: {r} {c}")
                game.do_move((r,c))
            game.display()
        winner = game.check_winner()
        if winner == 0:
            print("Draw")
        else:
            print(f"Player {winner} wins")
    else:
        print("Run with --train or --play or --selfplay")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--selfplay', action='store_true')
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--model', type=str, default='az_gomoku.pth')
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--player', type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)
