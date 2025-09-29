# Pure MCTS (no neural network) implementation + Tic-Tac-Toe demo
# This code will:
# 1) Define a small Game interface (RoomLog implementation)
# 2) Implement a pure MCTS algorithm (Selection, Expansion, Simulation, Backpropagation)
# 3) Demonstrate a single MCTS decision from the empty board and play one game where MCTS plays vs Random
# Notes:
# - This is intentionally small and didactic, not hyper-optimized.
# - The rollout policy is purely random (pure MCTS).
# - The UCT formula uses an exploration constant c (default sqrt(2)).
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import math, random, time
import concurrent.futures


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


class RoomLog:
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
        self.idx_item = 0
        self.n_assignments = 0
        self.stats = {'[Conflict]': 0,
                      '[Aforo-Alum<0]': 0,
                      '[Aforo=Alumn]|[Aforo-Alumn<=2]': 0,
                      '[Aforo-Alumn>2]': 0}
        self.n_franjas = self.get_n_franjas()
        self.n_aulas = len(self.roomlog.keys())
        # self.board = [0] * 9
        # self.player_to_move = 1

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
            for periodo_franja in self.dim_periodo_franja.keys():
                franjas = self.dim_periodo_franja[periodo_franja]['FRANJAS']
                dias = self.dim_periodo_franja[periodo_franja]['DIAS']
                for dia in dias:
                    room_log[aula][dia] = {}
                    for franja in franjas:
                        room_log[aula][dia][franja] = 0

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

    def do_move(self, action: int):
        # assert self.board[move] == 0, "Invalid move"
        # self.board[move] = self.player_to_move
        # switch player
        # self.player_to_move = 3 - self.player_to_move
        assert self.idx_item <= (len(self.items) - 1)
        # return None, None, True

        item = self.items[self.idx_item].copy()
        aula = self.dim_aulas['AULA'][action]
        aforo = self.dim_aulas['AFORO'][action]
        roomlog = self.roomlog.copy()
        # response = ''
        # '[Conflict]'
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
        # self.n_assignments += 1
        # for periodo_franja in self.dim_periodo_franja.keys():
        for dia in dias:
            for franja in franjas:
                roomlog[aula][dia][franja] = 1

        self.roomlog = roomlog.copy()
        # done = False
        self.idx_item += 1
        # next_state = self.get_state()
        # if next_state is None:
        #     done = True
        # info = item.copy()
        # info['AULA'] = aula
        # info['AFORO'] = aforo
        # info['RESPONSE'] = response
        return reward, response

    def get_legal_moves(self):
        if self.idx_item >= len(self.items):
            return []

        item = self.items[self.idx_item].copy()
        aulas = self.dim_aulas['AULA']
        roomlog = self.roomlog.copy()
        dias = self.dim_frecuencia[item['FRECUENCIA']]['DIAS']
        franjas = self.dim_horario[item['HORARIO']]

        available = []
        for idx, aula in enumerate(aulas):
            conflict = []
            for dia in dias:
                for franja in franjas:
                    conflict.append(True if roomlog[aula][dia][franja] == 1 else False)
            if not any(conflict):
                available.append(idx)

        # [i for i,v in enumerate(self.board) if v == 0]
        return available

    def get_assignments_rate(self):
        # return 1 or 2 for winner, 0 for draw, None if game not finished
        return self.n_assignments/len(self.items)  # game not finished

    def is_terminal(self):
        # self.get_winner() is not None
        return self.idx_item >= (len(self.items) - 1) | (len(self.get_legal_moves()) == 0)


class Node:
    def __init__(self, move=None, parent=None, untried_actions=None):
        self.move = move                  # the move that led to this node (from parent)
        self.parent = parent              # parent node
        self.children = []                # list of child nodes
        self.w = 0.0                   # number of wins for player_just_moved
        self.visits = 0                   # visit count
        self.untried_actions = [] if untried_actions is None else untried_actions[:]  # moves not expanded yet
        # self.player_just_moved = player_just_moved  # who moved to get to this node

    def uct_select_child(self, c_param=math.sqrt(2)):
        # Select a child according to UCT (upper confidence bound applied to trees)
        # If a child has 0 visits we consider its UCT value infinite to ensure it's visited.
        best = max(self.children, key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.w / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        ))
        return best

    def add_child(self, move, untried_actions):
        child = Node(move=move, parent=self, untried_actions=untried_actions)
        self.untried_actions.remove(move)
        self.children.append(child)
        return child

    def update(self, reward):
        # reward: winner 1/2/0 (0 -> draw)
        self.visits += 1
        # if reward == 0:
        self.w += reward
        # half for draw
        # elif reward == self.player_just_moved:
        #     self.w += 1.0
        # if result != player_just_moved -> add 0


def UCT(state, iter_max=5000, c_param=math.sqrt(2)):
    # PATH = Path("project")
    # SEDE = 'Ica'
    # iter_max = 5000
    # c_param=math.sqrt(2)
    # dataset_01 = DataSet(PATH)
    # state = RoomLog(dataset_01, SEDE)
    root_node = Node(move=None,
                     parent=None,
                     untried_actions=state.get_legal_moves())

    idx_item = state.idx_item
    roomlog = copy.deepcopy(state.roomlog.copy())
    # state.roomlog['101']['LUN']['07:00 - 08:30'] = 1

    for i in range(iter_max):
        # i = 0
        rewards = []
        node = root_node
        state.idx_item = idx_item
        state.roomlog = copy.deepcopy(roomlog.copy())

        # 1. Selection: descend until we find a node with untried actions or a leaf
        while node.untried_actions == [] and node.children:
            node = node.uct_select_child(c_param)
            reward, _ = state.do_move(node.move)
            rewards.append(reward)

        # 2. Expansion: if we can expand (i.e. state not terminal) pick an untried action
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            reward, _ = state.do_move(action)
            rewards.append(reward)

            # create child node for the action
            # state.items[state.idx_item]

            child_untried = state.get_legal_moves()
            node = node.add_child(move=action,
                                  untried_actions=child_untried)

        # 3. Simulation: play randomly until the game ends
        while not state.is_terminal():
            possible_moves = state.get_legal_moves()
            reward, _ = state.do_move(random.choice(possible_moves))
            rewards.append(reward)

        # 4. Backpropagation: update node statistics with simulation result
        # assignments_rate = state.get_assignments_rate()  # 1/2/0
        while node is not None:
            node.update(sum(rewards)/len(state.items))
            node = node.parent
            # # for reward in rewards:

    # return the move that was most visited
    best_child = max(root_node.children, key=lambda c: c.visits)
    state.idx_item = idx_item
    state.roomlog = copy.deepcopy(roomlog.copy())

    return best_child.move, root_node, state  # also return root node so the caller can inspect children statistics


def UCT_worker(state, iter_max, c_param):
    # Clone state for isolation
    cloned_state = RoomLog(state.dataset, state.sede)
    cloned_state.idx_item = state.idx_item
    cloned_state.roomlog = copy.deepcopy(state.roomlog)

    move, root, _ = UCT(cloned_state, iter_max=iter_max, c_param=c_param)
    return root


def parallel_UCT(state, iter_max=5000, c_param=math.sqrt(2), n_workers=4):
    # split iterations across workers
    iters_per_worker = iter_max // n_workers
    roots = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(UCT_worker, state, iters_per_worker, c_param)
                   for _ in range(n_workers)]
        for f in concurrent.futures.as_completed(futures):
            roots.append(f.result())

    # merge children statistics from workers
    merged_root = Node(move=None, parent=None,
                       untried_actions=state.get_legal_moves())
    move_to_node = {}

    for r in roots:
        for child in r.children:
            if child.move not in move_to_node:
                move_to_node[child.move] = Node(move=child.move,
                                                parent=merged_root,
                                                untried_actions=[])
            move_to_node[child.move].visits += child.visits
            move_to_node[child.move].w += child.w

    merged_root.children = list(move_to_node.values())
    best_child = max(merged_root.children, key=lambda c: c.visits)

    return best_child.move, merged_root, state


def demo():
    # --- Demo: use UCT on an empty RoomLog board ---
    path = Path("project")
    sede = 'Ica'
    dataset_01 = DataSet(path)
    state = RoomLog(dataset_01, sede)
    aulas = []
    while state.idx_item < len(state.items):
        start_time = time.time()
        result = copy.deepcopy(state.items[state.idx_item].copy())
        if len(state.get_legal_moves()) == 0:
            result['ASSIGNMENTS'] = {'AULA': None,
                                     'AFORO': None}
            state.idx_item += 1
        else:
            move, root_node, state = parallel_UCT(state, iter_max=5000, n_workers=7)
            # move, root_node, state = UCT(state, iter_max=5000)   # 2000 rollouts from empty board
            result['ASSIGNMENTS'] = {'AULA':state.dim_aulas['AULA'][move],
                                     'AFORO':state.dim_aulas['AFORO'][move]}
            state.do_move(move)
        aulas.append(result)
        duration = time.time() - start_time
        print(f"duration: {duration:.3f}s")
    df = pd.DataFrame(aulas)
    df.to_excel('sususu.xlsx', index=False)


if __name__ == '__main__':
    demo()


