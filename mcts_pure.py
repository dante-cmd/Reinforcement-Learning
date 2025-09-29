# Pure MCTS (no neural network) implementation + Tic-Tac-Toe demo
# This code will:
# 1) Define a small Game interface (RoomLog implementation)
# 2) Implement a pure MCTS algorithm (Selection, Expansion, Simulation, Backpropagation)
# 3) Demonstrate a single MCTS decision from the empty board and play one game where MCTS plays vs Random
# Notes:
# - This is intentionally small and didactic, not hyper-optimized.
# - The rollout policy is purely random (pure MCTS).
# - The UCT formula uses an exploration constant c (default sqrt(2)).

import math, random, time
random.seed(42)


class TicTacToe:
    def __init__(self):
        # board: 0 empty, 1 player1, 2 player2
        self.board = [0]*9
        self.player_to_move = 1

    def clone(self):
        other = TicTacToe()
        other.board = self.board.copy()
        other.player_to_move = self.player_to_move
        return other

    def do_move(self, move:int):
        assert self.board[move] == 0, "Invalid move"
        self.board[move] = self.player_to_move
        # switch player
        self.player_to_move = 3 - self.player_to_move

    def get_legal_moves(self):
        return [i for i,v in enumerate(self.board) if v == 0]

    def get_winner(self):
        # return 1 or 2 for winner, 0 for draw, None if game not finished
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for (a,b,c) in lines:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        if all(v != 0 for v in self.board):
            return 0  # draw
        return None  # game not finished

    def is_terminal(self):
        return self.get_winner() is not None

    def display(self):
        chars = ['.', 'X', 'O']
        s = ""
        for i in range(9):
            s += chars[self.board[i]] + (" " if (i%3)!=2 else "\n")
        print(s.strip())


class Node:
    def __init__(self, move=None, parent=None, player_just_moved=None, untried_actions=None):
        self.move = move                  # the move that led to this node (from parent)
        self.parent = parent              # parent node
        self.children = []                # list of child nodes
        self.wins = 0.0                   # number of wins for player_just_moved
        self.visits = 0                   # visit count
        self.untried_actions = [] if untried_actions is None else untried_actions[:]  # moves not expanded yet
        self.player_just_moved = player_just_moved  # who moved to get to this node

    def uct_select_child(self, c_param=math.sqrt(2)):
        # Select a child according to UCT (upper confidence bound applied to trees)
        # If a child has 0 visits we consider its UCT value infinite to ensure it's visited.
        best = max(self.children, key=lambda child: (
            float('inf') if child.visits == 0 else
            (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
        ))
        return best

    def add_child(self, move, player_just_moved, untried_actions):
        child = Node(move=move, parent=self, player_just_moved=player_just_moved, untried_actions=untried_actions)
        self.untried_actions.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        # result: winner 1/2/0 (0 -> draw)
        self.visits += 1
        if result == 0:
            self.wins += 0.5  # half for draw
        elif result == self.player_just_moved:
            self.wins += 1.0
        # if result != player_just_moved -> add 0


def UCT(root_state, itermax=1000, c_param=math.sqrt(2)):
    root_node = Node(move=None,
                     parent=None,
                     player_just_moved=3 - root_state.player_to_move,
                     untried_actions=root_state.get_legal_moves())

    for i in range(itermax):
        node = root_node
        state = root_state.clone()

        # 1. Selection: descend until we find a node with untried actions or a leaf
        while node.untried_actions == [] and node.children:
            node = node.uct_select_child(c_param)
            state.do_move(node.move)

        # 2. Expansion: if we can expand (i.e. state not terminal) pick an untried action
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            state.do_move(action)
            # create child node for the action
            child_untried = state.get_legal_moves()
            child_player_just_moved = 3 - state.player_to_move
            node = node.add_child(move=action,
                                  player_just_moved=child_player_just_moved,
                                  untried_actions=child_untried)

        # 3. Simulation: play randomly until the game ends
        while state.get_winner() is None:
            possible_moves = state.get_legal_moves()
            state.do_move(random.choice(possible_moves))

        # 4. Backpropagation: update node statistics with simulation result
        result = state.get_winner()  # 1/2/0
        while node is not None:
            node.update(result)
            node = node.parent

    # return the move that was most visited
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.move, root_node  # also return root node so the caller can inspect children statistics


# --- Demo: use UCT on an empty RoomLog board ---

print("Demo: single MCTS decision from the empty Tic-Tac-Toe board")
state = TicTacToe()
start_time = time.time()
move, root_node = UCT(state, itermax=2000)   # 2000 rollouts from empty board
duration = time.time() - start_time
print(f"Chosen move (index 0..8): {move}, duration: {duration:.3f}s")
print("Root's children statistics (move: visits, win_rate_for_player_who_just_moved):")
for child in sorted(root_node.children, key=lambda c: -c.visits):
    wr = child.wins / child.visits if child.visits > 0 else 0
    print(f" move {child.move}: visits={child.visits}, wins={child.wins:.1f}, win_rate={wr:.3f}")
print("\nBoard (X=player1, O=player2). MCTS recommends move index:", move)
state.do_move(move)
state.display()

# --- Play a full game: MCTS (player1) vs Random (player2) ---
print("\nNow play one game with MCTS as player1 and Random as player2 (MCTS uses 600 rollouts per decision).")
state = TicTacToe()
while state.get_winner() is None:
    if state.player_to_move == 1:
        move, _ = UCT(state, itermax=600)
    else:
        move = random.choice(state.get_legal_moves())
    state.do_move(move)

print("Final board:")
state.display()
winner = state.get_winner()
if winner == 0:
    print("Result: Draw")
else:
    print(f"Result: Player {winner} ({'X' if winner==1 else 'O'}) wins")
