"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""

# import numpy as np
# import gomoku as gm

# class Submission:
#     def __init__(self, board_size, win_size):
#         self.board_size = board_size
#         self.win_size = win_size
#         self.learning_rate = 0.5
#         self.discount_factor = 0.5
#         self.epsilon = 0.1
#         self.q_values = {}

#     def __call__(self, state):
#         if np.random.rand() < self.epsilon:
#             # Explore: Choose a random action
#             action = state.valid_actions()[np.random.choice(len(state.valid_actions()))]
#         else:
#             # Exploit: Choose the action with the highest Q-value or evaluation
#             q_action = self.get_best_action(state)
#             winning_action = self.get_winning_action(state)
#             defensive_action = self.get_defensive_action(state)
#             eval_action = winning_action or defensive_action or q_action
#             action = eval_action if np.random.rand() < self.epsilon else eval_action

#         # Perform the chosen action and update Q-values
#         next_state = state.perform(action)
#         reward = self.calculate_reward(next_state)
#         self.update_q_values(state, action, reward, next_state)

#         return action

#     def get_winning_action(self, state):
#         player = gm.MAX if state.is_max_turn() else gm.MIN

#         # Check if the AI is one move away from winning
#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))

#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             if pos is not None:
#                 return pos  # Make the winning move

#         return None  # No winning move available

#     def get_defensive_action(self, state):
#         opponent = gm.MIN if state.is_max_turn() else gm.MAX

#         # Check if opponent is one move away from winning
#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))

#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             if pos is not None:
#                 return pos  # Block the opponent's winning move

#         return None  # No defensive move available

#     def find_empty(self, state, p, r, c):
#         if p == 0:  # horizontal
#             return r, c + state.board[gm.EMPTY, r, c:c + state.win_size].argmax()
#         if p == 1:  # vertical
#             return r + state.board[gm.EMPTY, r:r + state.win_size, c].argmax(), c
#         if p == 2:  # diagonal
#             rng = np.arange(state.win_size)
#             offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
#             return r + offset, c + offset
#         if p == 3:  # antidiagonal
#             rng = np.arange(state.win_size)
#             offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
#             return r - offset, c + offset
#         # None indicates no empty found
#         return None

#     def get_q_value(self, state, action):
#         state_key = self.state_to_key(state)
#         if state_key not in self.q_values:
#             self.q_values[state_key] = {}

#         if action not in self.q_values[state_key]:
#             self.q_values[state_key][action] = 0.0

#         return self.q_values[state_key][action]

#     def update_q_values(self, state, action, reward, next_state):
#         current_q = self.get_q_value(state, action)
#         max_future_q = max([self.get_q_value(next_state, a) for a in state.valid_actions()])
#         new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
#         self.q_values[self.state_to_key(state)][action] = new_q

#     def get_best_action(self, state):
#         return max(state.valid_actions(), key=lambda a: self.get_q_value(state, a))

#     def calculate_reward(self, state):
#         if state.is_game_over():
#             if state.current_score() == 1:
#                 return 1.0  # Max player (AI) wins
#             elif state.current_score() == -1:
#                 return -1.0  # Min player wins
#             else:
#                 return 0.0  # It's a draw
#         return 0.0  # No immediate reward

#     def evaluate(self, state):
#         # A simple evaluation function based on the number of player's pieces
#         max_count = state.board[gm.MAX].sum()
#         min_count = state.board[gm.MIN].sum()
#         return max_count - min_count

#     def state_to_key(self, state):
#         # Convert the state to a string key for Q-values dictionary
#         return str(state.board)



# import numpy as np
# import gomoku as gm

# class Submission:
#     def __init__(self, board_size, win_size, max_depth=4):
#         self.board_size = board_size
#         self.win_size = win_size
#         self.max_depth = max_depth

#     def __call__(self, state):
#         _, action = self.minimax_alpha_beta(state, self.max_depth)
#         return action

#     def get_winning_action(self, state):
#         player = gm.MAX if state.is_max_turn() else gm.MIN
#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))

#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             if pos is not None:
#                 return pos  # Make the winning move

#         return None

#     def get_defensive_action(self, state):
#         opponent = gm.MIN if state.is_max_turn() else gm.MAX
#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))

#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             if pos is not None:
#                 return pos  # Block the opponent's winning move

#         return None

#     def find_empty(self, state, p, r, c):
#         if p == 0:  # horizontal
#             return r, c + state.board[gm.EMPTY, r, c:c + state.win_size].argmax()
#         if p == 1:  # vertical
#             return r + state.board[gm.EMPTY, r:r + state.win_size, c].argmax(), c
#         if p == 2:  # diagonal
#             rng = np.arange(state.win_size)
#             offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
#             return r + offset, c + offset
#         if p == 3:  # antidiagonal
#             rng = np.arange(state.win_size)
#             offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
#             return r - offset, c + offset
#         return None

#     def minimax_alpha_beta(self, state, max_depth, alpha=-np.inf, beta=np.inf):
#         score, action = self.look_ahead(state)
#         if score != 0:
#             return score, action

#         if state.is_game_over():
#             return state.current_score(), None

#         actions = state.valid_actions()
#         rank = -state.corr[:, 1:].sum(axis=(0, 1)) - np.random.rand(*state.board.shape[1:])
#         rank = rank[state.board[gm.EMPTY] > 0]
#         scrambler = np.argsort(rank)

#         if max_depth == 0:
#             return state.current_score(), actions[scrambler[0]]

#         if self.turn_bound(state) > max_depth:
#             return 0, actions[scrambler[0]]

#         best_action = None
#         if state.is_max_turn():
#             bound = -np.inf
#             for a in scrambler:
#                 action = actions[a]
#                 child = state.perform(action)
#                 utility, _ = self.minimax_alpha_beta(child, max_depth - 1, alpha, beta)

#                 if utility > bound:
#                     bound, best_action = utility, action
#                 if bound >= beta:
#                     break
#                 alpha = max(alpha, bound)

#         else:
#             bound = +np.inf
#             for a in scrambler:
#                 action = actions[a]
#                 child = state.perform(action)
#                 utility, _ = self.minimax_alpha_beta(child, max_depth - 1, alpha, beta)

#                 if utility < bound:
#                     bound, best_action = utility, action
#                 if bound <= alpha:
#                     break
#                 beta = min(beta, bound)

#         return bound, best_action

#     def look_ahead(self, state):
#         player = state.current_player()
#         sign = +1 if player == gm.MAX else -1
#         magnitude = state.board[gm.EMPTY].sum()

#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))
#         if idx.shape[0] > 0:
#             p, r, c = idx[0]
#             action = self.find_empty(state, p, r, c)
#             return sign * magnitude, action

#         opponent = gm.MIN if state.is_max_turn() else gm.MAX
#         loss_empties = set()
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))
#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             loss_empties.add(pos)
#             if len(loss_empties) > 1:
#                 score = -sign * (magnitude - 1)
#                 return score, pos

#         return 0, None
#     def get_defensive_action(self, state):
#         opponent = gm.MIN if state.is_max_turn() else gm.MAX
#         corr = state.corr
#         idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))

#         for p, r, c in idx:
#             pos = self.find_empty(state, p, r, c)
#             if pos is not None:
#                 blocking_action = self.get_blocking_action(state, pos)
#                 if blocking_action is not None:
#                     return blocking_action  # Block the opponent's winning move

#         return None  # No defensive move available

#     def get_blocking_action(self, state, pos):
#         # Check if blocking the opponent's move would create a winning move for the AI
#         for p in range(4):
#             row, col = pos
#             for _ in range(self.win_size - 1):
#                 row, col = row - self.directions[p][0], col - self.directions[p][1]
#             if state.is_valid_action((row, col)) and state.board[gm.EMPTY, row, col]:
#                 # Check if blocking the opponent's move doesn't create a winning move for them
#                 temp_state = state.copy()
#                 temp_state = temp_state.perform((row, col))
#                 if not self.get_winning_action(temp_state):
#                     return row, col

#         return None

#     def turn_bound(self, state):
#         is_max = state.is_max_turn()
#         fewest_moves = state.board[gm.EMPTY].sum()

#         corr = state.corr
#         min_routes = (corr[:, gm.EMPTY] + corr[:, gm.MIN] == state.win_size)
#         max_routes = (corr[:, gm.EMPTY] + corr[:, gm.MAX] == state.win_size)
#         min_turns = 2 * corr[:, gm.EMPTY] - (0 if is_max else 1)
#         max_turns = 2 * corr[:, gm.EMPTY] - (1 if is_max else 0)

#         if min_routes.any():
#             moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
#             fewest_moves = min(fewest_moves, moves_to_win)
#         if max_routes.any():
#             moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
#             fewest_moves = min(fewest_moves, moves_to_win)

#         return fewest_moves

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CNNModel(nn.Module):
    def _init_(self, board_size, win_size):
        super(CNNModel, self)._init_()
        self.board_size = board_size
        self.win_size = win_size

        # Define your CNN layers here
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 256)
        self.fc2 = nn.Linear(256, board_size * board_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Define the forward pass of your CNN
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class GomocupDataset:
    def _init_(self, dataset):
        self.dataset = dataset

    def get_state_at_index(self, index):
        # Assuming your dataset has a method to retrieve Gomoku state at a given index
        return self.dataset[index]

class Submission:
    def __init__(self, board_size, win_size, gomocup_dataset=None):
        if gomocup_dataset==None:
            gomocup_dataset=GomocupDataset
        self.board_size = board_size
        self.win_size = win_size
        self.gomocup_dataset = GomocupDataset(gomocup_dataset)

        # Initialize your CNN model
        self.model = CNNModel(board_size, win_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def preprocess_state(self, state):
        # Convert the Gomoku state to a format suitable for the CNN input
        index = state.get_current_index()
        board = self.gomocup_dataset.get_state_at_index(index)

        # Assume the board is a 2D numpy array
        board = np.expand_dims(board, axis=0)  # Add a batch dimension
        board = torch.from_numpy(board).float()

        return board

    def _call_(self, state):
        # Implement the logic for making a move using the CNN model
        board_tensor = self.preprocess_state(state)

        # Forward pass through the CNN model
        output = self.model(board_tensor)

        # Get the index of the action with the highest probability
        action_index = torch.argmax(output).item()

        # Convert the flattened index to (row, column) coordinates
        action_row = action_index // self.board_size
        action_col = action_index % self.board_size

        return action_row, action_col