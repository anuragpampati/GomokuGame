import numpy as np
import gomoku as gm


class Submission:
    def __init__(self, board_size, win_size, max_depth=4):
        self.board_size = board_size
        self.win_size = win_size
        self.max_depth = max_depth

    def __call__(self, state):
        _, action = self.minimax_alpha_beta(state, self.max_depth)
        return action

    def get_winning_action(self, state):
        player = gm.MAX if state.is_max_turn() else gm.MIN
        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))

        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            if pos is not None:
                return pos  # Make the winning move

        return None

    def get_defensive_action(self, state):
        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))

        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            if pos is not None:
                return pos  # Block the opponent's winning move

        return None

    def find_empty(self, state, p, r, c):
        if p == 0:  # horizontal
            return r, c + state.board[gm.EMPTY, r, c:c + state.win_size].argmax()
        if p == 1:  # vertical
            return r + state.board[gm.EMPTY, r:r + state.win_size, c].argmax(), c
        if p == 2:  # diagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
            return r + offset, c + offset
        if p == 3:  # antidiagonal
            rng = np.arange(state.win_size)
            offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
            return r - offset, c + offset
        return None

    def minimax_alpha_beta(self, state, max_depth, alpha=-np.inf, beta=np.inf):
        score, action = self.look_ahead(state)
        if score != 0:
            return score, action

        if state.is_game_over():
            return state.current_score(), None

        actions = state.valid_actions()
        rank = -state.corr[:, 1:].sum(axis=(0, 1)) - np.random.rand(*state.board.shape[1:])
        rank = rank[state.board[gm.EMPTY] > 0]
        scrambler = np.argsort(rank)

        if max_depth == 0:
            return state.current_score(), actions[scrambler[0]]

        if self.turn_bound(state) > max_depth:
            return 0, actions[scrambler[0]]

        best_action = None
        if state.is_max_turn():
            bound = -np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax_alpha_beta(child, max_depth - 1, alpha, beta)

                if utility > bound:
                    bound, best_action = utility, action
                if bound >= beta:
                    break
                alpha = max(alpha, bound)

        else:
            bound = +np.inf
            for a in scrambler:
                action = actions[a]
                child = state.perform(action)
                utility, _ = self.minimax_alpha_beta(child, max_depth - 1, alpha, beta)

                if utility < bound:
                    bound, best_action = utility, action
                if bound <= alpha:
                    break
                beta = min(beta, bound)

        return bound, best_action

    def look_ahead(self, state):
        player = state.current_player()
        sign = +1 if player == gm.MAX else -1
        magnitude = state.board[gm.EMPTY].sum()

        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size - 1))
        if idx.shape[0] > 0:
            p, r, c = idx[0]
            action = self.find_empty(state, p, r, c)
            return sign * magnitude, action

        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        loss_empties = set()
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))
        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            loss_empties.add(pos)
            if len(loss_empties) > 1:
                score = -sign * (magnitude - 1)
                return score, pos

        return 0, None
    def get_defensive_action(self, state):
        opponent = gm.MIN if state.is_max_turn() else gm.MAX
        corr = state.corr
        idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))

        for p, r, c in idx:
            pos = self.find_empty(state, p, r, c)
            if pos is not None:
                blocking_action = self.get_blocking_action(state, pos)
                if blocking_action is not None:
                    return blocking_action  # Block the opponent's winning move

        return None  # No defensive move available

    def get_blocking_action(self, state, pos):
        # Check if blocking the opponent's move would create a winning move for the AI
        for p in range(4):
            row, col = pos
            for _ in range(self.win_size - 1):
                row, col = row - self.directions[p][0], col - self.directions[p][1]
            if state.is_valid_action((row, col)) and state.board[gm.EMPTY, row, col]:
                # Check if blocking the opponent's move doesn't create a winning move for them
                temp_state = state.copy()
                temp_state = temp_state.perform((row, col))
                if not self.get_winning_action(temp_state):
                    return row, col

        return None

    def turn_bound(self, state):
        is_max = state.is_max_turn()
        fewest_moves = state.board[gm.EMPTY].sum()

        corr = state.corr
        min_routes = (corr[:, gm.EMPTY] + corr[:, gm.MIN] == state.win_size)
        max_routes = (corr[:, gm.EMPTY] + corr[:, gm.MAX] == state.win_size)
        min_turns = 2 * corr[:, gm.EMPTY] - (0 if is_max else 1)
        max_turns = 2 * corr[:, gm.EMPTY] - (1 if is_max else 0)

        if min_routes.any():
            moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)
        if max_routes.any():
            moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
            fewest_moves = min(fewest_moves, moves_to_win)

        return fewest_moves

