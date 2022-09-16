from math import sqrt

from ..games.game import Game


class MCTS:
    def __init__(self, game: Game, net, alpha):
        self.game = game
        self.visited = set()
        self.net = net
        self.Q = dict()
        self.N = dict()
        self.P = dict()
        self.alpha = alpha

    def search(self, board, player) -> float:
        if self.game.get_game_ended(board, player):
            return self.game.get_reward(board, player)

        s = self.game.hash(board, player)
        action_size = self.game.get_action_size()
        if s not in self.visited:
            self.visited.add(s)
            cboard = self.game.get_canonical_form(board, player)
            (p, v) = self.net.predict(cboard)
            self.P[s] = p
            self.Q[s] = [0] * action_size
            self.N[s] = [0] * action_size
            return v

        best_action = 0
        max_ucb = -float("inf")
        Ns = sum(self.N[s])

        for action in range(action_size):
            if self.game.is_valid(board, player, action):
                ucb = self.Q[s][action] + self.alpha * sqrt(Ns) / (
                    1 + self.N[s][action]
                )
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action

        next_board, next_player = self.game.get_next_state(board, player, best_action)
        v = -self.search(next_board, next_player)
        self.Q[s][best_action] = (
            self.Q[s][best_action] * self.N[s][best_action] + v
        ) / (self.N[s][best_action] + 1)
        self.N[s][best_action] += 1
        return v
