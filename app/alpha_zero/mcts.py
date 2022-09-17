from math import sqrt
from typing import List

import torch
import torch.nn as nn

from ..games.game import Game


class MCTS:
    def __init__(
        self, game: Game, model: nn.Module, alpha: float, tau: float, num_search: int
    ):
        self.game = game
        self.visited = set()
        self.model = model
        self.Q = dict()
        self.N = dict()
        self.P = dict()
        self.alpha = alpha
        self.tau = tau
        self.num_search = num_search

    def search(self, board: List[List[float]], player: int) -> float:
        if self.game.get_game_ended(board, player):
            return self.game.get_reward(board, player)

        s = self.game.hash(board, player)
        action_size = self.game.get_action_size()
        if s not in self.visited:
            self.visited.add(s)
            cboard = self.game.get_canonical_form(board, player)
            p, v = self.model(torch.Tensor(cboard))
            p = p.detach().numpy()[0].tolist()
            v = v.detach().numpy()[0, 0]
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

    def get_action_prob(self, board: List[List[float]]) -> List[float]:
        for i in range(self.num_search):
            self.search(board, 1)
        s = self.game.hash(board, 1)
        p = []
        for a in range(self.game.get_action_size()):
            p.append(self.N[s][a] ** (1 / self.tau))
        sm = sum(p)
        for a in range(len(p)):
            p[a] /= sm
        return p
