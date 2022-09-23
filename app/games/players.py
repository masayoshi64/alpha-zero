import numpy as np
import torch.nn as nn
import torch

from .game import Game
from ..alpha_zero.mcts import MCTS
from .player_base import Player
from ..alpha_zero.utils import get_board_view


class RandomPlayer(Player):
    """合法手の中から一様ランダムにactionを選択するプレイヤー"""

    def __init__(self, game: Game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.get_action_size())
        while not self.game.is_valid(board, 1, a):
            a = np.random.randint(self.game.get_action_size())
        return a


class HumanPlayer(Player):
    """inputにより人間が直接手を指定するプレイヤー"""

    def __init__(self, game: Game):
        self.game = game

    def play(self, board):
        a = int(input())
        while not self.game.is_valid(board, 1, a):
            a = int(input())
        return a


class MCTSPlayer(Player):
    def __init__(self, mcts: MCTS):
        self.mcts = mcts

    def play(self, board):
        p = self.mcts.get_action_prob(board)
        return np.random.choice(list(range(len(p))), p=p)

    def reset(self):
        self.mcts.reset()


class NeuralNetPlayer(Player):
    def __init__(self, model: nn.Module):
        self.model = model

    def play(self, board):
        p, v = self.model(torch.Tensor(get_board_view(board)))
        p = p[0].detach().numpy()
        return np.argmax(p)


class AlphaBetaPlayer(Player):
    def __init__(self, game: Game):
        self.game = game

    def play(self, board):
        best_action = 0
        alpha = -float("inf")
        beta = float("inf")
        for action in range(self.game.get_action_size()):
            if self.game.is_valid(board, 1, action):
                next_board, next_player = self.game.get_next_state(board, 1, action)
                if next_player == 1:
                    score = self.search(next_board, next_player, alpha, beta)
                else:
                    score = -self.search(next_board, next_player, -beta, -alpha)
                if alpha < score:
                    alpha = score
                    best_action = action
        return best_action

    def search(self, board, player, alpha, beta):
        if self.game.get_game_ended(board, player):
            return self.game.get_reward(board, player)

        for action in range(self.game.get_action_size()):
            if self.game.is_valid(board, player, action):
                next_board, next_player = self.game.get_next_state(
                    board, player, action
                )
                # playerから見たboardの評価値
                if next_player == player:
                    score = self.search(next_board, next_player, alpha, beta)
                else:
                    score = -self.search(next_board, next_player, -beta, -alpha)
                # 関心のある値の上限であるbetaをscoreが超えたら打ち切り
                if beta <= score:
                    return score
                alpha = max(alpha, score)
        return alpha
