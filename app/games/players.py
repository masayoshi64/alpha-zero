import numpy as np

from .game import Game
from ..alpha_zero.mcts import MCTS
from .player_base import Player


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
