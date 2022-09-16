import numpy as np

from app.alpha_zero.mcts import MCTS
from app.games.game import Game
from app.games.tictactoe import TicTacToeGame
from app.games.arena import Arena
from app.games.players import RandomPlayer


# MCTSによる遷移確率が最大の行動をとる
class MCTSPlayer:
    def __init__(self, mcts: MCTS, n):
        self.mcts = mcts
        self.n = n

    def play(self, board):
        for _ in range(self.n):
            self.mcts.search(board, 1)
        s = self.mcts.game.hash(board, 1)
        return np.argmax(self.mcts.N[s])


# 常に一様分布と0を返すネットワーク
class ConstantNet:
    def __init__(self, game: Game):
        self.game = game

    def predict(self, board):
        action_size = self.game.get_action_size()
        return [1 / action_size] * action_size, 0


# （ほぼ）完全読みでランダムプレイヤーに9割勝てるか
def test_mcts():
    game = TicTacToeGame(3)

    net = ConstantNet(game)
    mcts = MCTS(game, net, 0.1)
    mcts_player = MCTSPlayer(mcts, 100)

    random_player = RandomPlayer(game)

    arena = Arena(mcts_player.play, random_player.play, game)
    r = arena.play_games(20)

    assert r > 0.9
