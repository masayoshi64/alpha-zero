from app.alpha_zero.mcts import MCTS
from app.games.tictactoe import TicTacToeGame
from app.games.arena import Arena
from app.games.players import RandomPlayer, MCTSPlayer
from app.alpha_zero.models import ConstantModel


# （ほぼ）完全読みでランダムプレイヤーに9割勝てるか
def test_mcts():
    game = TicTacToeGame(3)

    net = ConstantModel(game)
    mcts = MCTS(game, net, 0.1, 1, 100)
    mcts_player = MCTSPlayer(mcts)

    random_player = RandomPlayer(game)

    arena = Arena(mcts_player, random_player, game)
    r = arena.play_games(20)
    arena = Arena(random_player, mcts_player, game)
    r += -arena.play_games(20)
    r /= 2

    assert r > 0.9
