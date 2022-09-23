from app.alpha_zero.mcts import MCTS
from app.games.tictactoe import TicTacToeGame
from app.games.players import RandomPlayer, MCTSPlayer
from app.alpha_zero.models import ConstantModel
from app.alpha_zero.utils import eval_player


# （ほぼ）完全読みでランダムプレイヤーに9割勝てるか
def test_mcts():
    game = TicTacToeGame(3)

    net = ConstantModel(game)
    mcts = MCTS(game, net, 0.1, 1, 100)
    mcts_player = MCTSPlayer(mcts)

    random_player = RandomPlayer(game)

    r = eval_player(mcts_player, random_player, game, 20)
    assert r > 0.9
