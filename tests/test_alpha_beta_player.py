from app.games.tictactoe import TicTacToeGame
from app.games.players import RandomPlayer, AlphaBetaPlayer
from app.alpha_zero.utils import eval_player


# 完全読みでランダムプレイヤーに9割勝てるか
def test_alpha_beta():
    game = TicTacToeGame(3)

    alpha_beta_player = AlphaBetaPlayer(game)
    random_player = RandomPlayer(game)

    r = eval_player(alpha_beta_player, random_player, game, 20)
    assert r > 0.9
