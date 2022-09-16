from app.alpha_zero.mcts import MCTS
from app.games.tictactoe import TicTacToeGame
from app.games.arena import Arena
from app.games.players import RandomPlayer, MCTSPlayer
from app.alpha_zero.nnet import ConstantNet


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
