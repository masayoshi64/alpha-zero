from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import OneLayerModel
from app.alpha_zero.mcts import MCTS
from app.games.arena import Arena
from app.games.tictactoe import TicTacToeGame
from app.games.players import RandomPlayer, MCTSPlayer


def test_trainer():
    game = TicTacToeGame(3)
    trainer = Trainer(game, 20, 10, 10, 50, 0, 0.01, 0.1, 1.0, 10)
    model = OneLayerModel(game)
    model = trainer.train(model)
    mcts = MCTS(game, model, 0.1, 1, 10)
    player1 = MCTSPlayer(mcts)
    player2 = RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    r = arena.play_games(50)
    win_rate = (r + 1) / 2
    # 勝率9割以上
    assert win_rate > 0.9
