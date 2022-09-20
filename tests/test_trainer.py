from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import OneLayerModel, ConstantModel
from app.games.tictactoe import TicTacToeGame


def test_trainer():
    game = TicTacToeGame(3)
    trainer = Trainer(game, 20, 10, 10, 50, 0.01, 0, 0.1, 1.0, 10)
    model = OneLayerModel(game)
    model = trainer.train(model)
    cmodel = ConstantModel(game)
    r = trainer.eval(model, cmodel)
    assert r > 0.5
