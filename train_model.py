import logging

import torch

from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import OneLayerModel
from app.games.tictactoe import TicTacToeGame


def main():
    game = TicTacToeGame(3)
    trainer = Trainer(game, 20, 10, 100, 100, 0, 0.01, 0.1, 1.0, 10)
    model = OneLayerModel(game)
    model = trainer.train(model)
    torch.save(model, "models/model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
