import logging

import torch
import wandb

from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import OneLayerModel
from app.games.tictactoe import TicTacToeGame


def main():
    config = {
        "num_iter": 20,
        "num_episode": 10,
        "num_epoch": 100,
        "num_game": 100,
        "lr": 0.01,
        "r_thresh": 0,
        "alpha": 0.1,
        "tau": 1.0,
        "num_search": 10,
    }
    wandb.init(project="alpha-zero", config=config)
    game = TicTacToeGame(3)
    trainer = Trainer(game, use_wandb=True, **config)
    model = OneLayerModel(game)
    model = trainer.train(model)
    torch.save(model, "models/model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
