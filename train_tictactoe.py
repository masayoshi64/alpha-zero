import logging

import torch
import wandb

from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import TicTacToeModel
from app.games.tictactoe import TicTacToeGame


def main():
    config = {
        "num_iter": 20,
        "buffer_size": 30000,
        "num_episode": 500,
        "num_epoch": 100,
        "num_game": 100,
        "batch_size": 100,
        "lr": 0.002,
        "r_thresh": 0,
        "alpha": 0.1,
        "tau": 1.0,
        "num_search": 10,
        "use_wandb": False,
    }
    if config["use_wandb"]:
        wandb.init(project="alpha-zero", config=config)
    game = TicTacToeGame(3)
    trainer = Trainer(game, **config)
    model = TicTacToeModel(game)
    model = trainer.train(model)
    torch.save(model, "models/model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
