import logging

import torch
import wandb

from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import ReversiModel
from app.games.reversi import ReversiGame


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
        "use_wandb": True,
    }
    if config["use_wandb"]:
        wandb.init(project="alpha-zero", config=config)
    game = ReversiGame(4)
    trainer = Trainer(game, **config)
    model = ReversiModel(game)
    model = trainer.train(model)
    torch.save(model, "models/reversi4_model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
