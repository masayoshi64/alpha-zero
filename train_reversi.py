import logging
from argparse import ArgumentParser

import torch
import wandb

from app.alpha_zero.trainer import Trainer
from app.alpha_zero.models import ReversiModel
from app.games.reversi import ReversiGame


def main():
    parser = ArgumentParser()
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    use_wandb = args.use_wandb

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
    }
    if use_wandb:
        wandb.init(project="alpha-zero", config=config)
    game = ReversiGame(4)
    trainer = Trainer(game, **config, use_wandb=use_wandb)
    model = ReversiModel(game)
    model = trainer.train(model)
    torch.save(model, "models/reversi4_model.pt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
