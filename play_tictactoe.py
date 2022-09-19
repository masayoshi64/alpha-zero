import argparse

import torch

from app.games.arena import Arena
from app.games.tictactoe import TicTacToeGame
from app.games.players import RandomPlayer, HumanPlayer, MCTSPlayer
from app.alpha_zero.mcts import MCTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("type")
    args = parser.parse_args()
    player_type = args.type

    game = TicTacToeGame(3)
    player1 = HumanPlayer(game)
    if player_type == "random":
        player2 = RandomPlayer(game)
    elif player_type == "mcts":
        model = torch.load("models/model.pt")
        mcts = MCTS(game, model, 0.1, 1, 10)
        player2 = MCTSPlayer(mcts)
    else:
        raise ValueError(f"Invalid player type: {player_type}")

    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)


if __name__ == "__main__":
    main()
