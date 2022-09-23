import argparse

import torch

from app.games.arena import Arena
from app.games.tictactoe import TicTacToeGame
from app.games.reversi import ReversiGame
from app.games.players import (
    RandomPlayer,
    HumanPlayer,
    MCTSPlayer,
    AlphaBetaPlayer,
    NeuralNetPlayer,
)
from app.alpha_zero.mcts import MCTS
from app.alpha_zero.models import ConstantModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game")
    parser.add_argument("type")
    parser.add_argument("--white", action="store_true")
    args = parser.parse_args()
    game_type = args.game
    player_type = args.type
    play_white = args.white

    game_dict = {"tictactoe": TicTacToeGame(3), "reversi4": ReversiGame(4)}
    game = game_dict[game_type]
    player1 = HumanPlayer(game)
    if player_type == "random":
        player2 = RandomPlayer(game)
    elif player_type == "mcts":
        model = ConstantModel(game)
        mcts = MCTS(game, model, 0.1, 0, 50)
        player2 = MCTSPlayer(mcts)
    elif player_type == "nnet":
        model = torch.load(f"models/model_{game_type}.pt")
        player2 = NeuralNetPlayer(model)
    elif player_type == "alphabeta":
        player2 = AlphaBetaPlayer(game)
    else:
        raise ValueError(f"Invalid player type: {player_type}")

    # 白番をプレイする場合スワップ
    if play_white:
        player1, player2 = player2, player1

    arena = Arena(player1, player2, game)
    arena.play_game(verbose=1)


if __name__ == "__main__":
    main()
