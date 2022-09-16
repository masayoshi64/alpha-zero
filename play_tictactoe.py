import argparse

from app.games.arena import Arena
from app.games.tictactoe import TicTacToeGame
from app.games.players import RandomPlayer, HumanPlayer, MCTSPlayer
from app.alpha_zero.nnet import ConstantNet
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
        net = ConstantNet(game)
        mcts = MCTS(game, net, 0.1)
        player2 = MCTSPlayer(mcts, 1000)
    else:
        raise ValueError(f"Invalid player type: {player_type}")

    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)


if __name__ == "__main__":
    main()
