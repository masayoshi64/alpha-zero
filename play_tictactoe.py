from games.arena import Arena
from games.tictactoe import TicTacToeGame
from games.players import RandomPlayer, HumanPlayer


def main():
    game = TicTacToeGame(3)
    player1 = HumanPlayer(game)
    player2 = RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)


if __name__ == "__main__":
    main()
