from games.tictactoe import TicTacToeGame
from games.arena import Arena
from games.players import RandomPlayer


def test_tictactoe():
    print("hello")
    game = TicTacToeGame(3)
    player1, player2 = RandomPlayer(game), RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)
