from app.games.tictactoe import TicTacToeGame
from app.games.reversi import ReversiGame
from app.games.arena import Arena
from app.games.players import RandomPlayer


# ゲームが正しくプレイできているか
# コマンドで実行し正しくプレイできているかを確認する必要がある


def test_tictactoe():
    game = TicTacToeGame(3)
    player1, player2 = RandomPlayer(game), RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)


def test_reversi():
    game = ReversiGame(6)
    player1, player2 = RandomPlayer(game), RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)
