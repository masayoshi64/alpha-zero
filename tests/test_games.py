from app.games.tictactoe import TicTacToeGame
from app.games.arena import Arena
from app.games.players import RandomPlayer


# ゲームが正しくプレイできているか
# コマンドで実行し正しくプレイできているかを確認する必要がある


def test_tictactoe():
    print("hello")
    game = TicTacToeGame(3)
    player1, player2 = RandomPlayer(game), RandomPlayer(game)
    arena = Arena(player1.play, player2.play, game)
    arena.play_game(verbose=1)
