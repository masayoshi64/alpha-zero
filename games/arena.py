import logging

from .game import Game


class Arena:
    def __init__(self, player1, player2, game: Game):
        self.game = game
        self.player1 = player1
        self.player2 = player2

    def play_game(self, verbose=0):
        board = self.game.get_initial_board()
        cur_player = 1
        turn = 1
        players = {1: self.player1, -1: self.player2}
        while self.game.get_game_ended(board, cur_player) == 2:
            if verbose:
                print(f"Turn {turn}, player {cur_player}")
                print(*board, sep="\n")
            action = players[cur_player](
                self.game.get_canonical_form(board, cur_player)
            )
            # 合法種でなければエラー
            if not self.game.is_valid(board, cur_player, action):
                logging.error(f"Action {action} is not valid")
                assert self.game.is_valid(board, cur_player, action)

            board, cur_player = self.game.get_next_state(board, cur_player, action)
            turn += 1
        if verbose:
            print(
                f"Game finished: Turn {turn}, result {self.game.get_game_ended(board, cur_player)}"
            )
            print(*board, sep="\n")
