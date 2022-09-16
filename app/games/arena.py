import logging
from typing import Callable

from .game import Game


class Arena:
    """ゲームを実行するクラス"""

    def __init__(self, player1: Callable, player2: Callable, game: Game):
        """
        Args:
            player1 (Callable): プレイヤー1のアクションを返す関数
            player2 (Callable): プレイヤー2のアクションを返す関数
            game (Game): ゲーム
        """
        self.game = game
        self.player1 = player1
        self.player2 = player2

    def play_game(self, verbose: int = 0) -> float:
        """ゲームを1回実行

        Args:
            verbose (int, optional): 0の場合何も出力しない，1の場合途中の盤面と結果を出力する. Defaults to 0.

        Returns:
            float: プレイヤー1から見た報酬
        """
        board = self.game.get_initial_board()
        cur_player = 1
        turn = 1
        players = {1: self.player1, -1: self.player2}
        while not self.game.get_game_ended(board, cur_player):
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
                f"Game finished: Turn {turn}, result {self.game.get_reward(board, 1)}"
            )
            print(*board, sep="\n")
        return self.game.get_reward(board, 1)

    def play_games(self, n: int) -> float:
        """ゲームをn回実行する

        Args:
            n (int): ゲーム数

        Returns:
            float: プレイヤー1から見た平均報酬
        """
        ave_r = 0
        for _ in range(n):
            r = self.play_game()
            ave_r += r / n
        return ave_r
