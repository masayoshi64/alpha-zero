import copy
from typing import List, Tuple

from .game import Game


class ReversiGame(Game):
    """オセロ"""

    def __init__(self, n):
        """
        Args:
            n (int): 盤のサイズ
        """
        assert n % 2 == 0
        self.n = n
        self.dirs = [
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 0),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ]

    def get_initial_board(self):
        board = [[0] * self.n for _ in range(self.n)]
        board[self.n // 2][self.n // 2] = -1
        board[self.n // 2][self.n // 2 - 1] = 1
        board[self.n // 2 - 1][self.n // 2] = 1
        board[self.n // 2 - 1][self.n // 2 - 1] = -1
        return board

    def get_next_state(self, board, player, action):
        x, y = action // self.n, action % self.n
        flip_stones = self.get_flip_stones(board, player, action)

        next_board = copy.deepcopy(board)
        next_board[x][y] = player
        for fx, fy in flip_stones:
            next_board[fx][fy] = player

        # -playerがパスならもう一度playerの番
        if not self.is_pass(board, -player):
            player *= -1
        return (next_board, player)

    def is_valid(self, board, player, action):
        # 返せる石がない場所には置けない
        flip_stones = self.get_flip_stones(board, player, action)
        return len(flip_stones) > 0

    def get_game_ended(self, board, player):
        # 本来は-playerに対してもpassか判定する必要があるが，
        # playerがget_next_actionで得られたものなら
        # playerがpassなら-playerはパスであったはずなので不要
        return self.is_pass(board, player)  # and self.is_pass(board, -player)

    def get_reward(self, board, player):
        diff = 0
        for x in range(self.n):
            for y in range(self.n):
                if board[x][y] == player:
                    diff += 1
                elif board[x][y] == -player:
                    diff -= 1

        # 半分より多くとっていれば+1,半分より少なければ-1,ちょうど半分なら0
        if diff > 0:
            return 1
        elif diff < 0:
            return -1
        else:
            return 0

    def get_canonical_form(self, board, player):
        b = copy.deepcopy(board)
        if player == 1:
            return b
        else:
            for x in range(self.n):
                for y in range(self.n):
                    b[x][y] *= -1
            return b

    def get_action_size(self):
        return self.n * self.n

    def hash(self, board, player):
        return (str(board), player)

    def get_height(self) -> int:
        return self.n

    def get_width(self) -> int:
        return self.n

    def get_flip_stones(
        self, board: List[List[float]], player: int, action: int
    ) -> List[Tuple[int, int]]:
        """返される石の場所のリストを返す

        Args:
            board (List[List[int]]): 盤面
            player (int): プレイヤー
            action (int): アクション

        Returns:
            List[Tuple[int, int]]: ひっくり返る相手の石の場所のリスト
        """

        x, y = action // self.n, action % self.n

        # 空きマス出ないなら置けない
        if board[x][y] != 0:
            return []

        flip_stones = []
        for dx, dy in self.dirs:
            nx, ny = x + dx, y + dy
            tmp = []

            # 相手のコマがある限り進む
            while self.on_board(nx, ny) and board[nx][ny] == -player:
                tmp.append((nx, ny))
                nx += dx
                ny += dy

            # 自分のコマに当たればひっくり返せる
            if self.on_board(nx, ny) and board[nx][ny] == player:
                flip_stones.extend(tmp)

        return flip_stones

    def on_board(self, x: int, y: int) -> bool:
        """盤内にあるかのutil関数

        Args:
            x (int): 縦
            y (int): 横

        Returns:
            bool: 盤内にあればTrueなければFalse
        """

        return (0 <= x < self.n) and (0 <= y < self.n)

    def is_pass(self, board: List[List[float]], player: int) -> bool:
        """パスか判定

        Args:
            board (List[List[int]]): 盤面
            player (int): プレイヤー

        Returns:
            bool: パスならばTrue, おける場所があるならFalse
        """

        for x in range(self.n):
            for y in range(self.n):
                action = x * self.n + y
                flip_stones = self.get_flip_stones(board, player, action)
                if len(flip_stones) > 0:
                    return False
        return True
