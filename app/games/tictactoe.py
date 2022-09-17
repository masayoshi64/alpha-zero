import copy

from .game import Game


class TicTacToeGame(Game):
    """n目並べ"""

    def __init__(self, n):
        """
        Args:
            n (int): 盤のサイズ
        """
        self.n = n

    def get_initial_board(self):
        board = [[0] * self.n for _ in range(self.n)]
        return board

    def get_next_state(self, board, player, action):
        x, y = action // self.n, action % self.n
        assert board[x][y] == 0
        next_board = copy.deepcopy(board)
        next_board[x][y] = player
        player *= -1
        return (next_board, player)

    def is_valid(self, board, player, action):
        x, y = action // self.n, action % self.n
        return board[x][y] == 0

    def judge(self, board):
        # 横
        for x in range(self.n):
            sm = 0
            for y in range(self.n):
                sm += board[x][y]
            if sm == self.n:
                return 1
            elif sm == -self.n:
                return -1

        # 縦
        for y in range(self.n):
            sm = 0
            for x in range(self.n):
                sm += board[x][y]
            if sm == self.n:
                return 1
            elif sm == -self.n:
                return -1

        # 斜め（左上to右下）
        sm = 0
        for x in range(self.n):
            sm += board[x][x]
        if sm == self.n:
            return 1
        elif sm == -self.n:
            return -1

        # 斜め（右上to左下）
        sm = 0
        for x in range(self.n):
            sm += board[x][self.n - x - 1]
        if sm == self.n:
            return 1
        elif sm == -self.n:
            return -1

        # 引き分けの判定
        is_ended = True
        for x in range(self.n):
            for y in range(self.n):
                if board[x][y] == 0:
                    is_ended = False

        return 0 if is_ended else 2

    def get_game_ended(self, board, player):
        result = self.judge(board)
        if result == 2:
            return False
        else:
            return True

    def get_reward(self, board, player):
        result = self.judge(board)
        assert result != 2
        if player == 1:
            return result
        else:
            return -result

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
