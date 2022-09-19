from typing import List, Tuple, Hashable


class Game:
    """ゲームの抽象クラス"""

    def __init__(self):
        pass

    def get_initial_board(self) -> List[List[float]]:
        """初期盤面を返す

        Returns:
            List[List[float]]: 初期盤面
        """
        raise NotImplementedError()

    def get_next_state(
        self, board: List[List[float]], player: int, action: int
    ) -> Tuple[List[List[float]], int]:
        """次の盤面とプレイヤーを返す

        Args:
            board (List[List[float]]): 現在の盤面
            player (int): 現在のプレイヤー
            action (int): とった行動

        Returns:
            Tuple[List[List[float]], int]: 次の盤面とプレイヤー
        """
        raise NotImplementedError()

    def is_valid(self, board: List[List[float]], player: int, action: int) -> bool:
        """actionが合法手か判定

        Args:
            board (List[List[float]]): 現在の盤面
            player (int): 現在のプレイヤー
            action (int): 行動

        Returns:
            bool: actionが合法手であるときTrue
        """
        raise NotImplementedError()

    def get_game_ended(self, board: List[List[float]], player: int) -> bool:
        """ゲームの終了判定

        Args:
            board (List[List[float]]): 現在の盤面
            player (int): 現在のプレイヤー

        Returns:
            bool: ゲームが終了しているかどうか
        """
        raise NotImplementedError()

    def get_reward(self, board: List[List[float]], player: int) -> bool:
        """ゲーム終了時のplayerから見た報酬を計算

        Args:
            board (List[List[float]]): 現在の盤面
            player (int): プレイヤー

        Returns:
            bool: playerから見た現在の盤面における報酬
        """
        raise NotImplementedError()

    def get_canonical_form(
        self, board: List[List[float]], player: int
    ) -> List[List[float]]:
        """playerが1であるとき盤面を反転する

        Args:
            board (List[List[float]]): 盤面
            player (int): プレイヤー

        Returns:
            List[List[float]]: playerによらない盤面
        """
        raise NotImplementedError()

    def get_action_size(self) -> int:
        """非合法手を含めたactionの数を返す

        Returns:
            int: すべてのアクションの数
        """
        raise NotImplementedError()

    def hash(self, board: List[List[float]], player: int) -> Hashable:
        """boardとplayerの組みをハッシュ可能な状態に変換

        Args:
            board (List[List[float]]): 盤面
            player (int): プレイヤー

        Returns:
            Hashable: ハッシュ可能なクラス
        """
        raise NotImplementedError()

    def get_height(self) -> int:
        """盤の高さを返す

        Returns:
            int: 高さ
        """
        raise NotImplementedError()

    def get_width(self) -> int:
        """盤の幅を返す

        Returns:
            int: 幅
        """
        raise NotImplementedError()
