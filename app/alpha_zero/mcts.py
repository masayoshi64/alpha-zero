from math import sqrt
from typing import List

import torch
import torch.nn as nn

from ..games.game import Game


class MCTS:
    def __init__(
        self, game: Game, model: nn.Module, alpha: float, tau: float, num_search: int
    ):
        """
        Args:
            game (Game): ゲーム
            model (nn.Module): 盤面を受け取り(p, v)を返すモデル
            alpha (float): 探索の重み
            tau (float): 温度パラメータ
            num_search (int): シミュレーション回数
        """
        self.game = game
        self.visited = set()
        self.model = model
        self.Q = dict()
        self.N = dict()
        self.P = dict()
        self.alpha = alpha
        self.tau = tau
        self.num_search = num_search

    def search(self, board: List[List[float]], player: int) -> float:
        """MCTS探索

        Args:
            board (List[List[float]]): 盤面
            player (int): プレイヤー

        Returns:
            float: playerから見た盤面の評価値の推定値
        """

        # ゲームが終了したらplayerから見た報酬を返す
        if self.game.get_game_ended(board, player):
            return self.game.get_reward(board, player)

        s = self.game.hash(board, player)
        action_size = self.game.get_action_size()

        # これまで訪れたことのない状態なら(p, v)を計算して保存
        if s not in self.visited:
            self.visited.add(s)
            cboard = self.game.get_canonical_form(board, player)
            p, v = self.model(torch.Tensor(cboard))
            p = p.detach().numpy()[0].tolist()
            v = v.detach().numpy()[0, 0]
            self.P[s] = p
            self.Q[s] = [0] * action_size
            self.N[s] = [0] * action_size
            return v

        # UCBが最大となるactionを選択
        best_action = 0
        max_ucb = -float("inf")
        Ns = sum(self.N[s])

        for action in range(action_size):
            if self.game.is_valid(board, player, action):
                ucb = self.Q[s][action] + self.alpha * sqrt(Ns) / (
                    1 + self.N[s][action]
                )
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action

        # 次の状態を取得
        next_board, next_player = self.game.get_next_state(board, player, best_action)

        # 再帰的に探索する
        v = (
            self.search(next_board, next_player)
            if player == next_player
            else -self.search(next_board, next_player)
        )

        # Q, N を更新
        self.Q[s][best_action] = (
            self.Q[s][best_action] * self.N[s][best_action] + v
        ) / (self.N[s][best_action] + 1)
        self.N[s][best_action] += 1
        return v

    def get_action_prob(self, board: List[List[float]]) -> List[float]:
        """行動確率を取得

        Args:
            board (List[List[float]]): プレイヤー1から見た盤面

        Returns:
            List[float]: プレイヤー1の行動確率
        """

        # 探索をnum_search回行う
        for i in range(self.num_search):
            self.search(board, 1)
        s = self.game.hash(board, 1)

        # 行動確率を計算
        p = []
        for a in range(self.game.get_action_size()):
            p.append(self.N[s][a] ** (1 / self.tau))

        # 正規化
        sm = sum(p)
        for a in range(len(p)):
            p[a] /= sm

        return p

    def reset(self) -> None:
        self.visited = set()
        self.Q = dict()
        self.N = dict()
        self.P = dict()
