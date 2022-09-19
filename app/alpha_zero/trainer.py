import copy
import logging
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from ..games.arena import Arena
from ..games.game import Game
from ..games.players import RandomPlayer
from .mcts import MCTS


class AlphaZeroDataset(Dataset):
    def __init__(self, experiences: List[Tuple[List[List[float]], List[float], float]]):
        """
        Args:
            experiences (List[Tuple[List[List[float]], List[float], float]]): (cboard, p, v)
        """
        self.boards, self.p, self.v = zip(*experiences)

    def __getitem__(self, index):
        return torch.Tensor(self.boards[index]), (
            torch.Tensor(self.p[index]),
            self.v[index],
        )

    def __len__(self):
        return len(self.boards)


class Trainer:
    def __init__(
        self,
        game: Game,
        num_iter: int,
        num_episode: int,
        num_epoch: int,
        num_game: int,
        lr: float,
        r_thresh: float,
        alpha: float,
        tau: float,
        num_search: int,
        use_wandb: float = False,
    ):
        """
        Args:
            game (Game): ゲーム
            num_iter (int): モデル更新回数
            num_episode (int): 自己対戦の回数
            num_epoch (int): 学習時にエポック数
            num_game (int): 勝率計算時の対戦数
            lr (float): 学習率
            r_thresh (float): モデル更新の際の平均報酬の閾値
            alpha (float): MCTSのalpha
            tau (float): MCTSのtau
            num_search (int): MCTSのnum_search
            use_wandb (float, optional): wandbを使うならTrue
        """
        self.game = game
        self.num_iter = num_iter
        self.num_episode = num_episode
        self.num_epoch = num_epoch
        self.num_game = num_game
        self.lr = lr
        self.r_thresh = r_thresh
        self.alpha = alpha
        self.tau = tau
        self.num_search = num_search
        self.use_wandb = use_wandb

    def play_episode(
        self, model: nn.Module
    ) -> List[Tuple[List[List[float]], List[float], float]]:
        """ゲームを1回プレイしその履歴を返す

        Args:
            model (nn.Module): boardを受け取り(p, v)を返すモデル

        Returns:
            List[Tuple[List[List[float]], List[float], float]]: (cboard, p, v)
        """
        mcts = MCTS(self.game, model, self.alpha, self.tau, self.num_search)
        board = self.game.get_initial_board()
        player = 1
        experience = []
        while not self.game.get_game_ended(board, player):
            cboard = self.game.get_canonical_form(board, player)
            p = mcts.get_action_prob(cboard)

            # 学習用にはカノニカルな表現を用いる
            # 報酬は途中解らないのでとりあえずplayerを入れておく
            experience.append([cboard, p, player])

            action = np.random.choice(len(p), p=p)
            board, player = self.game.get_next_state(board, player, action)

        # 報酬を入れ直す
        v = self.game.get_reward(board, 1)
        for i in range(len(experience)):
            # playerから見た報酬を入れる
            experience[i][2] = experience[i][2] * v

        return experience

    def train(self, model_: nn.Module) -> nn.Module:
        """モデルのトレーニング

        Args:
            model_ (nn.Module): 初期モデル

        Returns:
            nn.Module: 学習済みモデル
        """
        model = copy.deepcopy(model_)
        for i in tqdm(range(self.num_iter)):
            experiences = []
            # 自己対戦を行う
            for episode in range(self.num_episode):
                experiences += self.play_episode(model)

            # new_modelに対して学習を行う
            new_model = copy.deepcopy(model)
            optimizer = torch.optim.SGD(new_model.parameters(), self.lr)
            dataset = AlphaZeroDataset(experiences)
            dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
            for epoch in range(self.num_epoch):
                epoch_loss = 0
                for x, (p, v) in dataloader:
                    p_pred, v_pred = new_model(x)
                    loss = self.loss_p(p, p_pred) + self.loss_v(v, v_pred)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_loss += loss.item()

                logging.info(f"training loss: {epoch_loss}")
                if self.use_wandb:
                    wandb.log({"loss": epoch_loss})

            # modelと対戦時のnew_modelの平均報酬を計算
            pmcts = MCTS(self.game, model, self.alpha, self.tau, self.num_search)
            nmcts = MCTS(self.game, new_model, self.alpha, self.tau, self.num_search)
            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x)),
                lambda x: np.argmax(nmcts.get_action_prob(x)),
                self.game,
            )
            r = arena.play_games(self.num_game)
            logging.info(f"average reward: {r}")

            # 平均報酬がr_threshより高ければモデルを更新
            if r > self.r_thresh:
                logging.info("model updated")
                model = new_model

            # ランダムモデルと対戦させ評価
            if self.use_wandb:
                random_player = RandomPlayer(self.game)
                mcts = MCTS(self.game, model, self.alpha, self.tau, self.num_search)
                arena = Arena(
                    lambda x: np.argmax(mcts.get_action_prob(x)),
                    random_player.play,
                    self.game,
                )
                r = arena.play_games(self.num_game)
                wandb.log({"ave_reward": r})

        return model

    def loss_p(self, p: torch.Tensor, p_pred: torch.Tensor) -> torch.Tensor:
        """行動確率pに関する損失

        Args:
            p (torch.Tensor): MCTSによる行動確率
            p_pred (torch.Tensor): 予測行動確率

        Returns:
            torch.Tensor: クロスエントロピー
        """
        return torch.sum(p * torch.log(p_pred)) / p.size()[0]

    def loss_v(self, v: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """評価値vに関する損失

        Args:
            v (torch.Tensor): 報酬
            v_pred (torch.Tensor): 予測評価値

        Returns:
            torch.Tensor: 2乗誤差
        """
        return torch.sum((v - v_pred) ** 2) / v.size()[0]
