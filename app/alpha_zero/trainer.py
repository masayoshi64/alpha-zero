import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..games.arena import Arena
from ..games.game import Game
from .mcts import MCTS


class AlphaZeroDataset(Dataset):
    def __init__(self, experiences):
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
        num_iter,
        num_episode,
        num_epoch,
        num_game,
        lr,
        r_thresh,
        alpha,
        tau,
        num_search,
    ):
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

    def play_episode(self, model):
        mcts = MCTS(self.game, model, self.alpha, self.tau, self.num_search)
        board = self.game.get_initial_board()
        player = 1
        experience = []
        while not self.game.get_game_ended(board, player):
            cboard = self.game.get_canonical_form(board, player)
            p = mcts.get_action_prob(cboard)
            experience.append([cboard, p, player])
            action = np.random.choice(len(p), p=p)
            board, player = self.game.get_next_state(board, player, action)
        v = self.game.get_reward(board, 1)
        for i in range(len(experience)):
            experience[i][2] = experience[i][2] * v
        return experience

    def train(self, model_: nn.Module) -> nn.Module:
        model = copy.deepcopy(model_)
        for i in tqdm(range(self.num_iter)):
            experiences = []
            for episode in range(self.num_episode):
                experiences += self.play_episode(model)

            new_model = copy.deepcopy(model)
            optimizer = torch.optim.SGD(new_model.parameters(), self.lr)
            dataset = AlphaZeroDataset(experiences)
            dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
            for epoch in range(self.num_epoch):
                for x, (p, v) in dataloader:
                    p_pred, v_pred = new_model(x)
                    loss = self.loss_p(p, p_pred) + self.loss_v(v, v_pred)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            pmcts = MCTS(self.game, model, self.alpha, self.tau, self.num_search)
            nmcts = MCTS(self.game, new_model, self.alpha, self.tau, self.num_search)
            arena = Arena(
                lambda x: np.argmax(pmcts.get_action_prob(x)),
                lambda x: np.argmax(nmcts.get_action_prob(x)),
                self.game,
            )
            r = arena.play_games(self.num_game)
            logging.info(f"average reward: {r}")
            if r > self.r_thresh:
                logging.info("model updated")
                model = new_model
        return model

    def loss_p(self, p, p_pred):
        return torch.sum(p * torch.log(p_pred))  # / p.size()[0]

    def loss_v(self, v, v_pred):
        return torch.sum((v - v_pred) ** 2) / v.size()[0]
