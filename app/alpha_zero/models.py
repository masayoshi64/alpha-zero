from typing import Tuple
import torch.nn as nn
import torch

from ..games.game import Game


class ConstantModel(nn.Module):
    def __init__(self, game: Game):
        super().__init__()
        self.game = game

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_size = self.game.get_action_size()
        return torch.Tensor([[1 / action_size] * action_size]), torch.Tensor([[0]])


class OneLayerModel(nn.Module):
    def __init__(self, game: Game, **args):
        super().__init__()
        self.height = game.get_height()
        self.width = game.get_width()
        self.action_size = game.get_action_size()
        self.fc_p = nn.Linear(self.height * self.width, self.action_size)
        self.fc_v = nn.Linear(self.height * self.width, 1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, self.height * self.width)
        p = self.fc_p(x)
        p = self.softmax(p)
        v = self.fc_v(x)
        v = self.tanh(v)
        return p, v
