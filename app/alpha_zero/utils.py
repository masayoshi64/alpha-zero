from typing import List

import numpy as np

from ..games.arena import Arena
from ..games.player_base import Player
from ..games.game import Game


def eval_player(
    eval_player: Player, standard_player: Player, game: Game, num_game: int
) -> float:
    arena1 = Arena(eval_player, standard_player, game)
    arena2 = Arena(standard_player, eval_player, game)
    r = (arena1.play_games(num_game) - arena2.play_games(num_game)) / 2
    return r


def get_board_view(cboard: List[List[float]]):
    board = np.array(cboard)
    return [(board == 1).tolist(), (board == -1).tolist()]
