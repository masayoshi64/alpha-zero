from ..games.arena import Arena
from ..games.players import Player
from ..games.game import Game


def eval_player(
    eval_player: Player, standard_player: Player, game: Game, num_game: int
) -> float:
    arena1 = Arena(eval_player, standard_player, game)
    arena2 = Arena(standard_player, eval_player, game)
    r = (arena1.play_games(num_game) - arena2.play_games(num_game)) / 2
    return r
