from ..games.game import Game


class ConstantNet:
    def __init__(self, game: Game):
        self.game = game

    def predict(self, board):
        action_size = self.game.get_action_size()
        return [1 / action_size] * action_size, 0
