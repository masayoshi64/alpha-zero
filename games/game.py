class Game:
    def __init__(self):
        pass

    def get_initial_board(self):
        raise NotImplementedError()

    def get_next_state(self, board, player, action):
        raise NotImplementedError()

    def is_valid(self, board, player, action):
        raise NotImplementedError()

    def get_game_ended(self, board, player):
        raise NotImplementedError()

    def get_canonical_form(self, board, player):
        raise NotImplementedError()

    def get_action_size(self):
        raise NotImplementedError()
