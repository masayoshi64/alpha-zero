from typing import List


class Player:
    def __init__(self):
        pass

    def play(self, board: List[List[float]]) -> int:
        raise NotImplementedError()

    def reset(self) -> None:
        return
