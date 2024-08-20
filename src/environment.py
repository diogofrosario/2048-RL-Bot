import numpy as np
import random
import copy

from typing import Union

class Board:
    def __init__(self, board_size: int = 4, seed: int = None) -> None:
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # add initial values to start the game
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self) -> None:
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if self.board[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r][c] = 4 if random.random() > 0.9 else 2

    def clone(self):
        """Return a deep copy of the board."""
        new_board = Board(self.board_size)
        new_board.board = copy.deepcopy(self.board)
        return new_board

    def is_full(self):
        """Check if the board has no empty cells."""
        return not any(0 in row for row in self.board)

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.board)
    

class Movement:
    def __init__(self, board: Board):
        self.board = board
        self.score = 0

    def compress(self, row: list) -> list:
        new_row = [num for num in row if num != 0]
        new_row += [0] * (len(row) - len(new_row))
        return new_row

    def merge(self, row: list) -> list:
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                self.score += row[i]  # Add the merged value to the score
                row[i + 1] = 0
        return row

    def ensure_int(self):
        self.board.board = [[int(num) for num in row] for row in self.board.board]

    def move_left(self):
        self.score = 0  # Reset the score for the current move
        new_board = []
        for row in self.board.board:
            compressed_row = self.compress(row)
            merged_row = self.merge(compressed_row)
            final_row = self.compress(merged_row)
            new_board.append(final_row)
        self.board.board = new_board
        self.ensure_int()

    def move_right(self):
        self.score = 0  # Reset the score for the current move
        new_board = []
        for row in self.board.board:
            reversed_row = self.reverse(row)
            compressed_row = self.compress(reversed_row)
            merged_row = self.merge(compressed_row)
            final_row = self.reverse(self.compress(merged_row))
            new_board.append(final_row)
        self.board.board = new_board
        self.ensure_int()

    def move_up(self):
        self.score = 0  # Reset the score for the current move
        self.transpose()
        self.move_left()
        self.transpose()

    def move_down(self):
        self.score = 0  # Reset the score for the current move
        self.transpose()
        self.move_right()
        self.transpose()

    def reverse(self, row: list) -> list:
        """ Reverse the row """
        return row[::-1]

    def transpose(self):
        self.board.board = [list(row) for row in zip(*self.board.board)]

    def move(self, direction: int) -> Union[int, ValueError]:
        """Apply a move in the given direction (0: left, 1: right, 2: up, 3: down)."""
        if direction == 0:
            self.move_left()
        elif direction == 1:
            self.move_right()
        elif direction == 2:
            self.move_up()
        elif direction == 3:
            self.move_down()
        else:
            raise ValueError("Invalid direction")

        if not self.board.is_full():
            self.board.add_new_tile()

        return self.score  # Return the score obtained from this move



if __name__ == '__main__':
    board = Board()
    movement = Movement(board)

    print(board)

    # Simulate some moves
    print("left")
    movement.move_left()
    print(board)

    print("right")
    movement.move_right()
    print(board)

    print("up")
    movement.move_up()
    print(board)

    print("down")
    movement.move_down()
    print(board)
