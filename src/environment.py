import numpy as np
import random
from typing import Tuple

class Game2048:
    def __init__(self, size: int = 4):
        """
        Initialize the 2048 game environment.

        :param size: The size of the game board (size x size).
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.action_space = [0, 1, 2, 3]
        self.reset() # Set initial state of the game board

    def reset(self) -> np.ndarray:
        """
        Reset the game board to the initial state.

        :return: The initial state of the game board.
        """
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        self.score = 0
        return self.board

    def add_new_tile(self) -> None:
        """
        Add a new tile (2 or 4) to a random empty spot on the board.
        """
        empty_cells = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r][c] = 4 if random.random() > 0.9 else 2

    def is_game_over(self) -> bool:
        """
        Check if the game is over (no moves possible).

        :return: True if the game is over, False otherwise.
        """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return False
                if i < self.size - 1 and self.board[i][j] == self.board[i + 1][j]:
                    return False
                if j < self.size - 1 and self.board[i][j] == self.board[i][j + 1]:
                    return False
        return True

    def get_state(self) -> np.ndarray:
        """
        Get the current state of the game board.

        :return: The current game board state.
        """
        return self.board

    def render(self) -> None:
        """
        Print the current game board.
        """
        print('\n'.join(str(row) for row in self.board))

    def step(self, action: int) -> Tuple[np.ndarray, int, bool]:
        """
        Apply an action to the game board.

        :param action: The action to take (0: left, 1: right, 2: up, 3: down).
        :param movement: The movement object to apply the move logic.
        :return: A tuple containing the new board state, the reward from the action, and a boolean indicating if the game is over.
        """
        self.movement = Movement(self)
        initial_board = self.board.copy()
        if action == 0:
            self.movement.move_left()
        elif action == 1:
            self.movement.move_right()
        elif action == 2:
            self.movement.move_up()
        elif action == 3:
            self.movement.move_down()

        done = self.is_game_over()
        reward = self.score if not np.array_equal(initial_board, self.board) else 0

        if done:
            reward -= 1000  # Penalty for losing the game

        return self.board, reward, done, False, {}
    

class Movement:
    def __init__(self, board: Game2048):
        """
        Initialize the movement logic for the 2048 game.

        :param board: The game board object to interact with.
        """
        self.board = board

    def compress(self, row: np.ndarray) -> np.ndarray:
        """
        Compress the row by sliding all non-zero elements to the left.

        :param row: The row to compress.
        :return: The compressed row.
        """
        new_row = [num for num in row if num != 0]
        new_row += [0] * (len(row) - len(new_row))
        return np.array(new_row)

    def merge(self, row: np.ndarray) -> np.ndarray:
        """
        Merge adjacent cells in the row if they have the same value.

        :param row: The row to merge.
        :return: The row after merging.
        """
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                self.board.score += row[i]
                row[i + 1] = 0
        return row

    def reverse(self, row: np.ndarray) -> np.ndarray:
        """
        Reverse the elements in a row.

        :param row: The row to reverse.
        :return: The reversed row.
        """
        return row[::-1]

    def transpose(self) -> None:
        """
        Transpose the game board (swap rows with columns).
        """
        self.board.board = self.board.board.T

    def move_left(self) -> None:
        """
        Move all tiles on the board to the left.
        """
        for i in range(self.board.size):
            self.board.board[i] = self.compress(self.board.board[i])
            self.board.board[i] = self.merge(self.board.board[i])
            self.board.board[i] = self.compress(self.board.board[i])
        self.board.add_new_tile()

    def move_right(self) -> None:
        """
        Move all tiles on the board to the right.
        """
        for i in range(self.board.size):
            self.board.board[i] = self.reverse(self.board.board[i])
            self.board.board[i] = self.compress(self.board.board[i])
            self.board.board[i] = self.merge(self.board.board[i])
            self.board.board[i] = self.compress(self.board.board[i])
            self.board.board[i] = self.reverse(self.board.board[i])
        self.board.add_new_tile()

    def move_up(self) -> None:
        """
        Move all tiles on the board up.
        """
        self.transpose()
        self.move_left()
        self.transpose()

    def move_down(self) -> None:
        """
        Move all tiles on the board down.
        """
        self.transpose()
        self.move_right()
        self.transpose()
