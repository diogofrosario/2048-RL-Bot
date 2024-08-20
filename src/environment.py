import numpy as np
import random

class Board:
    def __init__(self, board_size: int = 4, seed: int = None):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # add initial values to start the game
        self.add_new_tile()
        self.add_new_tile()

    def add_new_tile(self):
        empty_cells = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if self.board[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r][c] = 4 if random.random() > 0.9 else 2

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.board)
    

class Movement:
    def __init__(self, board: Board):
        self.board = board

    def compress(self, row):
        """ Compress the row, sliding all non-zero elements to the left """
        new_row = [num for num in row if num != 0]
        new_row += [0] * (len(row) - len(new_row))
        return new_row

    def merge(self, row):
        """ Merge the row by summing the adjacent cells """
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
        return row
    
    def ensure_int(self):
        """ Ensure all elements in the board are of type int to avoid numpy.int(4) showing up in the board instead of 4 """ 
        self.board.board = [[int(num) for num in row] for row in self.board.board]

    def move_left(self):
        new_board = []
        for row in self.board.board:
            compressed_row = self.compress(row)
            merged_row = self.merge(compressed_row)
            final_row = self.compress(merged_row)
            new_board.append(final_row)
        self.board.board = new_board
        self.ensure_int()
        self.board.add_new_tile()

    def reverse(self, row):
        """ Reverse the row """
        return row[::-1]

    def transpose(self):
        """ Transpose the board (swap rows with columns) """
        self.board.board = [list(row) for row in zip(*self.board.board)]

    def move_right(self):
        new_board = []
        for row in self.board.board:
            reversed_row = self.reverse(row)
            compressed_row = self.compress(reversed_row)
            merged_row = self.merge(compressed_row)
            final_row = self.reverse(self.compress(merged_row))
            new_board.append(final_row)
        self.board.board = new_board
        self.ensure_int()
        self.board.add_new_tile()

    def move_up(self):
        self.transpose()
        self.move_left()
        self.ensure_int()
        self.transpose()

    def move_down(self):
        self.transpose()
        self.move_right()
        self.ensure_int()
        self.transpose()


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
