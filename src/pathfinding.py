from environment import Movement, Board

class MinimaxSolver:
    def __init__(self, depth=3) -> None:
        self.depth = depth

    def evaluate(self, score: int) -> int:
        """Evaluate the board based on the score obtained during the move."""
        return score

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple:
        if depth == 0 or board.is_full():
            return self.evaluate(0), None

        best_move = None
        movement = Movement(board)

        if maximizing_player:
            max_eval = float('-inf')
            for move in range(4):
                new_board = board.clone()
                movement.board = new_board
                move_score = movement.move(move)

                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False)
                eval += move_score  # Add the move score to the evaluation

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in range(4):
                new_board = board.clone()
                movement.board = new_board
                move_score = movement.move(move)

                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True)
                eval += move_score  # Add the move score to the evaluation

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def find_best_move(self, board: Board) -> int:
        _, best_move = self.minimax(board, self.depth, float('-inf'), float('inf'), True)
        return best_move


if __name__ == '__main__':
    board = Board()
    movement = Movement(board)
    solver = MinimaxSolver(depth=10)

    print(board)

    # Find and apply the best move
    best_move = solver.find_best_move(board)
    if best_move is not None:
        movement.move(best_move)

    print(board)