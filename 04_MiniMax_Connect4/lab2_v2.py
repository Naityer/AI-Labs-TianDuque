# lab2_v2
import copy
import random

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
ROWS = 6
COLS = 7

class ConnectFour:
    def __init__(self):
        self.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current_player = PLAYER_X
    
    def print_board(self):
        for row in self.board:
            print('|'.join(row))
        print('-' * (COLS * 2 - 1))
        print(' '.join(str(i) for i in range(COLS)))

    def is_board_full(self, board):
        return all(board[0][col] != EMPTY for col in range(COLS))
    
    def is_valid_move(self, col):
        return self.board[0][col] == EMPTY

    def drop_piece(self, board, col, piece):
        for row in reversed(range(ROWS)):
            if board[row][col] == EMPTY:
                board[row][col] = piece
                return True
        return False
    
    def check_winner(self, board, piece):
        # Comprobar en horizontal
        for row in range(ROWS):
            for col in range(COLS - 3):
                if board[row][col] == piece and all(board[row][col + i] == piece for i in range(4)):
                    return True

        # Comprobar en vertical
        for row in range(ROWS - 3):
            for col in range(COLS):
                if board[row][col] == piece and all(board[row + i][col] == piece for i in range(4)):
                    return True

        # Comprobar en diagonal ↘ (de izquierda a derecha, de arriba a abajo)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if board[row][col] == piece and all(board[row + i][col + i] == piece for i in range(4)):
                    return True

        # Comprobar en diagonal ↗ (de izquierda a derecha, de abajo a arriba)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if board[row][col] == piece and all(board[row - i][col + i] == piece for i in range(4)):
                    return True

        return False

    
    def evaluate_window(self, window, piece):
        score = 0
        opponent = PLAYER_O if piece == PLAYER_X else PLAYER_X
        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 10
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 5
        if window.count(opponent) == 3 and window.count(EMPTY) == 1:
            score -= 80
        return score
    
    def evaluate_position(self, board, piece):
        score = 0
        for row in range(ROWS):
            for col in range(COLS - 3):
                score += self.evaluate_window([board[row][col + i] for i in range(4)], piece)
        for row in range(ROWS - 3):
            for col in range(COLS):
                score += self.evaluate_window([board[row + i][col] for i in range(4)], piece)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                score += self.evaluate_window([board[row + i][col + i] for i in range(4)], piece)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                score += self.evaluate_window([board[row - i][col + i] for i in range(4)], piece)
        return score
    
    def get_valid_moves(self, board):
        return [col for col in range(COLS) if self.is_valid_move(col)]
    
    def minimax(self, board, depth, maximizing_player, alpha, beta):
        valid_moves = self.get_valid_moves(board)
        is_terminal = not valid_moves or self.check_winner(board, PLAYER_X) or self.check_winner(board, PLAYER_O)
        if depth == 0 or is_terminal:
            if self.check_winner(board, PLAYER_X):
                return (None, 100000)
            elif self.check_winner(board, PLAYER_O):
                return (None, -100000)
            else:
                return (None, self.evaluate_position(board, PLAYER_X))
        
        if maximizing_player:
            value = -float('inf')
            best_move = random.choice(valid_moves)
            for col in valid_moves:
                temp_board = copy.deepcopy(board)
                self.drop_piece(temp_board, col, PLAYER_X)
                new_score = self.minimax(temp_board, depth - 1, False, alpha, beta)[1]
                if new_score > value:
                    value = new_score
                    best_move = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_move, value
        else:
            value = float('inf')
            best_move = random.choice(valid_moves)
            for col in valid_moves:
                temp_board = copy.deepcopy(board)
                self.drop_piece(temp_board, col, PLAYER_O)
                new_score = self.minimax(temp_board, depth - 1, True, alpha, beta)[1]
                if new_score < value:
                    value = new_score
                    best_move = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_move, value

def main():
    game = ConnectFour()
    user_piece = PLAYER_X if input("Do you want to play first? (y/n): ").strip().lower() == 'y' else PLAYER_O
    ai_piece = PLAYER_O if user_piece == PLAYER_X else PLAYER_X
    difficulty = int(input("Enter AI difficulty (1-5): "))  # Agregar selección de dificultad
    game.print_board()
    
    while True:
        if game.current_player == user_piece:
            col = int(input("Enter column (0-6): "))
            if game.is_valid_move(col):
                game.drop_piece(game.board, col, user_piece)
                game.current_player = ai_piece
        else:
            print("AI is thinking...")
            col, _ = game.minimax(copy.deepcopy(game.board), difficulty, True, -float('inf'), float('inf'))
            game.drop_piece(game.board, col, ai_piece)
            game.current_player = user_piece
        
        game.print_board()
        if game.check_winner(game.board, user_piece):
            print("You win!")
            break
        elif game.check_winner(game.board, ai_piece):
            print("AI wins!")
            break
        elif not game.get_valid_moves(game.board):
            print("It's a draw!")
            break


if __name__ == "__main__":
    main()
