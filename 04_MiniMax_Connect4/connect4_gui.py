import pygame
import sys
import copy
from lab2_v2 import ConnectFour, PLAYER_X, PLAYER_O

# Configuración
ROWS, COLS = 6, 7
SQUARESIZE = 80
MARGIN = 20  # Margen alrededor del tablero
WIDTH = COLS * SQUARESIZE + 2 * MARGIN
HEIGHT = (ROWS + 1) * SQUARESIZE + 2 * MARGIN
SIZE = (WIDTH, HEIGHT)

# Colores
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Opciones de dificultad y profundidad
DIFFICULTY_LEVELS = {
    "Fácil": 4,
    "Normal": 6,
    "Imposible": 8
}

class ConnectFourGUI:
    def __init__(self, game, user_piece, ai_piece, depth):
        pygame.init()
        self.game = game
        self.user_piece = user_piece
        self.ai_piece = ai_piece
        self.depth = depth
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption("Connect 4 - Minimalist GUI")
        self.font = pygame.font.SysFont("monospace", 40)
        self.running = True

        # Centramos el tablero en la ventana
        total_board_height = (ROWS + 1) * SQUARESIZE  
        total_board_width = COLS * SQUARESIZE  
        self.offset_x = (WIDTH - total_board_width) // 2  
        self.offset_y = (HEIGHT - total_board_height) // 2  

        self.draw_board()
        pygame.display.update()  # Asegurar que el tablero inicial se muestra

        # Si la IA juega primero, realizar su movimiento inicial
        if self.game.current_player == self.ai_piece:
            pygame.time.wait(500)  # Breve pausa para que el usuario lo vea
            self.ai_move()
            pygame.display.update()  # Actualizar después del movimiento de la IA
            pygame.event.clear()  # Limpiar eventos para evitar problemas de doble clic

    def draw_board(self):
        self.screen.fill(GRAY)

        # Dibujar las celdas del tablero
        for r in range(ROWS):
            for c in range(COLS):
                pygame.draw.rect(self.screen, BLACK, 
                                 (c * SQUARESIZE + self.offset_x, (r + 1) * SQUARESIZE + self.offset_y, SQUARESIZE, SQUARESIZE), 2)
                cell_value = self.game.board[r][c]
                if cell_value != ' ':
                    text = self.font.render(cell_value, True, BLACK)
                    self.screen.blit(text, (c * SQUARESIZE + self.offset_x + 30, (r + 1) * SQUARESIZE + self.offset_y + 20))

        pygame.display.update()  # Refrescar pantalla después de redibujar

    def ai_move(self):
        if self.running and self.game.current_player == self.ai_piece:
            pygame.time.wait(500)  # Simula el pensamiento de la IA
            col, _ = self.game.minimax(copy.deepcopy(self.game.board), self.depth, True, -float('inf'), float('inf'))
            self.game.drop_piece(self.game.board, col, self.ai_piece)

            self.draw_board()  # Redibujar tablero para actualizar la pantalla
            pygame.display.update()  # Asegurar que el movimiento de la IA se vea correctamente
            
            if self.game.check_winner(self.game.board, self.ai_piece):
                print("AI wins!")
                self.running = False
            else:
                self.game.current_player = self.user_piece

            pygame.event.clear()  # Limpiar eventos para evitar clics rezagados

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and self.game.current_player == self.user_piece:
                    col = (event.pos[0] - self.offset_x) // SQUARESIZE  
                    if self.game.is_valid_move(col):
                        self.game.drop_piece(self.game.board, col, self.user_piece)
                        self.draw_board()  # Actualizar pantalla tras el turno del jugador
                        pygame.display.update()  

                        if self.game.check_winner(self.game.board, self.user_piece):
                            print("You win!")
                            self.running = False
                        else:
                            self.game.current_player = self.ai_piece
                            pygame.time.wait(500)  # Espera antes de que la IA juegue
                            self.ai_move()
        pygame.quit()


def get_game_settings():
    """Preguntar al usuario quién juega primero y la dificultad."""
    print("¿Quién juega primero?")
    print("1 - Jugador")
    print("2 - IA")
    first_player = input("Selecciona (1 o 2): ").strip()

    user_piece = PLAYER_X if first_player == "1" else PLAYER_O
    ai_piece = PLAYER_O if user_piece == PLAYER_X else PLAYER_X

    print("\nSelecciona el nivel de dificultad:")
    for i, level in enumerate(DIFFICULTY_LEVELS.keys(), start=1):
        print(f"{i} - {level}")

    while True:
        difficulty_choice = input("Selecciona (1-3): ").strip()
        if difficulty_choice in ["1", "2", "3"]:
            difficulty = list(DIFFICULTY_LEVELS.keys())[int(difficulty_choice) - 1]
            depth = DIFFICULTY_LEVELS[difficulty]
            break
        else:
            print("Entrada inválida, intenta de nuevo.")

    return user_piece, ai_piece, depth


if __name__ == "__main__":
    user_piece, ai_piece, depth = get_game_settings()
    game = ConnectFour()
    
    # Si la IA inicia, actualizar antes del primer turno
    if user_piece != PLAYER_X:
        game.current_player = ai_piece
    else:
        game.current_player = user_piece

    gui = ConnectFourGUI(game, user_piece, ai_piece, depth)
    gui.run()
