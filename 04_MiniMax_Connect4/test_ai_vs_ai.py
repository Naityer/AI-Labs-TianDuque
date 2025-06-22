import copy
import random
from lab2_v2 import ConnectFour, PLAYER_X, PLAYER_O

def simulate_game(difficulty, starting_player):
    # Crear el juego
    game = ConnectFour()
    user_piece = starting_player
    ai_piece = PLAYER_O if user_piece == PLAYER_X else PLAYER_X

    # Guardar las columnas en las que se colocan las fichas
    moves = {'Player X': [], 'Player O': []}

    while True:
        # Jugada del jugador
        if game.current_player == user_piece:
            col, _ = game.minimax(copy.deepcopy(game.board), difficulty, True, -float('inf'), float('inf'))
            game.drop_piece(game.board, col, user_piece)
            moves['Player X'].append(col)  # Guardar la columna de la jugada
            game.current_player = ai_piece
        
        # Jugada de la IA
        elif game.current_player == ai_piece:
            col, _ = game.minimax(copy.deepcopy(game.board), difficulty, False, -float('inf'), float('inf'))
            
            # Agregar aleatorización en caso de empate en la evaluación
            valid_moves = game.get_valid_moves(game.board)
            best_moves = []
            best_score = -float('inf')
            
            # Evaluar las jugadas válidas y seleccionar las de la mejor puntuación
            for move in valid_moves:
                new_board = copy.deepcopy(game.board)
                game.drop_piece(new_board, move, ai_piece)
                score = game.evaluate_position(new_board, ai_piece)
                
                if score > best_score:
                    best_moves = [move]
                    best_score = score
                elif score == best_score:
                    best_moves.append(move)  # Agregar jugadas con la misma puntuación

            # Seleccionar una jugada aleatoria entre las mejores jugadas
            col = random.choice(best_moves)
            
            game.drop_piece(game.board, col, ai_piece)
            moves['Player O'].append(col)  # Guardar la columna de la jugada
            game.current_player = user_piece
        
        # Comprobar ganador
        if game.check_winner(game.board, user_piece):
            return "Player X wins", moves  # Cambié el texto para que coincida con el diccionario
        elif game.check_winner(game.board, ai_piece):
            return "Player O wins", moves  # Cambié el texto para que coincida con el diccionario
        elif not game.get_valid_moves(game.board):
            return "Draw", moves  # Cambié el texto para que coincida con el diccionario


def run_multiple_simulations(num_simulations, difficulty):
    results = {"Player X wins": 0, "Player O wins": 0, "Draw": 0}
    starting_info = []
    moves_info = []  # Lista para guardar las jugadas

    # Abrir el archivo log.txt para escribir
    with open('log.txt', 'w') as log_file:
        for i in range(num_simulations):
            starting_player = PLAYER_X if i % 2 == 0 else PLAYER_O  # Alternar quien empieza
            result, moves = simulate_game(difficulty, starting_player)
            results[result] += 1  # Ahora el resultado coincide con las claves del diccionario

            # Guardar información sobre quién empezó cada partida
            starting_info.append(f"Simulation {i + 1}: Player {starting_player} started")

            # Guardar las jugadas
            moves_info.append(moves)

            # Escribir progreso en el archivo
            log_file.write(f"Simulation {i + 1} progress: {i + 1}/{num_simulations} completed\n")

        # Escribir la información de quién empezó
        log_file.write("\nStarting player information:\n")
        for info in starting_info:
            log_file.write(f"{info}\n")

        # Escribir la información de las jugadas
        log_file.write("\nMoves Information:\n")
        for i, moves in enumerate(moves_info):
            log_file.write(f"Simulation {i + 1}:\n")
            log_file.write(f"  Player X moves: {moves['Player X']}\n")
            log_file.write(f"  Player O moves: {moves['Player O']}\n")
        
        # Escribir los resultados finales
        log_file.write(f"\nResults after {num_simulations} simulations:\n")
        log_file.write(f"Player X wins: {results['Player X wins']}\n")
        log_file.write(f"Player O wins: {results['Player O wins']}\n")
        log_file.write(f"Draws: {results['Draw']}\n")
        
        # Escribir el resumen final de victorias: jugador vs IA
        log_file.write("\nFinal Summary of Wins:\n")
        # Consideramos al jugador como "Player X wins" o "Player O wins", dependiendo de quién sea
        # El jugador es el que empieza, así que si 'Player X wins' es el resultado, es victoria del jugador.
        # Si 'Player O wins', es victoria de la IA.
        player_wins = results['Player X wins'] if starting_player == PLAYER_X else results['Player O wins']
        ia_wins = results['Player O wins'] if starting_player == PLAYER_X else results['Player X wins']

        log_file.write(f"Total wins by Player: {player_wins}\n")
        log_file.write(f"Total wins by AI: {ia_wins}\n")


if __name__ == "__main__":
    num_simulations = 100  # Número de partidas a simular
    difficulty = 5        # Dificultad de la IA (puedes modificarla de 1 a 5)
    run_multiple_simulations(num_simulations, difficulty)
