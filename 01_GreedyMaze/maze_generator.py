import random
from random import shuffle

# Función para generar laberinto usando DFS (Randomized Depth-First Search)
def generate_maze_dfs(rows, cols):
    maze = [[1] * cols for _ in range(rows)]
    visited = set()

    def carve_path(x, y):
        maze[x][y] = 0
        visited.add((x, y))
        neighbors = [(x + dx, y + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]]
        shuffle(neighbors)

        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                if maze[nx][ny] == 1:
                    maze[(x + nx) // 2][(y + ny) // 2] = 0  # Carve the wall
                    carve_path(nx, ny)

    start_x, start_y = 0, 0
    carve_path(start_x, start_y)
    maze[0][0] = 0  # Ensuring the start is open
    maze[rows-1][cols-1] = 0  # Ensuring the end is open
    return maze

def generate_maze_kruskal(rows, cols):
    def find(x):
        # Encuentra el representante del conjunto de x (con compresión de ruta)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        # Une dos conjuntos disjuntos
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX

    # Inicializar laberinto con paredes (1)
    maze = [[1] * cols for _ in range(rows)]
    
    # Inicializar la estructura de conjuntos disjuntos
    parent = {}
    for i in range(1, rows, 2):
        for j in range(1, cols, 2):
            parent[(i, j)] = (i, j)  # Cada celda es su propio conjunto
    
    # Lista de posibles aristas (conexiones entre celdas)
    edges = []
    for i in range(1, rows - 1, 2):
        for j in range(1, cols - 1, 2):
            if i + 2 < rows - 1:
                edges.append(((i, j), (i + 2, j)))  # Conexión vertical
            if j + 2 < cols - 1:
                edges.append(((i, j), (i, j + 2)))  # Conexión horizontal

    # Aleatorizar las aristas
    random.shuffle(edges)

    # Inicializar las celdas de camino (0) en la cuadrícula
    for i in range(1, rows, 2):
        for j in range(1, cols, 2):
            maze[i][j] = 0  # Convertir en camino

    # Aplicar Kruskal para construir el laberinto
    for (x1, y1), (x2, y2) in edges:
        if find((x1, y1)) != find((x2, y2)):  # Si están en diferentes conjuntos
            union((x1, y1), (x2, y2))
            # Quitar la pared entre las celdas
            wall_x, wall_y = (x1 + x2) // 2, (y1 + y2) // 2
            maze[wall_x][wall_y] = 0  # Convertir en camino

    # Definir inicio y fin del laberinto
    maze[1][0] = 0  # Entrada
    maze[rows - 2][cols - 1] = 0  # Salida

    return maze

def generate_maze_prim(rows, cols):
    # Inicializar el laberinto completamente cerrado (lleno de paredes)
    maze = [[1] * cols for _ in range(rows)]

    # Lista de paredes
    walls = []

    # Seleccionar una celda inicial aleatoria dentro de los límites impares
    start_x, start_y = random.randrange(1, rows, 2), random.randrange(1, cols, 2)
    maze[start_x][start_y] = 0  # Marcar la celda como pasillo

    # Agregar las paredes de la celda inicial a la lista de paredes
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nx, ny = start_x + dx, start_y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            walls.append((nx, ny, start_x, start_y))  # Almacenar la pared y la celda de origen

    while walls:
        # Seleccionar una pared aleatoria
        index = random.randint(0, len(walls) - 1)
        x, y, px, py = walls.pop(index)

        # Verificar si la celda detrás de la pared es un pasillo
        if maze[x][y] == 1:
            # Convertir la pared en pasillo
            maze[x][y] = 0
            # Convertir la celda intermedia en pasillo
            maze[(x + px) // 2][(y + py) // 2] = 0

            # Agregar las nuevas paredes circundantes
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                    walls.append((nx, ny, x, y))

    return maze

