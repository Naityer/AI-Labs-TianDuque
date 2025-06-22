import tkinter as tk
import heapq
import importlib
from random import randint
from greedy import greedy

def select_algorithm():
    # List of the 3 available algorithms
    algorithms = {
        "dfs": "generate_maze_dfs",
        "kruskal": "generate_maze_kruskal",
        "prim": "generate_maze_prim"
    }
    while True:
        algo = input("Select the maze generation algorithm (dfs, kruskal, prim): ").strip().lower()
        if algo in algorithms:
            return algorithms[algo]
        print("Invalid algorithm. Please try again.")

def select_heuristic(prompt, used_heuristics):
    heuristics_module = importlib.import_module("heuristics")
    heuristics = [
        ("manhattan", heuristics_module.heuristic_manhattan),
        ("euclidean", heuristics_module.heuristic_euclidean),
        ("chebyshev", heuristics_module.heuristic_chebyshev),
        ("octile", heuristics_module.heuristic_octile),
        ("diagonal", heuristics_module.heuristic_diagonal),
        ("squared_euclidean", heuristics_module.heuristic_squared_euclidean),
        ("weighted_manhattan", heuristics_module.heuristic_weighted_manhattan),
        ("hamming", heuristics_module.heuristic_hamming),
        ("tie_breaking", heuristics_module.heuristic_tie_breaking),
        ("custom", heuristics_module.heuristic_custom)
    ]

    print(prompt)
    for index, (name, _) in enumerate(heuristics):
        print(f"{index}: {name}")

    while True:
        try:
            index = int(input("Select the heuristic index: "))
            if 0 <= index < len(heuristics):
                selected_name, selected_heuristic = heuristics[index]
                if selected_name in used_heuristics:
                    print("This heuristic has already been selected. Choose another one.")
                    continue
                used_heuristics.add(selected_name)
                return selected_heuristic
            else:
                print("Index out of range. Please try again.")
        except ValueError:
            print("Invalid input. Please try again.")

def visualize(viz, path, maze, start, finish, canvas, label):
    canvas.delete("all")
    rows, cols = len(maze), len(maze[0])
    
    for i in range(rows):
        for j in range(cols):
            color = "white" if maze[i][j] == 0 else "black"
            canvas.create_rectangle(j * 20, i * 20, (j + 1) * 20, (i + 1) * 20, fill=color, outline="gray")
    
    for x, y in viz:
        canvas.create_rectangle(y * 20, x * 20, (y + 1) * 20, (x + 1) * 20, fill="yellow", outline="gray")
    
    for x, y in path:
        canvas.create_rectangle(y * 20, x * 20, (y + 1) * 20, (x + 1) * 20, fill="blue", outline="gray")
    
    canvas.create_rectangle(start[1] * 20, start[0] * 20, (start[1] + 1) * 20, (start[0] + 1) * 20, fill="green", outline="gray")
    canvas.create_rectangle(finish[1] * 20, finish[0] * 20, (finish[1] + 1) * 20, (finish[0] + 1) * 20, fill="red", outline="gray")
    
    label.config(text=f"Steps: {len(viz)}")

def start_search(canvas, label, maze, start, finish, heuristic):
    num_steps, visited_cells, path = greedy(maze, start, finish, heuristic)
    print(f"Path found in {num_steps} steps using {heuristic.__name__}.")
    visualize(visited_cells, path, maze, start, finish, canvas, label)

def create_gui(rows, cols, maze_generator_func, heuristic):
    root = tk.Tk()
    root.title("Maze with Greedy Best-First Search")
    
    canvas = tk.Canvas(root, width=cols * 20, height=rows * 20)
    canvas.pack(side="left")
    
    legend_frame = tk.Frame(root)
    legend_frame.pack(side="right", padx=10, pady=10)
    
    legend_label = tk.Label(legend_frame, text="Color Legend:", font=("Arial", 12, "bold"))
    legend_label.grid(row=0, column=0, sticky="w", pady=5)
    
    legend_items = [
        ("Yellow", "Visited"),
        ("Blue", "Path"),
        ("Green", "Start"),
        ("Red", "Finish"),
        ("Black", "Wall")
    ]
    
    for idx, (color, description) in enumerate(legend_items):
        color_label = tk.Label(legend_frame, text=f"{color}: {description}", font=("Arial", 10))
        color_label.grid(row=idx+1, column=0, sticky="w", pady=2)
    
    label = tk.Label(root, text="Steps: 0")
    label.pack()
    
    maze_module = importlib.import_module("maze_generator")
    maze = getattr(maze_module, maze_generator_func)(rows, cols)
    start, finish = (randint(0, rows-1), randint(0, cols-1)), (randint(0, rows-1), randint(0, cols-1))
    
    while maze[start[0]][start[1]] == 1:
        start = (randint(0, rows-1), randint(0, cols-1))
    
    while maze[finish[0]][finish[1]] == 1:
        finish = (randint(0, rows-1), randint(0, cols-1))
    
    start_search(canvas, label, maze, start, finish, heuristic)
    
    root.mainloop()

def get_maze_dimensions():
    while True:
        try:
            rows = int(input("Number of rows (default 30): ") or 30)
            cols = int(input("Number of columns (default 50): ") or 50)
            return rows, cols
        except ValueError:
            print("Invalid input. Please try again.")

rows, cols = get_maze_dimensions()
maze_algorithm = select_algorithm()

used_heuristics = set()
heuristic = select_heuristic("Select the heuristic: ", used_heuristics)

create_gui(rows, cols, maze_algorithm, heuristic)
