import random
import time
import importlib
import matplotlib.pyplot as plt
from greedy import greedy  # Import the greedy function from the greedy.py file

# Function to generate random inputs
def randomize_input():
    # Random maze sizes
    rows = random.randint(20, 50)  # Rows between 20 and 50
    cols = random.randint(30, 70)  # Columns between 30 and 70
    
    # Available heuristics
    heuristics_module = importlib.import_module("heuristics")
    heuristics = [
        heuristics_module.heuristic_manhattan,
        heuristics_module.heuristic_euclidean,
        heuristics_module.heuristic_chebyshev,
        heuristics_module.heuristic_octile,
        heuristics_module.heuristic_diagonal,
        heuristics_module.heuristic_squared_euclidean,
        heuristics_module.heuristic_weighted_manhattan,
        heuristics_module.heuristic_hamming,
        heuristics_module.heuristic_tie_breaking,
        heuristics_module.heuristic_custom
    ]
    
    return rows, cols, heuristics

# Function to run the algorithm and heuristics
def run_test(rows, cols, maze_algorithm, heuristics, results):
    # Create the maze using the selected algorithm
    maze_module = importlib.import_module("maze_generator")
    maze = getattr(maze_module, maze_algorithm)(rows, cols)
    
    # Select random start and finish points
    start = (random.randint(0, rows-1), random.randint(0, cols-1))
    finish = (random.randint(0, rows-1), random.randint(0, cols-1))
    
    # Ensure start and finish points are not walls
    while maze[start[0]][start[1]] == 1:
        start = (random.randint(0, rows-1), random.randint(0, cols-1))
    while maze[finish[0]][finish[1]] == 1:
        finish = (random.randint(0, rows-1), random.randint(0, cols-1))

    # Test the 10 heuristics on the generated maze
    for heuristic in heuristics:
        print(f"Testing heuristic {heuristic.__name__} with the algorithm {maze_algorithm}...\n")

        # Measure execution time
        start_time = time.time()
        
        # Call the greedy function to get the data
        num_steps, visited_cells, path = greedy(maze, start, finish, heuristic)
        
        # If no solution was found (num_steps == -1), skip this run for metrics
        if num_steps == -1:
            continue
        
        # Measure execution time
        end_time = time.time()
        
        # Calculate total time
        execution_time = end_time - start_time
        
        # Store results
        results[maze_algorithm]['execution_time'].append(execution_time)
        results[maze_algorithm]['num_steps'].append(num_steps)
        results[maze_algorithm]['visited_cells'].append(len(visited_cells))  # Ensure to count visited cells correctly
        results[maze_algorithm]['path_length'].append(len(path))

# Main function to run the tests
def main():
    # Ask the user for the number of iterations to test
    num_tests = int(input("Enter the number of test iterations: "))
    
    # Dictionary to store results
    results = {
        'generate_maze_dfs': {'execution_time': [], 'num_steps': [], 'visited_cells': [], 'path_length': []},
        'generate_maze_kruskal': {'execution_time': [], 'num_steps': [], 'visited_cells': [], 'path_length': []},
        'generate_maze_prim': {'execution_time': [], 'num_steps': [], 'visited_cells': [], 'path_length': []}
    }
    
    for _ in range(num_tests):
        # Generate a random maze size and heuristics
        rows, cols, heuristics = randomize_input()
        
        print(f"\nRunning tests for a maze of size {rows}x{cols}...\n")
        
        # Test the 3 maze generation algorithms
        algorithms = ["generate_maze_dfs", "generate_maze_kruskal", "generate_maze_prim"]
        for algorithm in algorithms:
            run_test(rows, cols, algorithm, heuristics, results)

    # After all tests, generate the plots
    plot_results(results)

def plot_results(results):
    # Configure the first figure for the algorithms
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison of Heuristics for Different Maze Generation Algorithms')

    # Plot execution time
    axs[0, 0].boxplot([results[alg]['execution_time'] for alg in results], tick_labels=list(results.keys()))
    axs[0, 0].set_title('Execution Time by Algorithm')
    axs[0, 0].set_ylabel('Time (seconds)')

    # Plot number of steps
    axs[0, 1].boxplot([results[alg]['num_steps'] for alg in results], tick_labels=list(results.keys()))
    axs[0, 1].set_title('Number of Steps by Algorithm')
    axs[0, 1].set_ylabel('Number of Steps')

    # Plot visited cells
    axs[1, 0].boxplot([results[alg]['visited_cells'] for alg in results], tick_labels=list(results.keys()))
    axs[1, 0].set_title('Visited Cells by Algorithm')
    axs[1, 0].set_ylabel('Number of Visited Cells')

    # Plot path length
    axs[1, 1].boxplot([results[alg]['path_length'] for alg in results], tick_labels=list(results.keys()))
    axs[1, 1].set_title('Path Length by Algorithm')
    axs[1, 1].set_ylabel('Path Length')

    # Adjust layout and show the algorithm graph
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # New figure to compare heuristics by all metrics
    # Create a new figure (this will open a new window for the graph)
    plt.figure(figsize=(12, 8))  # New figure window
    ax2 = plt.gca()  # Get the current axis for plotting

    ax2.set_title('Comparison of Heuristics by Algorithm Metrics')
    ax2.set_xlabel('Heuristics')
    ax2.set_ylabel('Average Values')

    heuristics_names = [
        'Manhattan', 'Euclidean', 'Chebyshev', 'Octile', 'Diagonal', 
        'Squared Euclidean', 'Weighted Manhattan', 'Hamming', 
        'Tie Breaking', 'Custom'
    ]

    # Initialize the lists for metrics
    avg_execution_times = []
    avg_num_steps = []
    avg_visited_cells = []
    avg_path_lengths = []

    # Iterate over the heuristics and calculate their average values
    for heuristic in heuristics_names:
        total_time = 0
        total_steps = 0
        total_visited_cells = 0
        total_path_length = 0
        count = 0

        # Accumulate metrics for each heuristic in all algorithms
        for alg in results:
            heuristic_index = heuristics_names.index(heuristic)
            total_time += results[alg]['execution_time'][heuristic_index]
            total_steps += results[alg]['num_steps'][heuristic_index]
            total_visited_cells += results[alg]['visited_cells'][heuristic_index]
            total_path_length += results[alg]['path_length'][heuristic_index]
            count += 1
        
        # Calculate averages
        avg_time = total_time / count
        avg_steps = total_steps / count
        avg_visited = total_visited_cells / count
        avg_path_length = total_path_length / count

        avg_execution_times.append(avg_time)
        avg_num_steps.append(avg_steps)
        avg_visited_cells.append(avg_visited)
        avg_path_lengths.append(avg_path_length)

    # Group the bars for the metrics
    bar_width = 0.2
    index = range(len(heuristics_names))

    # Set the offsets for the bars
    bar1 = [x - 1.5 * bar_width for x in index]  # Execution Time
    bar2 = [x - 0.5 * bar_width for x in index]  # Number of Steps
    bar3 = [x + 0.5 * bar_width for x in index]  # Visited Cells
    bar4 = [x + 1.5 * bar_width for x in index]  # Path Length

    # Plot the grouped bars
    ax2.bar(bar1, avg_execution_times, width=bar_width, label='Execution Time', color='b')
    ax2.bar(bar2, avg_num_steps, width=bar_width, label='Number of Steps', color='g')
    ax2.bar(bar3, avg_visited_cells, width=bar_width, label='Visited Cells', color='r')
    ax2.bar(bar4, avg_path_lengths, width=bar_width, label='Path Length', color='orange')

    # Add labels and legend
    ax2.set_xticks(index)
    ax2.set_xticklabels(heuristics_names, rotation=45, ha='right')
    ax2.legend()

    # Adjust layout and show the grouped bar graph
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
