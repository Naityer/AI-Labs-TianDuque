import unittest
from greedy import greedy
import random
import math
from heuristics import (
    heuristic_manhattan,
    heuristic_euclidean,
    heuristic_chebyshev,
    heuristic_octile,
    heuristic_diagonal,
    heuristic_squared_euclidean,
    heuristic_weighted_manhattan,
    heuristic_hamming,
    heuristic_tie_breaking,
    heuristic_custom
)

# Updated maze generation function with a path
def generate_sample_maze(rows, cols):
    maze = [[1] * cols for _ in range(rows)]
    maze[1][0] = 0  # start
    maze[rows - 2][cols - 1] = 0  # finish
    
    # Create a simple path (a vertical and horizontal path from start to finish)
    for i in range(1, rows - 2):
        maze[i][0] = 0  # vertical path

    for j in range(0, cols - 1):
        maze[rows - 2][j] = 0  # horizontal path
    
    return maze

class TestGreedyAlgorithm(unittest.TestCase):
    def setUp(self):
        # Create a small 5x5 maze for testing
        self.rows = 5
        self.cols = 5
        self.maze = generate_sample_maze(self.rows, self.cols)
        self.start = (1, 0)
        self.finish = (self.rows - 2, self.cols - 1)

    def test_heuristic_manhattan(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_manhattan(current, goal)
        expected = 7  # |0-3| + |0-4| = 7
        self.assertEqual(result, expected, f"Expected 7 but got {result}")

    def test_heuristic_euclidean(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_euclidean(current, goal)
        expected = math.sqrt((0 - 3)**2 + (0 - 4)**2)  # sqrt(9 + 16) = sqrt(25) = 5
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_chebyshev(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_chebyshev(current, goal)
        expected = max(abs(0 - 3), abs(0 - 4))  # max(3, 4) = 4
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_octile(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_octile(current, goal)
        dx = abs(0 - 3)
        dy = abs(0 - 4)
        expected = max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_diagonal(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_diagonal(current, goal)
        expected = max(abs(0 - 3), abs(0 - 4))  # max(3, 4) = 4
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_squared_euclidean(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_squared_euclidean(current, goal)
        expected = (3**2 + 4**2)  # 9 + 16 = 25
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_weighted_manhattan(self):
        current = (0, 0)
        goal = (3, 4)
        weight = 1.5
        result = heuristic_weighted_manhattan(current, goal, weight)
        expected = weight * (abs(0 - 3) + abs(0 - 4))  # 1.5 * (3 + 4) = 10.5
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_hamming(self):
        current = (1, 1)
        goal = (1, 0)
        result = heuristic_hamming(current, goal)
        expected = 1  # Hamming distance = 1 (1 != 0)
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_tie_breaking(self):
        current = (0, 0)
        goal = (3, 4)
        weight = 1.1
        result = heuristic_tie_breaking(current, goal, weight)
        expected = weight * (abs(0 - 3) + abs(0 - 4))  # 1.1 * (3 + 4) = 7.7
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_heuristic_custom(self):
        current = (0, 0)
        goal = (3, 4)
        result = heuristic_custom(current, goal)
        manhattan = abs(0 - 3) + abs(0 - 4)  # 7
        euclidean = math.sqrt((0 - 3)**2 + (0 - 4)**2)  # 5
        expected = (manhattan + euclidean) / 2  # (7 + 5) / 2 = 6
        self.assertEqual(result, expected, f"Expected {expected}, but got {result}")

    def test_greedy_algorithm_path(self):
        # Run the greedy algorithm using the Manhattan heuristic
        num_steps, visited_cells, path = greedy(self.maze, self.start, self.finish, heuristic_manhattan)

        # Print maze and debug information (for visualization)
        print("Maze:")
        for row in self.maze:
            print(row)

        print(f"Visited Cells: {visited_cells}")
        print(f"Path: {path}")
        
        # Check that the path is not empty
        self.assertGreater(len(path), 0, "Path should not be empty")

        # Check if the path starts at the start point and ends at the finish point
        # The path is reversed in the greedy algorithm, so we check the last element as the start and first element as the finish
        self.assertEqual(path[0], self.start, "Path should start at the start point")
        self.assertEqual(path[-1], self.finish, "Path should end at the finish point")
        
        # Check if the number of steps is reasonable (steps should not exceed maze size)
        self.assertLessEqual(num_steps, self.rows * self.cols, "Steps exceed maze size")

    def test_no_wall_in_start_finish(self):
        # Check that the start and finish positions are open (0)
        self.assertEqual(self.maze[self.start[0]][self.start[1]], 0, "Start should be open")
        self.assertEqual(self.maze[self.finish[0]][self.finish[1]], 0, "Finish should be open")
    
    def test_maze_size(self):
        # Check maze size to ensure it's as expected
        self.assertEqual(len(self.maze), self.rows, f"Maze should have {self.rows} rows")
        self.assertEqual(len(self.maze[0]), self.cols, f"Maze should have {self.cols} columns")

if __name__ == '__main__':
    unittest.main()
