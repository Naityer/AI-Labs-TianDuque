import tkinter as tk
import heapq
from random import shuffle, randint
import math

# Archivo heuristics.py
# Definición de múltiples heurísticas

def heuristic_manhattan(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def heuristic_euclidean(current, goal):
    return math.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

def heuristic_chebyshev(current, goal):
    return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))

def heuristic_octile(current, goal):
    dx = abs(current[0] - goal[0])
    dy = abs(current[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def heuristic_diagonal(current, goal):
    return max(abs(current[0] - goal[0]), abs(current[1] - goal[1]))

def heuristic_squared_euclidean(current, goal):
    return (current[0] - goal[0])**2 + (current[1] - goal[1])**2

def heuristic_weighted_manhattan(current, goal, weight=1.5):
    return weight * (abs(current[0] - goal[0]) + abs(current[1] - goal[1]))

def heuristic_hamming(current, goal):
    return sum(c1 != c2 for c1, c2 in zip(current, goal))

def heuristic_tie_breaking(current, goal, weight=1.1):
    return weight * heuristic_manhattan(current, goal)

def heuristic_custom(current, goal):
    return (heuristic_manhattan(current, goal) + heuristic_euclidean(current, goal)) / 2
