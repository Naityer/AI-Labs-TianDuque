# greedy.py

import heapq

def greedy(maze, start, finish, heuristic):
    rows, cols = len(maze), len(maze[0])
    visited, pq, came_from, visualization, path = set(), [(heuristic(start, finish), start)], {}, [], []

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        visualization.append(current)
        if current == finish:
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return len(visualization), visualization, path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] == 0 and neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, finish), neighbor))
                came_from[neighbor] = current
    return -1, visualization, path
