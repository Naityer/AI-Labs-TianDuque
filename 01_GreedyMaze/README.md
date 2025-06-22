# Exercise 1: Search

## Professor
- Valeriya Khan

## Date
23/03  Summer 2025

## Variant 2
Greedy Best-First Search

## Students
- Tian Duque Rey
- Eduardo Sánchez Belchí

---


# **Maze Generator and Greedy Search Visualization**

This project provides a graphical interface to generate mazes using three different algorithms and visualize the **Greedy Best-First Search** algorithm with multiple heuristics.  

**Greedy Best-First Search (GBFS)** is an informed search algorithm that selects the node to expand based solely on a heuristic function \( h(n) \), which estimates the proximity of the current node to the goal. Unlike algorithms like A*, which consider the accumulated path cost, GBFS focuses only on the most promising path based on the heuristic, making it faster but less optimal in some cases. It expands the node with the lowest \( h(n) \) value and continues until it finds the goal or exhausts all nodes. Its efficiency depends on the quality of the heuristic used.

## **Installation & Requirements**  

Make sure you have Python installed. The required libraries are:  
- `tkinter` (for GUI)  
- `heapq` (for priority queue)  
- `random` (for randomization)  
- `importlib` (for dynamic module loading)  
- `math` (for heuristic calculations)  

To run the project, simply execute:  
```bash
python lab1_v2.py
```
To run the Data Analyser, simply execute:  
```bash
python analysis_gbf.py
```
To run the Unit Test, simply execute:  
```bash
python -m unittest test_code.py
```

## **Usage Instructions and Example Run**  
To run the project, simply execute:  
```bash
python lab1_v2.py
```
1. Run the script and enter the **maze dimensions**.
2. Select a **maze generation algorithm** (DFS, Kruskal, or Prim).  
3. Choose the **heuristic** for comparison.  
4. Watch the **search visualization** in the GUI.
   
```
Enter number of rows (default 30): 30  
Enter number of columns (default 50): 50  
Select maze generation algorithm (dfs, kruskal, prim): prim  
Select first heuristic: 0 (Manhattan)    
```
A graphical window will open showing the generated maze and the greedy search process => **Visualization Colors**  

- **Green**: Start position  
- **Red**: Goal position  
- **Yellow**: Explored nodes  
- **Blue**: Final path  
- **Black**: Walls

## **Features**  

- Generate mazes using **Depth-First Search (DFS), Kruskal's Algorithm, or Prim's Algorithm**.  
- Use **Greedy Best-First Search** to solve the maze.  
- Choose between **multiple heuristics** for pathfinding.  
- Visualize the algorithm’s step-by-step search process.  

## Maze Generation Algorithms

### Depth-First Search (DFS)
Starts from an initial cell and explores as deeply as possible by selecting random neighboring cells. If no options are available, it backtracks until a new path is found. This algorithm produces mazes with long corridors and fewer branches.

### Kruskal’s Algorithm
Treats each cell as an independent set and removes walls to connect disjoint sets. Walls are randomly selected and removed if they separate different regions. This results in mazes with multiple alternative paths.

### Prim’s Algorithm
Begins with a random cell and expands the maze by adding new cells from the existing frontier. Walls are removed to connect cells until the entire structure is covered. This algorithm creates more uniform and interconnected mazes.  

## **Search Analysis (analysis_gbf.py)**  
```bash
python analysis_gbf.py
```
This script evaluates the performance of the **Greedy Best-First Search** algorithm on mazes generated using different algorithms (**DFS, Kruskal, and Prim**), applying ten different heuristics. It randomly generates maze sizes, selects start and goal points, and runs the search algorithm while measuring **execution time, step count, visited cells, and path length**. The results are stored and visualized using **matplotlib**, enabling a comparative analysis of the efficiency of each heuristic and maze generation method in pathfinding problems. 

## Testing Script
```bash
python -m unittest test_code.py
```
This script implements unit tests to evaluate the functionality of the **Greedy Best-First Search** algorithm in solving mazes. Using the `unittest` library, it verifies the accuracy of various heuristics, including Manhattan, Euclidean, and Chebyshev, applied during the search process. 

The script generates a sample maze, runs the search algorithm, and ensures that the computed path is correct. Additionally, it checks that the maze dimensions are as expected and that the start and goal positions are traversable.

# Heuristics for Mazes

## 1. Manhattan Distance
📌 **Usage:** For mazes where you can only move up, down, left, and right.  
🔍 **Behavior:** Expands nodes first in a straight line towards the goal. If the maze has many walls, it may take longer paths.  
✏️ **Formula:**  
\[
d = |x_2 - x_1| + |y_2 - y_1|
\]

## 2. Euclidean Distance
📌 **Usage:** For movements in any direction, including diagonals.  
🔍 **Behavior:** Expands in a more circular pattern. If there are diagonal walls, it will try to move diagonally even if it can't.  
✏️ **Formula:**  
\[
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
\]

## 3. Chebyshev Distance
📌 **Usage:** When moving diagonally has the same cost as moving in a straight line.  
🔍 **Behavior:** Treats all directions as equally valid. Expands uniformly.  
✏️ **Formula:**  
\[
d = \max(|x_2 - x_1|, |y_2 - y_1|)
\]

## 4. Octile Distance
📌 **Usage:** For diagonal movements with adjusted cost.  
🔍 **Behavior:** Slightly favors diagonal movement if it is more efficient.  
✏️ **Formula:**  
\[
d = \max(d_x, d_y) + (\sqrt{2} - 1) \times \min(d_x, d_y)
\]

## 5. Diagonal Distance
📌 **Usage:** For mazes where moving diagonally has no extra cost.  
🔍 **Behavior:** Seeks more direct diagonal paths. Not ideal if walls block diagonals.  
✏️ **Formula:**  
\[
d = \max(|x_2 - x_1|, |y_2 - y_1|)
\]

## 6. Squared Euclidean Distance
📌 **Usage:** Similar to Euclidean, but without the square root calculation for greater efficiency.  
🔍 **Behavior:** Behaves like Euclidean but with less computational cost.  
✏️ **Formula:**  
\[
d = (x_2 - x_1)^2 + (y_2 - y_1)^2
\]

## 7. Weighted Manhattan Distance
📌 **Usage:** Similar to Manhattan, but with an additional weight to adjust priority.  
🔍 **Behavior:** Prefers certain paths based on terrain costs.  
✏️ **Formula:**  
\[
d = w \times (|x_2 - x_1| + |y_2 - y_1|)
\]

## 8. Hamming Distance
📌 **Usage:** For mazes with discrete movements, such as in state matrices.  
🔍 **Behavior:** Measures how many changes are needed to reach the goal.  
✏️ **Formula:**  
It does not have a standard formula like the others; it measures the number of differences between the current state and the goal.

## 9. Tie Breaking Heuristic
📌 **Usage:** To prevent BFS from exploring random directions in case of ties.  
🔍 **Behavior:** Favors more promising paths and reduces unnecessary ones.  
✏️ **Formula:**  
Depends on the implementation; there is no fixed formula.

## 10. Custom Heuristic (Manhattan + Euclidean)
📌 **Usage:** To balance between Manhattan and Euclidean.  
🔍 **Behavior:** Expands in a balanced way, taking advantage of the maze structure without overloading computation.  
✏️ **Formula:**  
\[
d = \frac{1}{2} (d_{\text{Manhattan}} + d_{\text{Euclidean}})
\]

# Comparison of Advantages and Disadvantages of Greedy Best-First Search (GBFS) with Different Heuristics

## 1. **Manhattan Distance**
**Advantages:**
- GBFS with the Manhattan heuristic is highly efficient in mazes where movement is restricted to up, down, left, and right. 
- The heuristic provides a straightforward estimate of the distance to the goal, enabling the algorithm to quickly prioritize promising paths and avoid unnecessary exploration.

**Disadvantages:**
- In environments with many obstacles, the Manhattan heuristic might lead the algorithm to take longer paths since it doesn't consider the presence of walls.
- Additionally, the lack of diagonal movement in the heuristic could result in missing more optimal paths in certain maze designs.

---

## 2. **Euclidean Distance**
**Advantages:**
- With the Euclidean heuristic, GBFS expands nodes in a more natural, circular pattern.
- This is particularly useful in mazes where diagonal movement is allowed, enabling the algorithm to explore the search space more efficiently by considering both straight and diagonal paths.

**Disadvantages:**
- In mazes with diagonal walls or restricted areas, the Euclidean heuristic may cause the algorithm to attempt diagonal movements even when they are not feasible, leading to inefficiencies and incorrect explorations.

---

## 3. **Chebyshev Distance**
**Advantages:**
- The Chebyshev heuristic treats diagonal movements as equally valid as horizontal and vertical moves.
- It offers balanced and uniform expansion across all directions, making it efficient in mazes where all types of movement are permissible.

**Disadvantages:**
- Although effective in many cases, it may become inefficient in mazes where diagonal movements are not always viable due to obstacles.
- This could lead to unnecessary exploration in directions that are not ideal for the given maze structure.

---

## 4. **Octile Distance**
**Advantages:**
- GBFS using the Octile heuristic works well in mazes where diagonal movements are useful but come at a slight cost.
- The heuristic prioritizes diagonal movements when they provide shorter paths, leading to faster exploration in open maze layouts.

**Disadvantages:**
- While this heuristic is useful, its more complex calculation compared to simpler heuristics like Manhattan or Euclidean can increase computational costs.
- Additionally, if the maze has walls blocking diagonal paths, the algorithm may not perform as efficiently.

---

## 5. **Diagonal Distance**
**Advantages:**
- In mazes where diagonal movements are free, this heuristic is ideal as it allows GBFS to take more direct paths, expanding efficiently in environments where walls do not block diagonal movement.

**Disadvantages:**
- If the maze has obstacles that block diagonal paths, the algorithm may lose efficiency by overly favoring diagonal movements, potentially missing better paths that avoid these obstacles.

---

## 6. **Squared Euclidean Distance**
**Advantages:**
- This heuristic is similar to Euclidean but without the computational overhead of calculating the square root, making it faster while still providing a good approximation of the true distance.
- It works well in scenarios where diagonal movements are viable.

**Disadvantages:**
- While more efficient computationally, the Squared Euclidean heuristic sacrifices some precision since it doesn’t calculate the exact distance, which could result in less optimal paths in some mazes.

---

## 7. **Weighted Manhattan Distance**
**Advantages:**
- This heuristic introduces flexibility by adjusting path priorities based on terrain costs. 
- It is particularly effective in mazes with varying terrain or obstacles, allowing GBFS to avoid high-cost areas and find more optimal solutions.

**Disadvantages:**
- The success of this heuristic depends on the proper calibration of the weights. If the weights are not set correctly, the algorithm may favor suboptimal paths, affecting its overall efficiency.

---

## 8. **Hamming Distance**
**Advantages:**
- The Hamming heuristic is useful in discrete mazes or state-space problems, as it measures the number of changes required to reach the goal.
- It is effective in scenarios where the maze consists of discrete moves or when dealing with matrices of states.

**Disadvantages:**
- This heuristic is not suitable for all maze types, particularly those involving physical distance or complex obstacles.
- Since it only measures the number of state differences, it doesn’t account for the spatial layout of the maze, limiting its application.

---

## 9. **Tie Breaking Heuristic**
**Advantages:**
- This heuristic helps prevent GBFS from exploring random directions when ties occur between nodes with the same heuristic value.
- It introduces a level of decision-making that can prioritize more promising paths, reducing unnecessary exploration and improving efficiency.

**Disadvantages:**
- Since the Tie Breaking heuristic does not have a fixed formula, its effectiveness depends on how it’s implemented.
- If not handled correctly, it may introduce inconsistency or inefficiency in ambiguous situations.

---

## 10. **Custom Heuristic (Manhattan + Euclidean)**
**Advantages:**
- By combining the strengths of both the Manhattan and Euclidean heuristics, this custom heuristic provides a balanced approach.
- It expands nodes efficiently, leveraging the advantages of both heuristics without overwhelming the computational resources.

**Disadvantages:**
- Although this heuristic offers a balanced approach, it may increase computational costs since it requires the calculation of both heuristics.
- If the maze doesn’t benefit from both approaches, the added complexity might not justify the potential improvement in pathfinding performance.

---

