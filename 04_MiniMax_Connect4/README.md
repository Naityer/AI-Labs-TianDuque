# Connect 4 with AI (Minimax with Alpha-Beta Pruning)

This repository contains an implementation of the **Connect 4** game with an artificial intelligence (AI) opponent based on the **Minimax algorithm with Alpha-Beta Pruning**. Players can either compete against the AI or simulate AI vs. AI matches to analyze its performance.

---

## 🚀 Features
- Classic **Connect 4** gameplay in the console.
- AI opponent powered by **Minimax with Alpha-Beta Pruning**.
- Adjustable AI difficulty (levels 1 to 5).
- AI vs. AI simulations with result logging.

---

## 🧩 Implemented Algorithm
### 🎯 Minimax with Alpha-Beta Pruning
The **Minimax algorithm** is a decision rule for minimizing the possible loss in a worst-case scenario. It recursively evaluates possible game states to determine the optimal move. To optimize its performance, **Alpha-Beta Pruning** is used to cut off branches that do not need to be explored.

### 🔍 Board Evaluation Function
The board state is evaluated based on the following heuristics:
- **Number of aligned pieces**: Higher scores for more consecutive pieces.
- **Blocking opponent moves**: Reduces the opponent’s winning chances.
- **Position weighting**: Favors central positions for better connectivity.

### 📌 Minimax Implementation Steps
1. Generate all possible valid moves.
2. Check if the board is in a terminal state (win, loss, or draw).
3. Recursively explore future moves up to a defined depth.
4. Use **Alpha-Beta Pruning** to discard unnecessary branches and improve efficiency.
5. Return the best possible move for the AI.

---

## 📂 Main Files
- **`lab2_v2.py`** → Core game implementation with AI logic.
- **`test_ai_vs_ai.py`** → AI vs. AI simulation script.
- **`connect4_gui.py`** → (If available) Graphical user interface version.

---

### ▶️ How to Run
#### Player vs AI Mode
```bash
python lab2_v2.py
```
Play in the console.

```bash
python connect4_gui.py
```
Play with an interface GUI

#### AI vs AI Simulation
```bash
python test_ai_vs_ai.py
```
This script runs multiple AI vs AI games and logs the results.

---

## 📊 Simulation Results
The script `test_ai_vs_ai.py` runs AI vs. AI games and saves the results in `log.txt`, including:
- The starting player for each match.
- Moves made by both players.
- Final results (wins and draws).

Example of AI vs AI simulation results:
```
Results after 100 simulations:
Player X wins: 85
Player O wins: 15
Draws: 0

Final Summary of Wins:
Total wins by Player: 15
Total wins by AI: 85
```

---
