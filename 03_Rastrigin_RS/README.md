# 🧬 Rastrigin_RS – Genetic Algorithm Optimization (Lab 3)

This project implements a Genetic Algorithm (GA) in Python to minimize the 2D Rastrigin function using **Rank Selection** as the parent selection strategy. It was developed as part of **Artificial Intelligence Lab 3** (Variant 2) for educational purposes.

## Professor
- Valeriya Khan

## Date
23/03 – Summer 2025

## Variant 2
Optimize Rastrigin function using Rank Selection

## Students
- Tian Duque Rey  
- Eduardo Sánchez Belchí

---

## 🛠️ Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

Install with:

```bash
pip install -r requirements.txt
```

## 📌 Objective

To find the **global minimum** of the Rastrigin function, a classic benchmark in evolutionary optimization, by evolving a population of real-valued solutions using biologically-inspired operators.

## ⚙️ Algorithm Overview

- **Selection**: Rank-based (individuals selected with probabilities based on their fitness ranking)
- **Crossover**: Random interpolation using real-valued parents  
  \[
  x_o = \alpha x_{p1} + (1 - \alpha) x_{p2}
  \]
- **Mutation**: Gaussian mutation applied to a random portion of the offspring
- **Elitism**: Part of the population is preserved unaltered

## 📈 Target Function

\[
f(x, y) = 20 + (x^2 - 10\cos(2\pi x)) + (y^2 - 10\cos(2\pi y))
\]

- **Search domain**: \( x, y \in [-5, 5] \)
- **Global minimum**: \( f(0, 0) = 0 \)

## 🧪 Experiments

The project includes **four standalone test scripts**, each corresponding to one experiment. You can execute them directly from the command line:

```bash
# Run Experiment 1: Hyperparameter search
python test_1.py

# Run Experiment 2: Randomness and population size
python test_2.py

# Run Experiment 3: Crossover rate impact
python test_3.py

# Run Experiment 4: Mutation rate and strength impact
python test_4.py
```

Each script is fully executable and will generate plots and CSV result files inside the `data_test/` directory.
