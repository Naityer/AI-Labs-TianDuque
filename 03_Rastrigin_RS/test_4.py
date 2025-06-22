import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

output_dir = "data_test"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Leer los mejores parámetros del Experimento 1
df_exp1 = pd.read_csv(os.path.join(output_dir, "experiment1_results.csv"))
best_row = df_exp1.loc[df_exp1["best_fitness"].idxmin()]

base_params = {
    "population_size": int(best_row["population"]),
    "crossover_rate": best_row["crossover_rate"],
    "num_generations": int(best_row["generations"]),
}

# 🔹 Rango de tasas de mutación y fuerza de mutación a explorar
mutation_rates = [0.01, 0.05, 0.1, 0.2]
mutation_strengths = [0.1, 0.3, 0.5, 0.8, 1.0]
random_seeds = [10, 20, 30, 40, 50]

avg_best_per_gen = {}
avg_mean_per_gen = {}

print("\n📊 Running Experiment 4: Mutation and Convergence")

summary_rows = []
num_generations = base_params["num_generations"]

for m_rate in mutation_rates:
    for m_strength in mutation_strengths:
        best_runs = []
        mean_runs = []

        for seed in random_seeds:
            ga = GeneticAlgorithm(
                population_size=base_params["population_size"],
                mutation_rate=m_rate,
                mutation_strength=m_strength,
                crossover_rate=base_params["crossover_rate"],
                num_generations=base_params["num_generations"],
            )
            _, best_fitness, avg_fitness = ga.evolve(seed=seed)
            best_runs.append(best_fitness)
            mean_runs.append(avg_fitness)

        label = f"rate={m_rate}, strength={m_strength}"
        avg_best = np.mean(best_runs, axis=0)
        avg_mean = np.mean(mean_runs, axis=0)
        avg_best_per_gen[label] = avg_best
        avg_mean_per_gen[label] = avg_mean

        for gen in range(num_generations):
            summary_rows.append({
                "Mutation Rate": m_rate,
                "Mutation Strength": m_strength,
                "Generation": gen,
                "Avg Best Fitness": avg_best[gen],
                "Avg Mean Fitness": avg_mean[gen],
            })

# 📈 Graficar resultados
plt.figure(figsize=(12, 6))
for label, values in avg_best_per_gen.items():
    plt.plot(values, label=f"Best {label}")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Best Fitness - Mutation Impact")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment4_best_fitness.png"))

plt.figure(figsize=(12, 6))
for label, values in avg_mean_per_gen.items():
    plt.plot(values, label=f"Avg {label}")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Average Fitness - Mutation Impact")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment4_average_fitness.png"))

# 📄 Guardar CSV con los resultados
summary_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(output_dir, "experiment4_summary.csv")
summary_df.to_csv(csv_path, index=False)

print("\n📄 Saved Experiment 4 plots and summary to:")
print(" - experiment4_best_fitness.png")
print(" - experiment4_average_fitness.png")
print(f" - {csv_path}")
