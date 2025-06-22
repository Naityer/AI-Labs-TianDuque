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
    "mutation_rate": best_row["mutation_rate"],
    "mutation_strength": best_row["mutation_strength"],
    "num_generations": int(best_row["generations"]),
}

# 🔹 Valores de crossover a probar
crossover_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
random_seeds = [10, 20, 30, 40, 50]  # Para promediar resultados

avg_best_per_gen = {}
avg_mean_per_gen = {}

print("\n📊 Running Experiment 3: Crossover Impact")

for c_rate in crossover_values:
    best_runs = []
    mean_runs = []

    for seed in random_seeds:
        ga = GeneticAlgorithm(
            population_size=base_params["population_size"],
            mutation_rate=base_params["mutation_rate"],
            mutation_strength=base_params["mutation_strength"],
            crossover_rate=c_rate,
            num_generations=base_params["num_generations"],
        )
        _, best_fitness, avg_fitness = ga.evolve(seed=seed)
        best_runs.append(best_fitness)
        mean_runs.append(avg_fitness)

    # Promediar resultados por generación
    avg_best = np.mean(best_runs, axis=0)
    avg_mean = np.mean(mean_runs, axis=0)
    avg_best_per_gen[c_rate] = avg_best
    avg_mean_per_gen[c_rate] = avg_mean

# 📈 Graficar resultados
plt.figure(figsize=(10, 6))
for c_rate, values in avg_best_per_gen.items():
    plt.plot(values, label=f"Best (Crossover {c_rate})")
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Across Generations for Different Crossover Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment3_best_fitness.png"))

plt.figure(figsize=(10, 6))
for c_rate, values in avg_mean_per_gen.items():
    plt.plot(values, label=f"Avg (Crossover {c_rate})")
plt.xlabel("Generation")
plt.ylabel("Average Fitness")
plt.title("Average Fitness Across Generations for Different Crossover Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment3_average_fitness.png"))

print("\n📄 Saved Experiment 3 plots to:")
print(" - experiment3_best_fitness.png")
print(" - experiment3_average_fitness.png")

# 📄 Guardar resultados en CSV
summary_rows = []
num_generations = base_params["num_generations"]

for c_rate in crossover_values:
    for gen in range(num_generations):
        summary_rows.append({
            "Crossover Rate": c_rate,
            "Generation": gen,
            "Avg Best Fitness": avg_best_per_gen[c_rate][gen],
            "Avg Mean Fitness": avg_mean_per_gen[c_rate][gen],
        })

df_summary = pd.DataFrame(summary_rows)
csv_path = os.path.join(output_dir, "experiment3_summary.csv")
df_summary.to_csv(csv_path, index=False)

print(f"\n📄 Saved crossover impact data to {csv_path}")
