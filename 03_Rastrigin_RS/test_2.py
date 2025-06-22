import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm

output_dir = "data_test"
os.makedirs(output_dir, exist_ok=True)

# ✅ Detectar mejores parametros desde el experimento 1
df_exp1 = pd.read_csv(os.path.join(output_dir, "experiment1_results.csv"))

# Seleccionamos la fila con el menor best_fitness
best_row = df_exp1.loc[df_exp1["best_fitness"].idxmin()]

best_params = {
    "population_size": int(best_row["population"]),
    "mutation_rate": best_row["mutation_rate"],
    "mutation_strength": best_row["mutation_strength"],
    "crossover_rate": best_row["crossover_rate"],
    "num_generations": int(best_row["generations"]),
}

print("\n✨ Best parameters selected from Experiment 1:")
print(best_params)

random_seeds = [10, 20, 30, 40, 50]
results = []
fitness_curves = []

print("\n🔹 Running with 5 different seeds...")

# 🔸 Parte 1: Aleatoriedad con mejores parametros
for seed in random_seeds:
    ga = GeneticAlgorithm(**best_params)
    _, best_fitness, _ = ga.evolve(seed=seed)
    results.append(best_fitness[-1])

results = np.array(results)
summary_df = pd.DataFrame({
    "Seed": random_seeds,
    "Final Fitness": results
})
summary_df.loc[len(summary_df.index)] = ["Mean", results.mean()]
summary_df.loc[len(summary_df.index)] = ["Std", results.std()]
summary_df.loc[len(summary_df.index)] = ["Min", results.min()]

summary_path = os.path.join(output_dir, "experiment2_randomness_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\n📄 Saved randomness summary to {summary_path}")

# 📈 Graficar barras del fitness final por semilla
plt.figure(figsize=(8, 5))
plt.bar([str(s) for s in random_seeds], results, color='skyblue')
plt.axhline(y=results.mean(), color='orange', linestyle='--', label='Mean')
plt.axhline(y=results.min(), color='green', linestyle='--', label='Min')
plt.xlabel("Seed")
plt.ylabel("Final Best Fitness")
plt.title("Experiment 2 - Final Fitness per Seed")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment2_randomness_barplot.png"))
plt.show()

# 🔸 Parte 2: Evaluar poblaciones reducidas
print("\n🔹 Running with smaller population sizes...")

pop_factors = [0.5, 0.25, 0.1]
small_pop_results = []

for factor in pop_factors:
    pop_size = int(best_params["population_size"] * factor)
    run_fitnesses = []
    for seed in random_seeds:
        ga = GeneticAlgorithm(
            population_size=pop_size,
            mutation_rate=best_params["mutation_rate"],
            mutation_strength=best_params["mutation_strength"],
            crossover_rate=best_params["crossover_rate"],
            num_generations=best_params["num_generations"],
        )
        _, best_fitness, _ = ga.evolve(seed=seed)
        run_fitnesses.append(best_fitness[-1])

    small_pop_results.append({
        "Population": pop_size,
        "Mean Fitness": np.mean(run_fitnesses),
        "Std Dev": np.std(run_fitnesses),
        "Best Fitness": np.min(run_fitnesses)
    })

pop_df = pd.DataFrame(small_pop_results)
pop_path = os.path.join(output_dir, "experiment2_population_summary.csv")
pop_df.to_csv(pop_path, index=False)
print(f"\n📄 Saved population reduction results to {pop_path}")

# 📈 Graficar comparativa por tamaño de población
plt.figure(figsize=(8, 5))
plt.errorbar(pop_df["Population"], pop_df["Mean Fitness"], yerr=pop_df["Std Dev"], fmt='o-', capsize=5, label='Mean Fitness ± Std')
plt.scatter(pop_df["Population"], pop_df["Best Fitness"], color='green', label='Best Fitness')
plt.xlabel("Population Size")
plt.ylabel("Fitness")
plt.title("Experiment 2 - Fitness vs Population Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "experiment2_population_plot.png"))
plt.show()
