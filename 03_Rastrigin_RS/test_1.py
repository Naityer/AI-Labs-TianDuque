import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from functions import rastrigin_2d

if __name__ == "__main__":
    # Crear carpeta de salida si no existe
    output_dir = "data_test"
    os.makedirs(output_dir, exist_ok=True)

    # 🚀 EXPERIMENTO 1 - Búsqueda exhaustiva de hiperparámetros
    param_grid = {
        "population_size": [50, 100, 200],  # ➕ Más tamaños
        "mutation_rate": [0.01, 0.05, 0.1, 0.2],  # ➕ Más variedad
        "mutation_strength": [0.05, 0.1, 0.3, 0.5, 0.8],  # ➕ Más granularidad
        "crossover_rate": [0.3, 0.5, 0.7, 0.9],
        "num_generations": [100, 200]  # ➕ Pruebas más largas
    }

    results = []
    combinations = list(itertools.product(
        param_grid["population_size"],
        param_grid["mutation_rate"],
        param_grid["mutation_strength"],
        param_grid["crossover_rate"],
        param_grid["num_generations"]
    ))

    best_fitness_series = []
    labels = []

    for idx, (pop, m_rate, m_strength, c_rate, n_gen) in enumerate(combinations):
        print(f"[Experiment 1] Running combination {idx + 1}/{len(combinations)}")

        ga = GeneticAlgorithm(
            population_size=pop,
            mutation_rate=m_rate,
            mutation_strength=m_strength,
            crossover_rate=c_rate,
            num_generations=n_gen,
        )

        best_solutions, best_fitness_values, avg_fitness_values = ga.evolve(seed=42)

        results.append({
            "test_id": idx + 1,  # ✅ ID del test
            "population": pop,
            "mutation_rate": m_rate,
            "mutation_strength": m_strength,
            "crossover_rate": c_rate,
            "generations": n_gen,
            "best_fitness": best_fitness_values[-1],
            "average_fitness": avg_fitness_values[-1],
            "solution": best_solutions[-1],
            "fitness_curve": best_fitness_values  # Guardar curva completa
        })

    # Ordenar resultados por mejor fitness final y quedarnos con los 10 mejores
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="best_fitness").head(10)

    # Graficar curvas de convergencia de los 10 mejores
    plt.figure(figsize=(12, 6))
    for i, row in df_sorted.iterrows():
        curve = row["fitness_curve"]
        label = f"Test {int(row['test_id'])} | Pop={row['population']}, MR={row['mutation_rate']}, MS={row['mutation_strength']}, CR={row['crossover_rate']}"
        plt.plot(curve, label=label)

    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Top 10 Convergence Curves (Generations = 200)")
    plt.legend(fontsize=7, loc="upper right")
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(output_dir, "fitness_plot_exp1_top10.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"\n📊 Saved plot to {plot_path}")

    # Eliminar curvas antes de guardar CSV final (no se puede guardar listas en CSV directamente)
    df.drop(columns=["fitness_curve"], inplace=True)
    csv_path = os.path.join(output_dir, "experiment1_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n📄 Saved results of Experiment 1 to {csv_path}")
