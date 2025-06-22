import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from main import train

# Definimos hiperparámetros a explorar
params = {
    'alpha': [0.01, 0.1, 0.5],
    'gamma': [0.9, 0.99, 0.999],
    'epsilon_decay': [0.99, 0.995, 0.999],
    'min_epsilon': [0.01, 0.05]
}

def parameter_study(env_name="MountainCar-v0", episodes=1000, seed=42):
    np.random.seed(seed)
    results = []

    # Probar cada combinación
    for combo in itertools.product(*params.values()):
        config = dict(zip(params.keys(), combo))
        print(f"🔬 Testing: {config}")
        agent, rewards = train(
            env_name=env_name,
            episodes=episodes,
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon=1.0,
            epsilon_decay=config['epsilon_decay'],
            min_epsilon=config['min_epsilon']
        )

        results.append({
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'epsilon_decay': config['epsilon_decay'],
            'min_epsilon': config['min_epsilon'],
            'mean_reward': np.mean(rewards[-100:]),
            'max_reward': np.max(rewards)
        })

    # Convertir a DataFrame y guardar CSV
    df = pd.DataFrame(results)
    df.to_csv("results/param_study_results.csv", index=False)

    # Visualizar efectos por hiperparámetro
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
    for i, param in enumerate(params.keys()):
        axes[i].scatter(df[param], df['mean_reward'])
        axes[i].set_title(f"Efecto de {param}")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Mean Reward (últimos 100)")
    plt.suptitle("Estudio de hiperparámetros - MountainCar")
    plt.tight_layout()
    plt.savefig("results/param_study_plot.png")
    plt.show()

if __name__ == "__main__":
    parameter_study()
