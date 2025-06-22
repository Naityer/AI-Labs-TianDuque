import os
import time
import gymnasium as gym
import matplotlib.pyplot as plt
from q_learning import QLearningAgent
import numpy as np
from datetime import datetime
from utils import save_summary  # ✅ nuevo

def train(env_name="MountainCar-v0", episodes=1000,
          alpha=0.1, gamma=0.99, epsilon=0.1,
          epsilon_decay=0.995, min_epsilon=0.01):
    """
    Entrena un agente Q-Learning en el entorno especificado.
    Incluye reward shaping y registro de resultados.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{env_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    env = gym.make(env_name)
    agent = QLearningAgent(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon
    )

    rewards = []
    episode_lengths = []
    epsilons = []

    start_time = time.time()  # ⏱️ inicio

    for episode in range(episodes):
        state_raw, _ = env.reset()
        state = agent.discretize(state_raw)
        total_reward = 0
        steps = 0
        done = False
        cima_alcanzada = False

        while not done:
            action = agent.choose_action(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize(next_state_raw)

            shaped_reward = agent.shaped_reward(next_state_raw, reward)

            if not cima_alcanzada and next_state_raw[0] >= 0.5:
                print(f"✅ Episodio {episode}: ¡Cima alcanzada!")
                cima_alcanzada = True

            agent.update(state, action, shaped_reward, next_state, done)
            state = next_state

            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilons.append(agent.epsilon)

        if episode % 100 == 0 or episode == episodes - 1:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode}: "
                  f"Reward={total_reward:.1f}, "
                  f"Avg(100)={avg_reward:.1f}, "
                  f"Length={steps}, "
                  f"Epsilon={agent.epsilon:.3f}")

    env.close()
    elapsed = time.time() - start_time
    save_results(agent, rewards, episode_lengths, epsilons, results_dir)
    save_summary(results_dir, config, rewards, epsilons, elapsed)
    return agent, rewards

def save_results(agent, rewards, lengths, epsilons, directory):
    """
    Guarda gráficos, Q-table y métricas por episodio.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Recompensas por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")

    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title("Duración de episodios")
    plt.xlabel("Episodio")
    plt.ylabel("Pasos")

    plt.tight_layout()
    plt.savefig(f"{directory}/training_results.png")
    plt.close()

    agent.save(directory)
    np.save(f"{directory}/rewards.npy", np.array(rewards))
    np.save(f"{directory}/lengths.npy", np.array(lengths))
    np.save(f"{directory}/epsilons.npy", np.array(epsilons))

    print(f"\n✅ Resultados guardados en: {directory}")

if __name__ == "__main__":
    config = {
        "env_name": "MountainCar-v0",
        "episodes": 2000,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "min_epsilon": 0.01
    }

    print("🚀 Iniciando entrenamiento...")
    print(f"Entorno: {config['env_name']}")
    print(f"Episodios: {config['episodes']}")
    print(f"Alpha: {config['alpha']}, Gamma: {config['gamma']}")
    print(f"Epsilon: {config['epsilon']} -> {config['min_epsilon']}")

    agent, rewards = train(**config)
