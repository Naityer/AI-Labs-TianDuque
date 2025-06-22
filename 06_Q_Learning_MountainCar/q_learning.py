import numpy as np
import json
from pathlib import Path

class QLearningAgent:
    """
    Agente Q-Learning para entornos de espacio continuo discretizado.
    Compatible con Gymnasium. Usa Q-table basada en bins fijos.
    """
    def __init__(self, env, state_bins=None, alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.state_bins = state_bins or self._default_bins(env)
        self.q_table = np.zeros((*self.state_bins, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def _default_bins(self, env):
        """Define discretización por defecto para algunos entornos conocidos."""
        if 'CartPole' in env.spec.id:
            return [10, 10, 10, 10]
        elif 'MountainCar' in env.spec.id:
            return [20, 20]
        return [10] * env.observation_space.shape[0]

    def discretize(self, state):
        """
        Convierte un estado continuo en índices discretos según los bins definidos.
        """
        state = np.clip(state, self.env.observation_space.low, self.env.observation_space.high)
        return tuple(
            np.digitize(
                x,
                np.linspace(self.env.observation_space.low[i], self.env.observation_space.high[i], self.state_bins[i] + 1)
            ) - 1
            for i, x in enumerate(state)
        )

    def choose_action(self, state):
        """
        Política ε-greedy: elige acción aleatoria con probabilidad ε, si no, la mejor acción conocida.
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, new_state, done):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning.
        """
        best_next = np.max(self.q_table[new_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Reduce ε gradualmente hasta un mínimo."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def shaped_reward(self, state, reward):
        """
        Recompensa modificada para MountainCar: premia velocidad y cercanía a la cima.
        """
        if "MountainCar" not in self.env.spec.id:
            return reward

        pos, vel = state
        if pos >= 0.5:
            print("✅ ¡Cima alcanzada!")
            reward += 100
        if pos >= 0.45:
            reward += 20
        if vel > 0:
            reward += 5 * vel
        return reward

    def save(self, path):
        """Guarda Q-table y parámetros en disco."""
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(f"{path}/q_table.npy", self.q_table)
        with open(f"{path}/params.json", 'w') as f:
            json.dump({
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'state_bins': self.state_bins
            }, f)

    def load(self, path):
        """Carga Q-table desde archivo."""
        self.q_table = np.load(f"{path}/q_table.npy")
