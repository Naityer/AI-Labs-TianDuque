import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def timestamp():
    """
    Devuelve un string con la fecha y hora actual en formato YYYYMMDD_HHMMSS.
    Útil para nombrar carpetas de logs o resultados.
    """
    return time.strftime("%Y%m%d_%H%M%S")

def setup_logging(env_name):
    """
    Crea una carpeta de resultados con timestamp para el entorno dado.
    Ejemplo: results/MountainCar-v0_20240422_153000
    """
    log_dir = Path(f"results/{env_name}_{timestamp()}")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def save_heatmap(q_table, path):
    """
    Guarda una imagen de la Q-table en forma de heatmap 2D.
    Útil para visualizar la mejor acción por estado (solo si estado es 2D).
    """
    if q_table.ndim != 3:
        print("⚠️ Q-table no visualizable como heatmap (no es 2D)")
        return

    heatmap = np.max(q_table, axis=-1)  # Mejor valor por estado
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.title("Heatmap de Q-table (valor máximo por estado)")
    plt.xlabel("Velocidad (discretizada)")
    plt.ylabel("Posición (discretizada)")
    plt.colorbar(label='Valor Q')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"📸 Heatmap guardado en: {path}")

def save_json(data, path):
    """
    Guarda un diccionario como archivo JSON formateado.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def save_summary(directory, config, rewards, epsilons, elapsed_seconds):
    """
    Guarda un archivo resumen con:
    - Configuración usada
    - Recompensas
    - Métricas clave (media, éxito)
    - Tiempo total de entrenamiento
    """
    summary = {
        "env": config.get("env_name"),
        "episodes": config.get("episodes"),
        "alpha": config.get("alpha"),
        "gamma": config.get("gamma"),
        "epsilon_start": config.get("epsilon"),
        "epsilon_decay": config.get("epsilon_decay"),
        "epsilon_min": config.get("min_epsilon"),
        "avg_reward_last_100": float(np.mean(rewards[-100:])),
        "max_reward": float(np.max(rewards)),
        "success": bool(np.mean(rewards[-100:]) > -110),
        "training_time_seconds": round(elapsed_seconds, 2)
    }

    save_json(summary, f"{directory}/summary.json")
    np.save(f"{directory}/epsilons.npy", np.array(epsilons))

def save_milestones(milestones, directory):
    """
    Guarda los hitos importantes de la ejecución en un archivo JSON:
    - Episodios donde se alcanzó la cima
    - Estadísticas resumidas del episodio (reward, epsilon, etc.)
    """
    path = Path(directory) / "milestones.json"
    save_json(milestones, path)
    print(f"📍 Milestones guardados en: {path}")
