import numpy as np
import random
from shapely.geometry import LineString, Point, Polygon as ShapelyPolygon
from environment import Environment
from prefabricated_paths import PrefabricatedPaths


class SafeEnvironmentGenerator:
    @staticmethod
    def generate_safe_env_for_path(
            path: np.ndarray,
            clearance: float = 0.8,
            num_obstacles: int = 25,
            bounds_pad: float = 2.0
    ) -> Environment:
        """
        Genera un ambiente in cui gli ostacoli sono garantiti per NON intersecare il percorso.

        :param path: Array numpy Nx2 o Nx3 che rappresenta la traiettoria.
        :param clearance: Distanza di sicurezza (in metri) dal percorso. Nessun ostacolo entrerà in quest'area.
        :param num_obstacles: Numero di tentativi di piazzamento ostacoli.
        :param bounds_pad: Spazio extra oltre i limiti della traiettoria per i bounds dell'ambiente.
        """
        env = Environment()

        # 1. Calcola i bounds dell'ambiente in base all'estensione del percorso
        path_xy = path[:, :2]
        x_min, y_min = np.min(path_xy[:, 0]), np.min(path_xy[:, 1])
        x_max, y_max = np.max(path_xy[:, 0]), np.max(path_xy[:, 1])

        b_left = x_min - bounds_pad
        b_bottom = y_min - bounds_pad
        b_right = x_max + bounds_pad
        b_top = y_max + bounds_pad

        env.set_bounds(b_left, b_bottom, b_right, b_top)

        # 2. Crea la "Safe Zone" (Corridoio di sicurezza) usando Shapely
        path_line = LineString(path_xy.tolist())
        safe_buffer = path_line.buffer(clearance, cap_style='flat', join_style='round')

        # 3. Funzione di supporto per verificare le collisioni
        def is_safe(geom) -> bool:
            # Verifica che sia dentro i limiti
            try:
                if not env.bounds.contains(geom):
                    return False
            except:
                return False

            # Verifica che NON intersechi il corridoio di sicurezza
            if geom.intersects(safe_buffer):
                return False

            # Verifica che non si sovrapponga ad altri ostacoli già inseriti
            for obs in env.obstacles:
                if geom.intersects(obs):
                    return False

            return True

        # 4. Generazione ostacoli
        attempts = 0
        max_attempts = num_obstacles * 10  # Limita i tentativi per evitare loop infiniti
        placed = 0

        while placed < num_obstacles and attempts < max_attempts:
            attempts += 1

            # Genera coordinate casuali all'interno dei bounds
            cx = random.uniform(b_left + 0.5, b_right - 0.5)
            cy = random.uniform(b_bottom + 0.5, b_top - 0.5)

            # Scegli casualmente il tipo di ostacolo (Cerchio o Rettangolo/Poligono)
            obs_type = random.choice(['circle', 'rectangle'])

            if obs_type == 'circle':
                radius = random.uniform(0.15, 0.5)
                geom = Point(cx, cy).buffer(radius, resolution=16)
                if is_safe(geom):
                    env.add_circle(cx, cy, radius)
                    placed += 1

            elif obs_type == 'rectangle':
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)
                # Crea un rettangolo usando Polygon di Shapely per la verifica
                geom = ShapelyPolygon([
                    (cx - w / 2, cy - h / 2),
                    (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2),
                    (cx - w / 2, cy + h / 2)
                ])
                if is_safe(geom):
                    env.add_rectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    placed += 1

        return env

    @staticmethod
    def generate_all_safe_environments(clearance: float = 0.8, num_obstacles: int = 25) -> None:
        """
        Genera e mostra in una griglia gli ambienti sicuri per tutti i tipi di path prefabbricati disponibili.
        """
        import matplotlib.pyplot as plt

        presets = PrefabricatedPaths.list_presets()
        num_presets = len(presets)

        cols = 3
        rows = (num_presets + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for i, preset_name in enumerate(presets):
            ax = axes[i]

            # Recupera la traiettoria corrente
            path = PrefabricatedPaths.get_preset(preset_name)

            # Genera l'ambiente protetto
            safe_env = SafeEnvironmentGenerator.generate_safe_env_for_path(
                path=path,
                clearance=clearance,
                num_obstacles=num_obstacles
            )

            # Disegna mappa e traiettoria
            safe_env.plot(ax=ax)
            ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, label="Percorso")

            ax.set_title(f"Preset: {preset_name}", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_aspect('equal', 'box')

        # Elimina i subplot vuoti in eccesso
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Genera e mostra contemporaneamente gli ambienti di sicurezza per tutti i tracciati
    SafeEnvironmentGenerator.generate_all_safe_environments(clearance=0.8, num_obstacles=20)