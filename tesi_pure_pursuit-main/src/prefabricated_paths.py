"""
Prefabricated Paths Registry
Fornisce un set di percorsi predefiniti (preset) pronti all'uso,
gestiti tramite un registro centralizzato.

Aggiunte per la tesi sul Pure Pursuit
"""

import numpy as np
import math
from path_generator import PathGenerator


class PrefabricatedPaths:

    @staticmethod
    def get_straight_short() -> np.ndarray:
        """Rettilineo breve di 5 metri."""
        return PathGenerator.straight_line(start=(0.0, 0.0), end=(5.0, 0.0), num_points=50)

    @staticmethod
    def get_straight_long() -> np.ndarray:
        """Rettilineo lungo di 20 metri."""
        return PathGenerator.straight_line(start=(0.0, 0.0), end=(20.0, 0.0), num_points=150)

    @staticmethod
    def get_circle_medium() -> np.ndarray:
        """Circonferenza standard con raggio di 3 metri centrata in (0,0)."""
        return PathGenerator.circle(center=(0.0, 0.0), radius=3.0, num_points=120)

    @staticmethod
    def get_circle_large() -> np.ndarray:
        """Circonferenza grande con raggio di 6 metri centrata in (0,0)."""
        return PathGenerator.circle(center=(0.0, 0.0), radius=6.0, num_points=200)

    @staticmethod
    def get_tight_slalom() -> np.ndarray:
        """Slalom stretto ad alta frequenza lungo l'asse X."""
        return PathGenerator.slalom(start_x=0.0, end_x=15.0, amplitude=0.8, wavelength=3.0, num_points=180)

    @staticmethod
    def get_wide_slalom() -> np.ndarray:
        """Slalom ampio e morbido lungo l'asse X."""
        return PathGenerator.slalom(start_x=0.0, end_x=20.0, amplitude=2.0, wavelength=8.0, num_points=150)

    @staticmethod
    def get_standard_eight() -> np.ndarray:
        """Traiettoria a 8 simmetrica e bilanciata."""
        return PathGenerator.figure_eight(center=(0.0, 0.0), scale_a=6.0, scale_b=3.0, num_points=200)

    @staticmethod
    def get_square_circuit() -> np.ndarray:
        """Circuito quadrato chiuso di 8x8 metri."""
        vertices = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
        return PathGenerator.custom_polygon(vertices, points_per_segment=35, close_loop=True)

    @staticmethod
    def get_f1_racetrack() -> np.ndarray:
        """Un circuito chiuso più complesso che simula una pista con curve miste."""
        vertices = [
            (0.0, 0.0),  # Partenza
            (10.0, 0.0),  # Rettilineo principale
            (12.0, 3.0),  # Curva veloce a destra
            (9.0, 6.0),  # Tornante a sinistra
            (5.0, 5.0),  # S-bend
            (2.0, 8.0),  # Allungo verso l'alto
            (-3.0, 4.0),  # Curva ampia di ritorno
        ]
        return PathGenerator.custom_polygon(vertices, points_per_segment=30, close_loop=True)

    @classmethod
    def get_preset(cls, name: str) -> np.ndarray:
        """
        Ritorna un percorso predefinito cercandolo per nome (case-insensitive).

        Esempio: get_preset("f1_racetrack")
        """
        presets = {
            "straight_short": cls.get_straight_short,
            "straight_long": cls.get_straight_long,
            "circle_medium": cls.get_circle_medium,
            "circle_large": cls.get_circle_large,
            "tight_slalom": cls.get_tight_slalom,
            "wide_slalom": cls.get_wide_slalom,
            "eight": cls.get_standard_eight,
            "square": cls.get_square_circuit,
            "pista_f1": cls.get_f1_racetrack
        }

        key = name.lower().strip()
        if key in presets:
            return presets[key]()
        else:
            raise ValueError(f"Preset '{name}' non trovato. Opzioni disponibili: {list(presets.keys())}")

    @classmethod
    def list_presets(cls) -> list:
        """Ritorna la lista dei nomi di tutti i preset disponibili."""
        return [
            "straight_short",
            "straight_long",
            "circle_medium",
            "circle_large",
            "tight_slalom",
            "wide_slalom",
            "eight",
            "square",
            "pista_f1",
        ]


# --- Utility helper ---

def convert_2d_path_to_3d_states(path_2d: np.ndarray) -> np.ndarray:
    """
    Utility per convertire un percorso 2D Nx2 [x, y] in pose 3D Nx3 [x, y, theta]
    calcolando l'orientamento corretto lungo i segmenti.
    """
    n = len(path_2d)
    states = np.zeros((n, 3), dtype=float)
    states[:, :2] = path_2d

    for i in range(n - 1):
        dx = path_2d[i + 1, 0] - path_2d[i, 0]
        dy = path_2d[i + 1, 1] - path_2d[i, 1]
        states[i, 2] = math.atan2(dy, dx)

    if n > 1:
        states[-1, 2] = states[-2, 2]

    return states