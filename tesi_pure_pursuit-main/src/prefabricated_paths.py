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
    """
    Classe contenitore che fornisce metodi statici per recuperare
    traiettorie pre-configurate. Agisce come un registro centralizzato.
    """

    @staticmethod
    def get_straight_short() -> np.ndarray:
        """
        Genera una traiettoria rettilinea breve.

        Returns:
            np.ndarray: Array Nx2 rappresentante un rettilineo di 5 metri.
        """
        # Richiama il generatore per fare una linea da X=0 a X=5 con 50 punti
        return PathGenerator.straight_line(start=(0.0, 0.0), end=(5.0, 0.0), num_points=50)

    @staticmethod
    def get_straight_long() -> np.ndarray:
        """
        Genera una traiettoria rettilinea lunga.

        Returns:
            np.ndarray: Array Nx2 rappresentante un rettilineo di 20 metri con 150 punti.
        """
        # Estende la linea fino a X=20 aumentando i punti per mantenere la risoluzione
        return PathGenerator.straight_line(start=(0.0, 0.0), end=(20.0, 0.0), num_points=150)

    @staticmethod
    def get_circle_medium() -> np.ndarray:
        """
        Genera una traiettoria circolare di medie dimensioni.

        Returns:
            np.ndarray: Array Nx2 rappresentante una circonferenza di raggio 3m.
        """
        # Crea un anello centrato nell'origine con raggio 3 metri
        return PathGenerator.circle(center=(0.0, 0.0), radius=3.0, num_points=120)

    @staticmethod
    def get_circle_large() -> np.ndarray:
        """
        Genera una traiettoria circolare ampia.

        Returns:
            np.ndarray: Array Nx2 rappresentante una circonferenza di raggio 6m.
        """
        # Crea un anello più grande (raggio 6) con più punti (200) per non perdere precisione nelle curve
        return PathGenerator.circle(center=(0.0, 0.0), radius=6.0, num_points=200)

    @staticmethod
    def get_tight_slalom() -> np.ndarray:
        """
        Genera un percorso a slalom stretto e frequente.

        Returns:
            np.ndarray: Array Nx2 con onda sinusoidale di piccola ampiezza e breve lunghezza d'onda.
        """
        # Slalom con curve strette (cambia direzione ogni 3 metri su X, spostandosi di 0.8 metri su Y)
        return PathGenerator.slalom(start_x=0.0, end_x=15.0, amplitude=0.8, wavelength=3.0, num_points=180)

    @staticmethod
    def get_wide_slalom() -> np.ndarray:
        """
        Genera un percorso a slalom largo e morbido.

        Returns:
            np.ndarray: Array Nx2 con onda sinusoidale ampia e distesa.
        """
        # Slalom molto più morbido (cambia direzione ogni 8 metri, allargandosi fino a 2 metri su Y)
        return PathGenerator.slalom(start_x=0.0, end_x=20.0, amplitude=2.0, wavelength=8.0, num_points=150)

    @staticmethod
    def get_standard_eight() -> np.ndarray:
        """
        Genera una traiettoria a forma di 8 (lemniscata).

        Returns:
            np.ndarray: Array Nx2 simmetrico e bilanciato a forma di otto.
        """
        # Usa i fattori di scala 6.0 e 3.0 per creare un otto ben proporzionato centrato in (0,0)
        return PathGenerator.figure_eight(center=(0.0, 0.0), scale_a=6.0, scale_b=3.0, num_points=200)

    @staticmethod
    def get_square_circuit() -> np.ndarray:
        """
        Genera un circuito a forma di quadrato.

        Returns:
            np.ndarray: Array Nx2 di un poligono chiuso di 8x8 metri.
        """
        # Definisce i 4 angoli del quadrato
        vertices = [(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)]
        # custom_polygon unisce i vertici interpolando 35 punti per lato e chiude il giro (close_loop=True)
        return PathGenerator.custom_polygon(vertices, points_per_segment=35, close_loop=True)

    @staticmethod
    def get_f1_racetrack() -> np.ndarray:
        """
        Genera un circuito chiuso asimmetrico e complesso.

        Returns:
            np.ndarray: Array Nx2 interpolato su una serie di vertici irregolari.
        """
        # Lista di waypoint chiave che simulano curve e rettilinei di una pista
        vertices = [
            (0.0, 0.0),  # Partenza
            (10.0, 0.0),  # Rettilineo
            (12.0, 3.0),  # Curva a sinistra parte 1
            (9.0, 6.0),  # Curva a sinistra parte 2
            (5.0, 5.0),  # Curva a sinistra parte 3
            (2.0, 8.0),  # Curva a destra
            (-3.0, 4.0),  # Curva a sinistra
        ]
        # Interpola 30 waypoint tra ogni coppia di vertici per rendere il percorso continuo
        return PathGenerator.custom_polygon(vertices, points_per_segment=30, close_loop=True)

    @classmethod
    def get_preset(cls, name: str) -> np.ndarray:
        """
        Ritorna un percorso predefinito in base al nome fornito.

        Args:
            name: Stringa col nome del preset (es. "straight_short").
                  Il check è case-insensitive.

        Returns:
            np.ndarray: Il percorso richiesto sotto forma di matrice Nx2.

        Raises:
            ValueError: Se il nome inserito non è nel dizionario dei preset.
        """
        # Dizionario che mappa la stringa testuale alla funzione associata che genera il percorso
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

        # Pulisce la stringa (rimuove spazi vuoti e mette in minuscolo) per evitare errori di battitura
        key = name.lower().strip()

        # Se il nome esiste tra le chiavi del dizionario, esegue la funzione corrispondente
        if key in presets:
            return presets[key]()
        else:
            # Se l'utente chiede un tracciato inesistente, il programma si ferma mostrando le opzioni valide
            raise ValueError(f"Preset '{name}' non trovato. Opzioni disponibili: {list(presets.keys())}")

    @classmethod
    def list_presets(cls) -> list:
        """
        Restituisce l'elenco dei percorsi disponibili.

        Returns:
            list: Lista di stringhe contenenti i nomi validi da passare a get_preset().
        """
        # Ritorna semplicemente l'elenco hardware-coded dei nomi validi
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
    Converte un percorso (X, Y) in una serie di pose (X, Y, Theta).
    L'orientamento (Theta) è calcolato guardando la direzione del segmento successivo.
    Theta comunica al robot come deve orientarsi per seguire la traiettoria
    Theta è espresso in radianti.

    Args:
        path_2d: array Nx2 contenente le sole coordinate spaziali.

    Returns:
        np.ndarray: Array Nx3 con [X, Y, Theta] (pose complete) per ogni waypoint.
    """
    # Ottiene il numero totale di punti nel percorso
    n = len(path_2d)

    # Crea una nuova matrice vuota di N righe e 3 colonne, inizializzata a zero
    states = np.zeros((n, 3), dtype=float)

    # Ricopia esattamente le colonne X e Y dalla matrice 2D alla nuova matrice 3D (lasciando la terza colonna a 0)
    states[:, :2] = path_2d

    # Cicla su tutti i punti tranne l'ultimo
    for i in range(n - 1):
        # Calcola la variazione su asse X (delta X) tra il punto corrente e il successivo
        dx = path_2d[i + 1, 0] - path_2d[i, 0]
        # Calcola la variazione su asse Y (delta Y) tra il punto corrente e il successivo
        dy = path_2d[i + 1, 1] - path_2d[i, 1]

        # Usa math.atan2(dy, dx) per trovare l'angolo in radianti (Theta) del vettore che unisce i due punti.
        # Lo salva nella terza colonna dell'array states.
        states[i, 2] = math.atan2(dy, dx)

    # L'ultimo punto del percorso non ha un "punto successivo" verso cui guardare.
    # Quindi, copia semplicemente l'orientamento del penultimo punto.
    if n > 1:
        states[-1, 2] = states[-2, 2]

    return states