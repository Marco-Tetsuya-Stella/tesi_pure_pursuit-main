"""
Path Generator per Robot Mobile
Fornisce una suite di generatori geometrici per creare percorsi discreti (waypoint)
in formato numpy array (forma N x 2), pronti per essere visualizzati o seguiti.

Aggiunte per la tesi sul Pure Pursuit
"""

import numpy as np


class PathGenerator:

    @staticmethod
    def straight_line(start: tuple, end: tuple, num_points: int = 100) -> np.ndarray:
        """
        Genera un percorso rettilineo tra un punto di partenza e uno di arrivo.

        :param start: Tupla (x, y) di partenza
        :param end: Tupla (x, y) di arrivo
        :param num_points: Numero di punti che compongono il percorso
        :return: Array numpy di forma (num_points, 2)
        """
        xs = np.linspace(start[0], end[0], num_points)
        ys = np.linspace(start[1], end[1], num_points)
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def circle(center: tuple, radius: float, num_points: int = 120, close_loop: bool = True) -> np.ndarray:
        """
        Genera un percorso circolare (anello).

        :param center: Tupla (x, y) del centro del cerchio
        :param radius: Raggio del cerchio in metri
        :param num_points: Numero di punti che compongono la circonferenza
        :param close_loop: Se True, l'ultimo punto coincide con il primo per chiudere il cerchio
        :return: Array numpy di forma (N, 2)
        """
        endpoint = not close_loop
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=endpoint)
        xs = center[0] + radius * np.cos(thetas)
        ys = center[1] + radius * np.sin(thetas)
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def slalom(start_x: float, end_x: float, amplitude: float, wavelength: float, num_points: int = 150) -> np.ndarray:
        """
        Genera un percorso sinusoidale (slalom) lungo l'asse X.

        :param start_x: Coordinata X iniziale
        :param end_x: Coordinata X finale
        :param amplitude: Ampiezza dell'onda (oscillazione su Y)
        :param wavelength: Lunghezza d'onda (distanza su X per compiere un ciclo completo)
        :param num_points: Numero di punti del percorso
        :return: Array numpy di forma (num_points, 2)
        """
        xs = np.linspace(start_x, end_x, num_points)
        ys = amplitude * np.sin(2 * np.pi * xs / wavelength)
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def figure_eight(center: tuple, scale_a: float, scale_b: float, num_points: int = 200) -> np.ndarray:
        """
        Genera una traiettoria a forma di 8 (lemniscata di Gerono).

        :param center: Tupla (x, y) del centro dell'otto
        :param scale_a: Scala orizzontale (semi-ampiezza X)
        :param scale_b: Scala verticale (semi-ampiezza Y)
        :param num_points: Numero di punti
        :return: Array numpy di forma (num_points, 2)
        """
        t = np.linspace(0, 2 * np.pi, num_points)
        xs = center[0] + scale_a * np.cos(t)
        ys = center[1] + scale_b * np.sin(2 * t) / 2.0
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def custom_polygon(vertices: list, points_per_segment: int = 30, close_loop: bool = True) -> np.ndarray:
        """
        Genera un percorso interpolando linearmente una lista di vertici (es. un quadrato o un circuito spezzato).

        :param vertices: Lista di tuple/punti [(x1, y1), (x2, y2), ...]
        :param points_per_segment: Quanti punti generare lungo ogni segmento del poligono
        :param close_loop: Se True, connette l'ultimo vertice al primo per chiudere il circuito
        :return: Array numpy di forma (N, 2)
        """
        if len(vertices) < 2:
            raise ValueError("Sono necessari almeno 2 vertici per generare un percorso.")

        pts = list(vertices)
        if close_loop and pts[-1] != pts[0]:
            pts.append(pts[0])

        path_segments = []
        for i in range(len(pts) - 1):
            p_start = pts[i]
            p_end = pts[i + 1]

            # Interpolazione per il segmento corrente
            segment_x = np.linspace(p_start[0], p_end[0], points_per_segment, endpoint=False)
            segment_y = np.linspace(p_start[1], p_end[1], points_per_segment, endpoint=False)
            path_segments.append(np.stack([segment_x, segment_y], axis=1))

        # Aggiunge l'ultimo punto esatto alla fine
        path_segments.append(np.array([pts[-1]]))
        return np.vstack(path_segments)