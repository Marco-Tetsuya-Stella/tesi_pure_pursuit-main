"""
Path Generator per Robot Mobile
Fornisce una suite di generatori geometrici per creare percorsi discreti (waypoint)
in formato numpy array (forma N x 2), pronti per essere visualizzati o seguiti.

Aggiunte per la tesi sul Pure Pursuit
"""

import numpy as np


class PathGenerator:
    """
    Classe contenitore per metodi statici che generano varie forme di traiettorie.
    Ogni metodo calcola matematicamente le coordinate dei waypoint.
    """

    @staticmethod
    def straight_line(start: tuple, end: tuple, num_points: int = 100) -> np.ndarray:
        """
        Genera un percorso rettilineo interpolando linearmente tra due punti.

        Args:
            start: Tupla (x, y) che rappresenta il punto di partenza.
            end: Tupla (x, y) che rappresenta il punto di arrivo.
            num_points: Numero totale di punti (waypoint) da generare lungo la retta.

        Returns:
            Array numpy di forma (num_points, 2) contenente la traiettoria di riferimento.
        """
        # np.linspace genera 'num_points' valori uniformemente distanziati tra l'inizio e la fine per l'asse X
        xs = np.linspace(start[0], end[0], num_points)

        # Stessa operazione per l'asse Y
        ys = np.linspace(start[1], end[1], num_points)

        # np.stack unisce gli array 1D delle X e delle Y mettendoli in colonne affiancate (axis=1)
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def circle(center: tuple, radius: float, num_points: int = 120, close_loop: bool = True) -> np.ndarray:
        """
        Genera un percorso circolare calcolando i punti lungo una circonferenza.

        Args:
            center: Tupla (x, y) indicante il centro del cerchio.
            radius: Raggio del cerchio espresso in metri.
            num_points: Numero di punti che compongono la circonferenza.
            close_loop: Booleano; se True, fa coincidere l'ultimo punto con il primo.

        Returns:
            Array numpy di forma (num_points, 2) rappresentante l'anello circolare.
        """
        # Se close_loop è False, endpoint è False per evitare di generare sia 0 che 2*pi (che coinciderebbero)
        endpoint = close_loop

        # Genera un array di angoli (in radianti) da 0 a 360 gradi (2*pigreco)
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=endpoint)

        # Calcola la coordinata X usando il coseno dell'angolo, scalato per il raggio e traslato dal centro
        xs = center[0] + radius * np.cos(thetas)

        # Calcola la coordinata Y usando il seno dell'angolo
        ys = center[1] + radius * np.sin(thetas)

        # Unisce le coordinate in una matrice Nx2
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def slalom(start_x: float, end_x: float, amplitude: float, wavelength: float, num_points: int = 150) -> np.ndarray:
        """
        Genera un percorso a forma di onda sinusoidale che si sviluppa lungo l'asse X.

        Args:
            start_x: Coordinata X di inizio dello slalom.
            end_x: Coordinata X di fine dello slalom.
            amplitude: Ampiezza dell'onda (quanto si sposta lateralmente sull'asse Y).
            wavelength: Lunghezza d'onda (distanza percorsa su X per completare un'oscillazione).
            num_points: Numero di punti da generare.

        Returns:
            Array numpy di forma (num_points, 2) che descrive lo slalom.
        """
        # Genera uniformemente le coordinate di avanzamento lungo l'asse X
        xs = np.linspace(start_x, end_x, num_points)

        # Calcola le coordinate Y applicando la funzione seno.
        # Si divide xs per wavelength per determinare quante onde intere ci stanno nel percorso.
        # Se dividiamo x per wavelength, scaliamo l'asse orizzontale in termini di "quante onde sono state completate
        # Quando x = wavelength}, il termine diventa 2*pi*1 = 2*pi, quindi la funzione sin(2\pi) ha completato esattamente 1 onda completa
        ys = amplitude * np.sin(2 * np.pi * xs / wavelength)

        # Impila le X e le Y in colonne
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def figure_eight(center: tuple, scale_a: float, scale_b: float, num_points: int = 200) -> np.ndarray:
        """
        Genera una traiettoria a forma di 8.

        Args:
            center: Tupla (x, y) che definisce il punto centrale dell'incrocio dell'otto.
            scale_a: Fattore di scala orizzontale (semi-ampiezza lungo X).
            scale_b: Fattore di scala verticale (semi-ampiezza lungo Y).
            num_points: Numero di punti da generare per il tracciato.

        Returns:
            Array numpy di forma (num_points, 2) rappresentante la figura a otto.
        """
        # Genera il parametro t (angolo) da 0 a 2*pigreco
        t = np.linspace(0, 2 * np.pi, num_points)

        # Coordinata X: usa un coseno semplice, che descrive un movimento da destra a sinistra e viceversa
        # la X fa un solo movimento avanti e indietro (parte dal massimo a destra, va tutto a sinistra e torna a destra)
        xs = center[0] + scale_a * np.cos(t)

        # Coordinata Y: usa un seno con frequenza doppia (2*t) diviso per 2, creando l'oscillazione su e giù
        # Usando 2t, la frequenza viene raddoppiata. Significa che mentre la X fa un solo viaggio avanti e indietro,
        # la Y fa due oscillazioni complete (sale, scende, va sotto e risale due volte).
        ys = center[1] + scale_b * np.sin(2 * t) / 2.0

        return np.stack([xs, ys], axis=1)

    @staticmethod
    def custom_polygon(vertices: list, points_per_segment: int = 30, close_loop: bool = True) -> np.ndarray:
        """
        Genera un percorso spezzato collegando linearmente una lista di vertici forniti.

        Args:
            vertices: Lista di tuple o coordinate [(x1, y1), (x2, y2), ...] dei vertici.
            points_per_segment: Numero di waypoint da interpolare per ogni lato del poligono.
            close_loop: Se True, aggiunge un segmento che riporta l'ultimo vertice al primo.

        Returns:
            Array numpy di forma (N, 2) contenente tutti i segmenti interpolati uniti.
        """
        # Controllo di sicurezza: serve un punto di inizio e uno di fine per fare un percorso
        if len(vertices) < 2:
            raise ValueError("Sono necessari almeno 2 vertici per generare un percorso.")

        # Copia la lista per non modificare i dati originali
        pts = list(vertices)

        # Se il loop va chiuso e l'ultimo punto non è già uguale al primo, duplica il primo punto alla fine
        if close_loop and pts[-1] != pts[0]:
            pts.append(pts[0])

        path_segments = []

        # Cicla attraverso tutte le coppie di vertici adiacenti
        for i in range(len(pts) - 1):
            p_start = pts[i]
            p_end = pts[i + 1]

            # Interpola i valori della X per questo specifico lato del poligono.
            # endpoint=False evita che l'ultimo punto di un lato si sovrapponga al primo punto del lato successivo.
            segment_x = np.linspace(p_start[0], p_end[0], points_per_segment, endpoint=False)

            # Interpola i valori della Y per lo stesso lato
            segment_y = np.linspace(p_start[1], p_end[1], points_per_segment, endpoint=False)

            # Salva i punti di questo lato nella lista dei segmenti
            path_segments.append(np.stack([segment_x, segment_y], axis=1))

        # Aggiunge esplicitamente l'ultimissimo punto per garantire che il traguardo (o la chiusura) sia esatto
        path_segments.append(np.array([pts[-1]]))

        # np.vstack concatena verticalmente tutte le matrici (tutti i lati) in un unico lungo array N x 2
        return np.vstack(path_segments)