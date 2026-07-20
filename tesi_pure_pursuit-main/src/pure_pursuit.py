"""
 Controller Pure Pursuit: calcola i comandi (v, omega) per far seguire al robot
 un path (traiettoria) di riferimento, usando un punto "lookahead" e la relativa
 formula di curvatura in coordinate locali del robot.
 Aggiunto per tesi per il pure pursuit
"""
import numpy as np


class PurePursuitController:
    """
    Implementa l'algoritmo di controllo Pure Pursuit per l'inseguimento di traiettorie.
    """

    def __init__(
            self,
            lookahead_distance: float = 0.5,
            max_angular_velocity: float = 2.0,
            target_linear_velocity: float = 0.4,
            stop_tolerance: float = 0.05,  # Tolleranza fisica (es. 5 cm)
            max_index_gap: int = 50  # NUOVO: Tolleranza logica (n. punti)
    ):
        """
        Inizializza i parametri operativi del controllore Pure Pursuit.

        Args:
            lookahead_distance: Raggio (m) entro cui il robot cerca il punto di mira sul percorso.
            max_angular_velocity: Limite massimo per la velocità di rotazione del robot.
            target_linear_velocity: Velocità di avanzamento costante desiderata per il robot.
            stop_tolerance: Distanza euclidea finale (m) sotto la quale il traguardo è considerato raggiunto.
            max_index_gap: Numero di waypoint massimi tollerati tra l'indice attuale e la fine del percorso
                           per confermare l'arrivo al traguardo.
        """
        self.L_d = lookahead_distance
        self.max_omega = max_angular_velocity
        self.target_v = target_linear_velocity

        # Indice di progresso lungo il path: serve a non cercare punti del percorso già superati
        self._last_idx = 0

        # Gestione interruzione[cite: 13]
        self.stop_tolerance = stop_tolerance
        self.max_index_gap = max_index_gap

    def reset(self) -> None:
        """
        Resetta il progresso lungo il path (da chiamare se si cambia percorso).

        Returns:
            Nessun valore restituito, azzera semplicemente l'indice interno.
        """
        self._last_idx = 0

    def find_lookahead_point(
            self, robot_pose: np.ndarray, path: np.ndarray
    ) -> np.ndarray:
        """
        Trova il goal point calcolando l'intersezione geometrica esatta tra la
        circonferenza di raggio L_d centrata sul robot e ciascun segmento del path.

        Args:
            robot_pose: Array [x, y, theta] con la posa globale attuale del robot.
            path: Array Nx2 o Nx3 contenente i waypoint della traiettoria da seguire.

        Returns:
            np.ndarray: Coordinate 2D [x, y] del punto di lookahead ottimale trovato.
        """
        # Estrae le coordinate del robot
        x_r, y_r, _ = robot_pose
        current_pos = np.array([x_r, y_r], dtype=float)
        n = len(path)
        L_d = self.L_d

        # Se il percorso ha meno di 2 punti, restituisce semplicemente l'ultimo punto disponibile
        if n < 2:
            return path[-1, :2]

        goal_pt = None
        # Inizia a cercare partendo dall'ultimo indice visitato, per non guardare indietro
        start_index = min(self._last_idx, n - 2)

        # Scorre i segmenti del percorso
        for i in range(start_index, n - 1):
            # Trasla il segmento di percorso ponendo il robot all'origine (0,0) per facilitare i calcoli
            p1 = path[i, :2] - current_pos
            p2 = path[i + 1, :2] - current_pos


            # Formule per l'intersezione Retta-Circonferenza
            # circonferenza centrata nell'origine con raggio r = Ld:
            # x^2 + y^2 = L_d^2
            # retta passante per due punti (x1, y1) e (x2, y2):
            # Ax + By + C = 0
            # dx = x2 - x1 e dy = y2 - y1: Rappresentano le componenti del vettore direzione del segmento.
            # dr^2 = dx^2 + dy^2: È la distanza al quadrato tra il punto P1 e il punto P2
            # D = x1*y2 - x2*y1: È il determinante della matrice delle coordinate dei due punti.
            # Discriminante (discriminant = L_d^2 * dr^2 - D^2):
            #   Se discriminant < 0: La retta è esterna al cerchio (nessuna intersezione).
            #   Se discriminant = 0: La retta è tangente al cerchio (1 punto di intersezione).
            #   Se discriminant > 0: La retta è secante (2 punti di intersezione).
            # sgn_dy: Gestisce il segno della coordinata y per garantire la corretta disposizione dei segni
            # nella soluzione algebrica parametrica.
            # formule complete generano direttamente le coordinate (x, y) dei due punti di intersezione della retta

            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1

            # Calcola il quadrato della lunghezza del segmento di percorso corrente
            dr2 = dx * dx + dy * dy

            # Se i due punti del segmento coincidono (distanza zero), salta al prossimo
            if dr2 < 1e-12:
                continue

            # Applica la formula matematica dell'intersezione retta-circonferenza
            D = x1 * y2 - x2 * y1
            discriminant = (L_d ** 2) * dr2 - D ** 2

            # Se il discriminante è negativo, il cerchio di raggio L_d non interseca questo segmento
            if discriminant < 0:
                continue

            # Se c'è intersezione, calcola le possibili soluzioni (fino a due punti di intersezione)
            sqrt_disc = np.sqrt(discriminant)
            sgn_dy = 1.0 if dy >= 0 else -1.0

            # Soluzioni per la prima intersezione
            sol_x1 = (D * dy + sgn_dy * dx * sqrt_disc) / dr2
            sol_y1 = (-D * dx + abs(dy) * sqrt_disc) / dr2

            # Soluzioni per la seconda intersezione
            sol_x2 = (D * dy - sgn_dy * dx * sqrt_disc) / dr2
            sol_y2 = (-D * dx - abs(dy) * sqrt_disc) / dr2

            # Ri-trasla le soluzioni nelle coordinate globali originali
            sol_pt1 = np.array([sol_x1, sol_y1]) + current_pos
            sol_pt2 = np.array([sol_x2, sol_y2]) + current_pos

            # Determina i confini del segmento di retta originario
            minX, maxX = sorted((path[i, 0], path[i + 1, 0]))
            minY, maxY = sorted((path[i, 1], path[i + 1, 1]))
            eps = 1e-9

            # NOTA: Una retta è infinita, mentre il segmento del percorso ha un inizio (P1) e una fine (P2).
            # Le equazioni algebriche trovano le intersezioni con la retta infinita. Successivamente,
            # le condizioni pt1_valid e pt2_valid controllano se tali intersezioni cadono effettivamente
            # all'interno dei confini fisici del segmento.

            # Verifica che il punto di intersezione cada effettivamente DENTRO i limiti del segmento
            pt1_valid = (minX - eps <= sol_pt1[0] <= maxX + eps) and (minY - eps <= sol_pt1[1] <= maxY + eps)
            pt2_valid = (minX - eps <= sol_pt2[0] <= maxX + eps) and (minY - eps <= sol_pt2[1] <= maxY + eps)

            # Se nessuna intersezione è sul segmento, passa oltre
            if not (pt1_valid or pt2_valid):
                continue

            # Se entrambe le intersezioni sono valide, sceglie quella più vicina al punto finale
            # del segmento (per procedere in avanti)
            next_pt = path[i + 1, :2]
            if pt1_valid and pt2_valid:
                if np.linalg.norm(sol_pt1 - next_pt) < np.linalg.norm(sol_pt2 - next_pt):
                    candidate = sol_pt1
                else:
                    candidate = sol_pt2
            elif pt1_valid:
                candidate = sol_pt1
            else:
                candidate = sol_pt2

            # Assicura che il punto scelto sia effettivamente più avanti rispetto alla posizione attuale
            if np.linalg.norm(candidate - next_pt) <= np.linalg.norm(current_pos - next_pt):
                goal_pt = candidate
                # Aggiorna l'indice così al prossimo ciclo partirà da qui
                self._last_idx = i
                break
            else:
                self._last_idx = i + 1

        # Fallback: se nessuna intersezione viene trovata, punta semplicemente all'ultimo punto visitato valido
        if goal_pt is None:
            fallback_idx = min(self._last_idx, n - 1)
            goal_pt = path[fallback_idx, :2].copy()

        return goal_pt

    def compute_commands(
            self, robot_pose: np.ndarray, path: np.ndarray
    ) -> tuple[float, float]:
        """
        Calcola i comandi cinematici per il robot al fine di raggiungere il lookahead point.

        Args:
            robot_pose: Posizione e orientamento attuali del robot [x, y, theta].
            path: L'intera matrice del percorso da seguire.

        Returns:
            Una tupla (v, omega) dove v è la velocità lineare e omega la velocità angolare.
        """
        x_r, y_r, theta_r = robot_pose

        # Controllo di arrivo a destinazione basato sugli indici
        # Recupera l'ultimo punto del tracciato per usarlo come traguardo
        final_point = path[-1, :2]
        # Calcola la distanza fisica euclidea tra il robot e il traguardo
        distance_to_goal = np.hypot(final_point[0] - x_r, final_point[1] - y_r)

        # Calcoliamo quanti punti "logici" mancano alla fine del tracciato rispetto al nostro progresso
        index_distance_to_end = (len(path) - 1) - self._last_idx

        # Se la distanza fisica è inferiore alla soglia E siamo vicini alla fine dell'array (es. ultimi 30 punti)
        # OPPURE se l'indice è rimasto bloccato vicinissimo alla fine (ultimi 3 punti dell'array)
        if distance_to_goal < self.stop_tolerance:
            if index_distance_to_end < self.max_index_gap:
                # Il robot si ferma inviando velocità zero
                return 0.0, 0.0
        # --------------------------------------------------------------------------

        # Richiama la funzione geometrica per trovare il punto di mira
        lookahead_pt = self.find_lookahead_point(robot_pose, path)

        # NOTA: (dx, dy): È il vettore posizione del punto di mira rispetto al robot nel sistema
        # di riferimento globale del enviroment
        # Matrice di Rotazione Inversa: Ruota il vettore (dx, dy) dell'angolo del robot
        # theta_r per portarlo nel sistema di riferimento locale del robot:
        #   local_x: Distanza del punto di mira davanti al robot (asse longitudinale).
        #   local_y: Distanza del punto di mira a sinistra/destra del robot (asse trasversale).

        # Calcola la differenza (delta) globale tra il punto di mira e il robot
        dx = lookahead_pt[0] - x_r
        dy = lookahead_pt[1] - y_r

        # Trasforma i delta globali (dx, dy) nel sistema di coordinate locale del robot tramite matrice
        # di rotazione inversa
        local_x = np.cos(theta_r) * dx + np.sin(theta_r) * dy
        local_y = -np.sin(theta_r) * dx + np.cos(theta_r) * dy

        # NOTA: Se il punto si trova dietro al robot (local_x negativo),
        # Poiché il Pure Pursuit guida solo in avanti, l'algoritmo forza una rotazione alla massima velocità
        # angolare (max_omega) nella direzione del punto np.sign(local_y).
        # il robot ruota sul posto per rimettersi in direzione
        if local_x < 0:
            v = self.target_v
            omega = np.sign(local_y) * self.max_omega
            return v, omega

        # Calcola la distanza esatta del lookahead nel sistema locale per prevenire divisioni per zero
        actual_L_d = np.hypot(local_x, local_y)
        if actual_L_d < 0.001:
            actual_L_d = 0.001

        # NOTA: Equazione fondamentale del Pure Pursuit.
        # Geometricamente, l'obiettivo è trovare un arco di cerchio che parta dal centro del robot
        # (tangente al suo asse di avanzamento) e arrivi esattamente al punto di mira (x, y)_locale avente una distanza
        # dal robot pari a L_d.
        # Dalla geometria del cerchio, la curvatura k (che è il reciproco del raggio del cerchio k = 1/R) è legata
        # alla posizione locale da:
        #   k = (2 * y_locale)/(L_d^2)
        # actual_L_d è calcolato tramite la distanza euclidea reale: sqrt{x_locale^2 + y_locale^2}
        # Poiché nel moto dei robot la velocità angolare è definita come omega = v*k, sostituendo la curvatura otteniamo:
        # omega = v * (2 * y_locale)/(L_d^2)
        #   Se il punto è a sinistra (y_locale > 0), omega è positiva e il robot curva a sinistra.
        #   Se il punto è a destra (y_locale < 0), omega è negativa e il robot curva a destra.
        #   Se il punto è perfettamente allineato (y_locale = 0), la curvatura è zero e il robot procede dritto.


        # Formula centrale del Pure Pursuit: calcola la curvatura dell'arco che unisce il robot al punto
        curvature = (2.0 * local_y) / (actual_L_d ** 2)

        # La velocità lineare è mantenuta costante, mentre quella angolare si calcola moltiplicando v per la curvatura
        v = self.target_v
        # La velocità angolare viene tagliata ("clippata") affinché non superi il limite fisico imposto
        omega = np.clip(v * curvature, -self.max_omega, self.max_omega)

        return v, omega