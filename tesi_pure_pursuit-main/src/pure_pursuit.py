# Controller Pure Pursuit: calcola i comandi (v, omega) per far seguire al robot
# un path (traiettoria) di riferimento, usando un punto "lookahead" e la relativa
# formula di curvatura in coordinate locali del robot.

import numpy as np


class PurePursuitController:

    def __init__(
            self,
            lookahead_distance: float = 0.5,
            max_angular_velocity: float = 2.0,
            target_linear_velocity: float = 0.4,
            stop_tolerance: float = 0.05,  # Tolleranza fisica (es. 5 cm)
            max_index_gap: int = 50  # NUOVO: Tolleranza logica (n. punti)
    ):
        self.L_d = lookahead_distance
        self.max_omega = max_angular_velocity
        self.target_v = target_linear_velocity

        # Indice di progresso lungo il path
        self._last_idx = 0

        # Gestione interruzione
        self.stop_tolerance = stop_tolerance
        self.max_index_gap = max_index_gap

    def reset(self) -> None:
        """Resetta il progresso lungo il path (da chiamare se si cambia percorso)."""
        self._last_idx = 0

    def find_lookahead_point(
            self, robot_pose: np.ndarray, path: np.ndarray
    ) -> np.ndarray:
        """
        Trova il goal point calcolando l'intersezione geometrica esatta tra la
        circonferenza di raggio L_d centrata sul robot e ciascun segmento del path.
        """
        x_r, y_r, _ = robot_pose
        current_pos = np.array([x_r, y_r], dtype=float)
        n = len(path)
        L_d = self.L_d

        if n < 2:
            return path[-1, :2]

        goal_pt = None
        start_index = min(self._last_idx, n - 2)

        for i in range(start_index, n - 1):
            p1 = path[i, :2] - current_pos
            p2 = path[i + 1, :2] - current_pos
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            dr2 = dx * dx + dy * dy

            if dr2 < 1e-12:
                continue

            D = x1 * y2 - x2 * y1
            discriminant = (L_d ** 2) * dr2 - D ** 2

            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            sgn_dy = 1.0 if dy >= 0 else -1.0

            sol_x1 = (D * dy + sgn_dy * dx * sqrt_disc) / dr2
            sol_x2 = (D * dy - sgn_dy * dx * sqrt_disc) / dr2
            sol_y1 = (-D * dx + abs(dy) * sqrt_disc) / dr2
            sol_y2 = (-D * dx - abs(dy) * sqrt_disc) / dr2

            sol_pt1 = np.array([sol_x1, sol_y1]) + current_pos
            sol_pt2 = np.array([sol_x2, sol_y2]) + current_pos

            minX, maxX = sorted((path[i, 0], path[i + 1, 0]))
            minY, maxY = sorted((path[i, 1], path[i + 1, 1]))
            eps = 1e-9

            pt1_valid = (minX - eps <= sol_pt1[0] <= maxX + eps) and (minY - eps <= sol_pt1[1] <= maxY + eps)
            pt2_valid = (minX - eps <= sol_pt2[0] <= maxX + eps) and (minY - eps <= sol_pt2[1] <= maxY + eps)

            if not (pt1_valid or pt2_valid):
                continue

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

            if np.linalg.norm(candidate - next_pt) <= np.linalg.norm(current_pos - next_pt):
                goal_pt = candidate
                self._last_idx = i
                break
            else:
                self._last_idx = i + 1

        if goal_pt is None:
            fallback_idx = min(self._last_idx, n - 1)
            goal_pt = path[fallback_idx, :2].copy()

        return goal_pt

    def compute_commands(
            self, robot_pose: np.ndarray, path: np.ndarray
    ) -> tuple[float, float]:

        x_r, y_r, theta_r = robot_pose

        # --- MODIFICATO: Controllo di arrivo a destinazione basato sugli indici ---
        final_point = path[-1, :2]
        distance_to_goal = np.hypot(final_point[0] - x_r, final_point[1] - y_r)

        # Calcoliamo quanti punti "logici" mancano alla fine del tracciato
        index_distance_to_end = (len(path) - 1) - self._last_idx

        # Se la distanza fisica è < 5 cm E siamo vicini alla fine dell'array (es. ultimi 30 punti)
        # OPPURE se l'indice è rimasto bloccato vicinissimo alla fine (ultimi 3 punti dell'array)
        if distance_to_goal < self.stop_tolerance:
            if index_distance_to_end < self.max_index_gap:
                return 0.0, 0.0
        # --------------------------------------------------------------------------

        lookahead_pt = self.find_lookahead_point(robot_pose, path)

        dx = lookahead_pt[0] - x_r
        dy = lookahead_pt[1] - y_r

        local_x = np.cos(theta_r) * dx + np.sin(theta_r) * dy
        local_y = -np.sin(theta_r) * dx + np.cos(theta_r) * dy

        if local_x < 0:
            v = self.target_v
            omega = np.sign(local_y) * self.max_omega
            return v, omega

        actual_L_d = np.hypot(local_x, local_y)
        if actual_L_d < 0.001:
            actual_L_d = 0.001

        curvature = (2.0 * local_y) / (actual_L_d ** 2)

        v = self.target_v
        omega = np.clip(v * curvature, -self.max_omega, self.max_omega)

        return v, omega