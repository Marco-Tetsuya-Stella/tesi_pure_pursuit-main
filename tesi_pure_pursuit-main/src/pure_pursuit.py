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
    ):
        self.L_d = lookahead_distance
        self.max_omega = max_angular_velocity
        self.target_v = target_linear_velocity

    def find_lookahead_point(
        self, robot_pose: np.ndarray, path: np.ndarray
    ) -> np.ndarray:
        x_r, y_r, _ = robot_pose

        # Calcola le distanze di tutti i punti dal robot
        distances = np.linalg.norm(path[:, :2] - np.array([x_r, y_r]), axis=1)

        # Trova l'indice del punto più vicino al robot
        closest_idx = np.argmin(distances)

        # Scorre in avanti partendo dal punto più vicino
        for i in range(closest_idx, len(path)):
            # Cerchiamo il punto che si avvicina di più alla distanza L_d
            if distances[i] >= self.L_d:
                return path[i, :2]

        # Se nessun punto successivo è abbastanza lontano, restituisce l'ultimo
        return path[-1, :2]

    def compute_commands(
        self, robot_pose: np.ndarray, path: np.ndarray
    ) -> tuple[float, float]:
        x_r, y_r, theta_r = robot_pose
        lookahead_pt = self.find_lookahead_point(robot_pose, path)

        dx = lookahead_pt[0] - x_r
        dy = lookahead_pt[1] - y_r

        # Trasforma il punto lookahead nel frame locale del robot
        local_x = np.cos(theta_r) * dx + np.sin(theta_r) * dy
        local_y = -np.sin(theta_r) * dx + np.cos(theta_r) * dy

        # GESTIONE CASO CRITICO: Il punto è dietro al robot
        if local_x < 0:
            # Il punto è alle spalle. Sterza al massimo verso il lato del punto
            v = self.target_v
            omega = np.sign(local_y) * self.max_omega
            return v, omega

        # Ricalcola la vera distanza geometrica locale (per evitare approssimazioni)
        # della Look-Ahead effettiva, nel caso in cui non abbiamo un punto perfetto a L_d
        actual_L_d = np.hypot(local_x, local_y)
        if actual_L_d < 0.001:
            actual_L_d = 0.001

        # Formula di curvatura corretta usando la distanza effettiva calcolata
        curvature = (2.0 * local_y) / (actual_L_d**2)

        v = self.target_v
        omega = np.clip(v * curvature, -self.max_omega, self.max_omega)

        return v, omega