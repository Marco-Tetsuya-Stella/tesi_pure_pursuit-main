"""

aggiunto per tesi per il pure pursuit
"""
import numpy as np


class NoisyOdometry:
    """
    Simula un'odometria con rumore realistico e intermittente:
    - Un rumore di fondo continuo molto lieve.
    - Eventi sporadici (es. slittamenti, perdita di aderenza) che introducono errori improvvisi.
    """

    def __init__(self, initial_state: np.ndarray,
                 base_noise_factor: float = 0.01,
                 slip_probability: float = 0.03,
                 slip_v_mag: float = 0.3,
                 slip_w_mag: float = 0.5):
        """
        Inizializza il simulatore di odometria intermittente.

        Args:
            initial_state: Array [x, y, theta] iniziale.
            base_noise_factor: Fattore di rumore di fondo proporzionale alla velocità (molto basso).
            slip_probability: Probabilità (0.0 a 1.0) che si verifichi un errore improvviso in un singolo step.
            slip_v_mag: Magnitudo massima dello slittamento sulla velocità lineare.
            slip_w_mag: Magnitudo massima dello slittamento sulla velocità angolare.
        """
        self.state = np.array(initial_state, dtype=float).copy()
        self.base_noise = base_noise_factor

        # Parametri per il rumore sporadico
        self.slip_prob = slip_probability
        self.slip_v_mag = slip_v_mag
        self.slip_w_mag = slip_w_mag

    def update(self, v: float, omega: float, dt: float) -> np.ndarray:
        """
        Applica il modello cinematico iniettando casualità sporadica.
        """
        # 1. Rumore di fondo (impercettibile, ma fisicamente realistico)
        std_v = self.base_noise * abs(v)
        std_w = self.base_noise * abs(omega)

        noisy_v = v + np.random.normal(0, std_v) if std_v > 0 else v
        noisy_omega = omega + np.random.normal(0, std_w) if std_w > 0 else omega

        # 2. Evento sporadico / Slittamento (Intermittente)
        # Avviene solo se il numero casuale supera la soglia di probabilità
        if np.random.random() < self.slip_prob:
            # Sceglie a caso il tipo di disturbo: lineare, angolare o entrambi
            disturbo = np.random.choice(['lineare', 'angolare', 'entrambi'])

            if disturbo in ['lineare', 'entrambi']:
                # Aggiunge un picco casuale uniforme (positivo o negativo) alla velocità
                noisy_v += np.random.uniform(-self.slip_v_mag, self.slip_v_mag)

            if disturbo in ['angolare', 'entrambi']:
                # Aggiunge uno strattone improvviso alla rotazione
                noisy_omega += np.random.uniform(-self.slip_w_mag, self.slip_w_mag)

        # 3. Integrazione di Eulero con i comandi (eventualmente alterati)
        self.state[0] += noisy_v * np.cos(self.state[2]) * dt
        self.state[1] += noisy_v * np.sin(self.state[2]) * dt
        self.state[2] += noisy_omega * dt

        # 4. Normalizzazione angolo [-pi, pi]
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi

        return self.state.copy()