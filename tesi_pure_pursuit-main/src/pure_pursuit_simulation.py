import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from lidar import Lidar
from environment import Environment
from prefabricated_paths import PrefabricatedPaths
from simulator import Simulator
from environment_presets_pure_pursuit import get_environment_for_preset, get_preset_env_defaults
from pure_pursuit import PurePursuitController
from loop_closure import (
    should_create_keyframe,
    add_keyframe,
    find_loop_candidate,
    try_loop_closure,
    angle_diff,
)
from icp import (
    compute_relative_transform_from_odometry,
    run_icp_scan_to_map_pair
)
from noisy_odometry import NoisyOdometry


def build_map_world(env: Environment) -> np.ndarray:
    """
        Estrae i punti degli ostacoli per usarli come mappa globale (target) per l'ICP.

        Args:
            env: L'oggetto Environment contenente gli ostacoli poligonali.

        Returns:
            Un array numpy bidimensionale contenente le coordinate (x, y) di tutti i punti
            che compongono i confini degli ostacoli nell'ambiente.
        """
    map_points = []
    # Scorre tutti gli ostacoli presenti nell'ambiente
    for obstacle in env.obstacles:
        # Estrae le coordinate x e y del perimetro esterno dell'ostacolo
        x_obs, y_obs = obstacle.exterior.xy
        # Accoppia le coordinate e le aggiunge alla lista dei punti della mappa
        for x, y in zip(x_obs, y_obs):
            map_points.append([x, y])
    return np.array(map_points)


def run_simulation(
        path_name: str = "tight_slalom",
        variant: str = "type2",
        use_loop_closure: bool = True,
        add_odom_noise: bool = False,
        dt: float = 0.05,
        total_steps: int = 10000,
        loop_cooldown: int = 40,
        loop_min_separation: int = 150,
        loop_search_radius: float = 0.8,
        lookahead_distance: float = 0.2,
        min_leave_start_dist: float = 0.5,
        min_icp_correspondences: int = 8,
        max_icp_rmse: float = 0.3,
        max_icp_angle_correction: float = np.deg2rad(20.0),
        max_icp_pos_correction: float = 0.6,
        env_clearance: float = None,
        env_n_obstacles: int = None,
        lidar_r_max: float = None,
        max_loop_angle_correction: float = np.deg2rad(25.0),
        max_loop_pos_correction: float = 0.8,
        min_loop_scan_points: int = 15,
        min_scan_points_for_icp: int = 12,
        verbose: bool = True,
):
    """
    Esegue la simulazione Pure Pursuit + ICP, permettendo di testare il sistema
    sia con che senza l'attivazione della logica di loop closure, e valutando
    l'effetto dell'odometria ideale o rumorosa.

    Args:
        path_name: Nome della traiettoria prefabbricata da seguire.
        variant: Variante della traiettoria o dell'ambiente da utilizzare.
        use_loop_closure: Booleano per attivare o disattivare il sistema di keyframe e loop closure.
        add_odom_noise: Booleano per attivare o disattivare il rumore artificiale sull'odometria.
        dt: Passo temporale di integrazione della simulazione in secondi.
        total_steps: Limite massimo di iterazioni per evitare loop infiniti.
        loop_cooldown: Iterazioni minime di pausa tra un loop closure accettato e il successivo.
        loop_min_separation: Distanza temporale minima (in step) tra la posa attuale e un keyframe candidato.
        loop_search_radius: Raggio metrico per la ricerca di keyframe candidati storici.
        lookahead_distance: Distanza di mira per il controllore Pure Pursuit.
        min_leave_start_dist: Distanza che il robot deve coprire prima che il simulatore possa valutare l'arrivo al traguardo (utile per percorsi chiusi).
        min_icp_correspondences: Soglia minima di punti corrispondenti affinché l'ICP sia ritenuto valido.
        max_icp_rmse: Errore quadratico medio massimo tollerato per accettare una correzione ICP.
        max_icp_angle_correction: Massima rotazione correttiva permessa all'ICP per evitare "salti" causati da instabilità numerica.
        max_icp_pos_correction: Massima traslazione correttiva permessa all'ICP.
        env_clearance: Distanza di sicurezza degli ostacoli (sovrascrive il default se specificato).
        env_n_obstacles: Numero di ostacoli desiderati (sovrascrive il default).
        lidar_r_max: Portata massima del sensore LiDAR in metri.
        max_loop_angle_correction: Massima correzione angolare permessa al loop closure per scartare i falsi positivi da perceptual aliasing.
        max_loop_pos_correction: Massima correzione spaziale permessa al loop closure.
        min_loop_scan_points: Punti minimi necessari in una scansione per tentare la loop closure.
        min_scan_points_for_icp: Punti minimi necessari in una scansione per l'allineamento ICP.
        verbose: Se True, stampa messaggi informativi in console durante l'esecuzione.

    Returns:
        Un dizionario contenente la cronologia della traiettoria di riferimento,
        quella reale percorsa, quella stimata, e varie metriche di successo/fallimento dell'ICP e del LC.
        Dizionario con:
            "path": traiettoria di riferimento (Nx2 o Nx3)
            "robot_history": traiettoria reale del robot (Nx3)
            "estimated_history": traiettoria stimata (Nx2), da odometria+ICP(+loop closure)
            "env": ambiente usato
            "n_loops": numero di loop closure accettati (0 se use_loop_closure=False)
            "n_loop_rejected": numero di loop closure rifiutati per plausibilità
            "n_icp_accepted": numero di correzioni ICP accettate
            "n_icp_rejected": numero di correzioni ICP scartate
            "variant": variante dell'ambiente/percorso utilizzata
            "noise_enabled": indica se il rumore sull'odometria è stato applicato
            "loop_closure_enabled": indica se la logica del loop closure è stata utilizzata
    """

    # --- 1. Inizializzazione Percorso e Ambiente ---
    path = PrefabricatedPaths.get_preset(path_name)

    # === AMBIENTE ===
    # Usa l'ambiente deterministico dedicato a questo preset, considerando anche
    # le variazioni e la possibilità di sovrascrivere i default.
    env = get_environment_for_preset(path_name, variant=variant, clearance=env_clearance, n_obstacles=env_n_obstacles)

    # Genera la "mappa del mondo" statica usata per allineare le scansioni del LiDAR
    map_world = build_map_world(env)

    # Configura il raggio del LiDAR in base all'ambiente se non specificato
    if lidar_r_max is None:
        lidar_r_max = get_preset_env_defaults(path_name, variant=variant)["r_max"]

    # Inizializza l'orientamento di partenza evitando errori se l'array del percorso ha solo due colonne (X, Y)
    if path.shape[1] < 3:
        dx_init = path[1, 0] - path[0, 0]
        dy_init = path[1, 1] - path[0, 1]
        initial_theta = np.arctan2(dy_init, dx_init)
    else:
        initial_theta = path[0, 2]

    # --- 2. Inizializzazione Sensori, Controller e Robot ---
    robot = Robot(x=path[0, 0], y=path[0, 1], theta=initial_theta)
    controller = PurePursuitController(lookahead_distance=lookahead_distance, target_linear_velocity=0.4,
                                       stop_tolerance=0.1, max_index_gap=30)
    lidar = Lidar(n_rays=360, angle_span=2 * np.pi, r_max=lidar_r_max, add_noise=True)

    # Prepara le strutture del simulatore per memorizzare i dati
    sim = Simulator(robot=robot)
    sim.history = [robot.state().copy()]
    sim.commands = []
    sim.commands_applied = []

    # --- 3. Strutture dati per Localizzazione (Odometria, ICP e Loop Closure) ---
    # Setup odometria
    estimated_pose = robot.state().copy()
    if add_odom_noise:
        noisy_odom = NoisyOdometry(robot.state())
        current_odom_state = noisy_odom.state.copy()
        previous_odom = current_odom_state.copy()
    else:
        current_odom_state = robot.state().copy()
        previous_odom = current_odom_state.copy()

    keyframes = []
    last_loop_k = -10 ** 9
    n_loops = n_loop_rejected = n_icp_accepted = n_icp_rejected = 0

    # Variabili per gestire la logica di fine percorso nei tracciati chiusi ad anello
    # Flag per gestire correttamente i percorsi chiusi (dove path[-1] ≈ path[0]):
    # la condizione di arrivo viene valutata solo dopo che il robot si è
    # effettivamente allontanato dal punto di partenza.
    start_pos = path[0, :2].copy()
    left_start = False
    estimated_history = []

    # --- 4. Ciclo di Simulazione Principale ---
    for step in range(total_steps):
        current_true_pose = robot.state()
        estimated_history.append(estimated_pose[:2].copy())

        # Decide quale odometria usare per questo step
        if add_odom_noise:
            current_odom = current_odom_state.copy()
        else:
            current_odom = current_true_pose.copy()

        # Effettua la lettura del LiDAR simulato basandosi sulla posizione reale del robot
        # Lo scanner LIDAR usa la posa reale
        scan_local = lidar.scan_hits(current_true_pose, env, frame='local')

        # === Fase di Localizzazione ===

        # A. Calcola lo spostamento del robot rispetto allo step precedente (Dead Reckoning)
        # Il calcolo del delta odometrico usa la posa stimata/rumorosa
        R_delta, t_delta = compute_relative_transform_from_odometry(previous_odom, current_odom)

        # B. Applica lo spostamento misurato alla posizione stimata (Predizione)
        cos_e, sin_e = np.cos(estimated_pose[2]), np.sin(estimated_pose[2])
        R_est_current = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        t_est_current = estimated_pose[:2]

        init_R = R_est_current @ R_delta
        init_t = t_est_current + (R_est_current @ t_delta)

        # Salva temporaneamente la nuova posa basata solo sull'odometria
        estimated_pose[0] = init_t[0]
        estimated_pose[1] = init_t[1]
        estimated_pose[2] = np.arctan2(init_R[1, 0], init_R[0, 0])

        # Salva la predizione odometrica pura: verrà usata come riferimento per
        # giudicare se una correzione ICP successiva è fisicamente plausibile.
        predicted_pose = estimated_pose.copy()

        # C. Correzione ICP Scan-to-Map (solo se il LiDAR rileva abbastanza punti)
        if len(scan_local) > min_scan_points_for_icp:
            # Tenta di far collimare la scansione attuale con la mappa globale del mondo
            icp_results = run_icp_scan_to_map_pair(
                map_world=map_world, curr_scan_local=scan_local,
                init_R=init_R, init_t=init_t, max_correspondence_distance=1.5
            )

            icp_init_res = icp_results['init']
            icp_t = icp_init_res['t']
            icp_theta = icp_init_res['alpha_rad']

            # Verifica la validità matematica dell'algoritmo ICP
            # GATING DI QUALITÀ, in due parti:
            # (a) qualità intrinseca del fit ICP (corrispondenze, RMSE, convergenza)
            # (b) plausibilità fisica della correzione rispetto alla predizione
            #     odometrica: un salto enorme rispetto ad essa è quasi sempre un fit
            #     corrotto (es. "aperture problem" su corrispondenze poco diversificate
            #     angolarmente), non una correzione legittima.
            fit_quality_ok = (
                    icp_init_res['converged']
                    and icp_init_res['n_corr_last'] >= min_icp_correspondences
                    and icp_init_res['rmse'] <= max_icp_rmse
            )

            # Verifica la plausibilità fisica: scarta le correzioni che spostano il robot
            # in modo irrealistico rispetto alla stima odometrica
            angle_correction = angle_diff(icp_theta, predicted_pose[2])
            pos_correction = float(np.linalg.norm(np.asarray(icp_t) - predicted_pose[:2]))

            confidence = min(1.0, icp_init_res['n_corr_last'] / (min_icp_correspondences * 2.5))
            trust_factor = 0.3 + 0.7 * confidence
            effective_max_angle = max_icp_angle_correction * trust_factor
            effective_max_pos = max_icp_pos_correction * trust_factor

            plausibility_ok = (angle_correction <= effective_max_angle and pos_correction <= effective_max_pos)

            # Se i controlli sono superati, la posa viene corretta definitivamente
            if fit_quality_ok and plausibility_ok:
                estimated_pose[0] = icp_t[0]
                estimated_pose[1] = icp_t[1]
                estimated_pose[2] = icp_theta
                n_icp_accepted += 1
            else:
                n_icp_rejected += 1

            # D. Logica di Loop Closure (solo se abilitata)
            if use_loop_closure:
                last_kf_pose = keyframes[-1].pose if keyframes else None
                # Se il robot si è mosso abbastanza, salva lo stato attuale come "fotografia" (Keyframe)
                if should_create_keyframe(estimated_pose, last_kf_pose):
                    kf = add_keyframe(keyframes, step, estimated_pose, scan_local)

                    # Verifica se è trascorso il periodo di cooldown e se ci sono sufficienti punti nella scansione
                    # scan corrente ha abbastanza punti da rendere RMSE/fitness statisticamente affidabili
                    if (step - last_loop_k) >= loop_cooldown and len(scan_local) >= min_loop_scan_points:
                        # Cerca nella cronologia un keyframe fisicamente vicino alla posa attuale
                        candidate = find_loop_candidate(
                            estimated_pose,
                            keyframes[:-1],  # esclude il keyframe appena aggiunto
                            step,
                            min_separation=loop_min_separation, search_radius=loop_search_radius,
                        )

                        if candidate:
                            # Tenta di sovrapporre la scansione attuale con quella del keyframe storico trovato
                            loop_res = try_loop_closure(
                                curr_scan_local=scan_local, curr_pose_pred=estimated_pose,
                                candidate_kf=candidate, max_corr_dist=0.3, max_rmse=0.05, min_fitness=0.7,
                            )

                            if loop_res:
                                # Controlla che il loop closure non proponga correzioni assurde (perceptual aliasing)
                                # GATE DI PLAUSIBILITÀ FISICA: RMSE e fitness misurano solo
                                # quanto bene i punti si allineano dato un abbinamento, non che
                                # l'abbinamento sia con il posto giusto ("perceptual aliasing" —
                                # ostacoli simili in punti diversi dell'ambiente possono produrre
                                # un fit numericamente ottimo ma geometricamente sbagliato). Un
                                # salto enorme rispetto alla stima corrente è quasi sempre un falso
                                # positivo, anche con RMSE/fitness eccellenti.
                                loop_angle_correction = angle_diff(loop_res["pose_corrected"][2], estimated_pose[2])
                                loop_pos_correction = float(
                                    np.linalg.norm(loop_res["pose_corrected"][:2] - estimated_pose[:2]))

                                loop_plausible = (
                                            loop_angle_correction <= max_loop_angle_correction and loop_pos_correction <= max_loop_pos_correction)

                                if loop_plausible:
                                    # Se tutto è coerente, applica la macro-correzione del loop closure
                                    estimated_pose = loop_res["pose_corrected"]
                                    last_loop_k = step
                                    n_loops += 1

                                    if verbose:
                                        print(
                                            f"[LOOP] step={step} | kf={candidate.k} | "
                                            f"rmse={loop_res['rmse']:.4f} | "
                                            f"fitness={loop_res['fitness']:.3f}"
                                        )
                                else:
                                    n_loop_rejected += 1
                                    if verbose:
                                        print(
                                            f"[LOOP SCARTATO] step={step} | kf={candidate.k} | "
                                            f"rmse={loop_res['rmse']:.4f} | fitness={loop_res['fitness']:.3f} | "
                                            f"salto_angolo={np.degrees(loop_angle_correction):.1f}° | "
                                            f"salto_pos={loop_pos_correction:.3f}m -> fit ottimo ma implausibile "
                                            f"(possibile perceptual aliasing)"
                                        )

        previous_odom = current_odom.copy()

        # === Fase di Controllo (Movimento) ===

        # Calcola la velocità lineare (v) e angolare (omega) affinché il robot segua il percorso
        v, omega = controller.compute_commands(estimated_pose, path)

        # Invia i comandi ai motori e avanza la fisica di uno step di tempo (dt)
        robot.set_command(v, omega)
        robot.step(dt)

        if add_odom_noise:
            current_odom_state = noisy_odom.update(v, omega, dt)

        # Salva la cronologia reale per i grafici finali
        # Salvataggio manuale dei dati nel simulatore ad ogni step
        sim.history.append(robot.state().copy())
        sim.commands.append([v, omega])
        sim.commands_applied.append([v, omega])

        # === Condizioni di Fine Simulazione ===

        # Verifica della condizione di arrivo al traguardo.
        # Aggiorna il flag "left_start" appena il robot si allontana a sufficienza
        # dal punto di partenza (necessario per i percorsi chiusi, dove l'ultimo
        # punto del percorso coincide con il primo).
        # Controlla se il robot si è staccato dal punto di partenza
        if not left_start:
            if np.linalg.norm(current_true_pose[:2] - start_pos) > min_leave_start_dist:
                left_start = True

        # Interrompe la simulazione se il traguardo è stato raggiunto (motori fermi)
        if left_start and (v == 0.0 and omega == 0.0):
            if verbose:
                label_lc = "CON" if use_loop_closure else "SENZA"
                label_noise = "RUMOROSA" if add_odom_noise else "IDEALE"
                print(
                    f"[{label_lc} Loop Closure | Odometria {label_noise}] Traguardo raggiunto al passo {step} (Variante: {variant.upper()})!")
                print(f"  ICP scan-to-map: {n_icp_accepted} accettati, {n_icp_rejected} scartati per bassa qualità")
            break

    # Riepilogo finale stampato a console
    if verbose and (n_icp_accepted + n_icp_rejected) > 0:
        print(
            f"[Riepilogo ICP] {path_name} ({'CON' if use_loop_closure else 'SENZA'} LC | Odometria {'RUMOROSA' if add_odom_noise else 'IDEALE'}): "
            f"{n_icp_accepted} accettati, {n_icp_rejected} scartati "
            f"({100 * n_icp_rejected / (n_icp_accepted + n_icp_rejected):.1f}% scartati)"
        )

    # Restituisce un riepilogo statistico della singola corsa
    return {
        "path": path,
        "robot_history": np.array(sim.history),
        "estimated_history": np.array(estimated_history),
        "env": env,
        "n_loops": n_loops,
        "n_loop_rejected": n_loop_rejected,
        "n_icp_accepted": n_icp_accepted,
        "n_icp_rejected": n_icp_rejected,
        "variant": variant,
        "noise_enabled": add_odom_noise,
        "loop_closure_enabled": use_loop_closure
    }


# def plot_comparison(res_ideal_no_lc: dict, res_ideal_lc: dict,
#                     res_noisy_no_lc: dict, res_noisy_lc: dict,
#                     path_name: str = "tight_slalom") -> None:
#     """
#     Crea un grafico a griglia 2x2 per confrontare visivamente i risultati
#     della navigazione testando tutte e quattro le combinazioni possibili:
#     odometria ideale vs rumorosa, combinate con e senza il sistema di loop closure.
#
#     Args:
#         res_ideal_no_lc: Dizionario dei risultati con odometria ideale e senza loop closure.
#         res_ideal_lc: Dizionario dei risultati con odometria ideale e con loop closure attivo.
#         res_noisy_no_lc: Dizionario dei risultati con odometria rumorosa e senza loop closure.
#         res_noisy_lc: Dizionario dei risultati con odometria rumorosa e con loop closure attivo.
#         path_name: Nome del preset usato, per scopi di titolazione.
#     """
#     # Creiamo una griglia 2x2 per mostrare le 4 combinazioni
#     fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=True)
#
#     # Appiattiamo l'array degli assi (da 2x2 a 1D) per iterare più facilmente
#     axes = axes.flatten()
#
#     results = [res_ideal_no_lc, res_ideal_lc, res_noisy_no_lc, res_noisy_lc]
#
#     # Prepara i titoli estraendo le metriche calcolate nelle varie simulazioni
#     titles = [
#         f"IDEALE | SENZA Loop Closure\n(ICP scartati: {res_ideal_no_lc['n_icp_rejected']}/{res_ideal_no_lc['n_icp_accepted'] + res_ideal_no_lc['n_icp_rejected']})",
#         f"IDEALE | CON Loop Closure\n(Loop: {res_ideal_lc['n_loops']} | ICP scart: {res_ideal_lc['n_icp_rejected']}/{res_ideal_lc['n_icp_accepted'] + res_ideal_lc['n_icp_rejected']})",
#         f"RUMOROSA | SENZA Loop Closure\n(ICP scartati: {res_noisy_no_lc['n_icp_rejected']}/{res_noisy_no_lc['n_icp_accepted'] + res_noisy_no_lc['n_icp_rejected']})",
#         f"RUMOROSA | CON Loop Closure\n(Loop: {res_noisy_lc['n_loops']} | ICP scart: {res_noisy_lc['n_icp_rejected']}/{res_noisy_lc['n_icp_accepted'] + res_noisy_lc['n_icp_rejected']})"
#     ]
#
#     # Itera sui quattro scenari per disegnare i rispettivi grafici
#     for ax, res, title in zip(axes, results, titles):
#         # 1. Disegna gli ostacoli in grigio per fornire contesto spaziale
#         env = res["env"]
#         for obstacle in env.obstacles:
#             x_obs, y_obs = obstacle.exterior.xy
#             ax.fill(x_obs, y_obs, color='gray', alpha=0.5)
#
#         # Estrai i dati di navigazione dal dizionario
#         path = res["path"]
#         robot_history = res["robot_history"]
#         estimated_history = res["estimated_history"]
#
#         # 2. Disegna i tre strati informativi (percorso teorico, posizione reale, posizione stimata dal robot)
#         ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
#         ax.plot(robot_history[:, 0], robot_history[:, 1], 'b-', label='Robot Traiettoria Reale')
#         ax.plot(estimated_history[:, 0], estimated_history[:, 1], 'r.', markersize=3, label='Stima ICP + Odom')
#
#         # 3. Formattazione del singolo grafico (legenda, griglia, proporzioni)
#         ax.legend(loc='lower right', fontsize='small')
#         ax.grid(True)
#         ax.set_aspect('equal')
#         ax.set_title(title, fontweight="bold")
#
#     # Recupera il nome della variante ambientale (default "Sconosciuta" se mancante)
#     variant_name = res_ideal_lc.get("variant", "Sconosciuta").upper()
#
#     # Aggiunge il titolo globale all'intera figura
#     fig.suptitle(f"Pure Pursuit ICP — Odometria IDEALE vs RUMOROSA su '{path_name}' | Variante: {variant_name}",
#                  fontsize=16, y=0.95)
#
#     # Aggiusta il layout per evitare sovrapposizioni tra i grafici e i titoli
#     plt.tight_layout(rect=[0, 0, 1, 0.93])
#     plt.show()


