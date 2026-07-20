"""

Aggiunto per la tesi riguradante il pure pursuit

"""
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
        use_loop_closure: bool = True,
        dt: float = 0.05,
        total_steps: int = 5000,
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
        verbose: bool = True,
):
    """
    Esegue la simulazione Pure Pursuit + ICP, permettendo di testare il sistema
    sia con che senza l'attivazione della logica di loop closure.

    Args:
        path_name: Nome della traiettoria prefabbricata da seguire.
        use_loop_closure: Booleano per attivare o disattivare il sistema di keyframe e loop closure[cite: 14].
        dt: Passo temporale di integrazione della simulazione in secondi[cite: 14].
        total_steps: Limite massimo di iterazioni per evitare loop infiniti[cite: 14].
        loop_cooldown: Iterazioni minime di pausa tra un loop closure accettato e il successivo[cite: 14].
        loop_min_separation: Distanza temporale minima (in step) tra la posa attuale e un keyframe candidato[cite: 14].
        loop_search_radius: Raggio metrico per la ricerca di keyframe candidati storici[cite: 14].
        lookahead_distance: Distanza di mira per il controllore Pure Pursuit[cite: 14].
        min_leave_start_dist: Distanza che il robot deve coprire prima che il simulatore possa valutare l'arrivo al traguardo (utile per percorsi chiusi)[cite: 14].
        min_icp_correspondences: Soglia minima di punti corrispondenti affinché l'ICP sia ritenuto valido[cite: 14].
        max_icp_rmse: Errore quadratico medio massimo tollerato per accettare una correzione ICP[cite: 14].
        max_icp_angle_correction: Massima rotazione correttiva permessa all'ICP per evitare "salti" causati da instabilità numerica[cite: 14].
        max_icp_pos_correction: Massima traslazione correttiva permessa all'ICP[cite: 14].
        env_clearance: Distanza di sicurezza degli ostacoli (sovrascrive il default se specificato)[cite: 14].
        env_n_obstacles: Numero di ostacoli desiderati (sovrascrive il default)[cite: 14].
        lidar_r_max: Portata massima del sensore LiDAR in metri[cite: 14].
        max_loop_angle_correction: Massima correzione angolare permessa al loop closure per scartare i falsi positivi da perceptual aliasing[cite: 14].
        max_loop_pos_correction: Massima correzione spaziale permessa al loop closure[cite: 14].
        min_loop_scan_points: Punti minimi necessari in una scansione per tentare la loop closure[cite: 14].
        verbose: Se True, stampa messaggi informativi in console durante l'esecuzione[cite: 14].

    Returns:
        Un dizionario contenente la cronologia della traiettoria di riferimento,
        quella reale percorsa, quella stimata, e varie metriche di successo/fallimento dell'ICP e del LC[cite: 14].
        Dizionario con:
            "path": traiettoria di riferimento (Nx2 o Nx3)
            "robot_history": traiettoria reale del robot (Nx3)
            "estimated_history": traiettoria stimata (Nx2), da odometria+ICP(+loop closure)
            "env": ambiente usato
            "n_loops": numero di loop closure accettati (0 se use_loop_closure=False)
    """

    # --- 1. Inizializzazione Percorso e Ambiente ---
    path = PrefabricatedPaths.get_preset(path_name)

    # === AMBIENTE ===
    # Usa l'ambiente deterministico dedicato a questo preset (PRESET_ENV_CONFIG
    # in environment_presets_pure_pursuit.py), a meno che env_clearance/
    # env_n_obstacles non vengano specificati esplicitamente per sovrascriverlo.
    env = get_environment_for_preset(path_name, clearance=env_clearance, n_obstacles=env_n_obstacles)

    # Genera la "mappa del mondo" statica usata per allineare le scansioni del LiDAR[cite: 14]
    map_world = build_map_world(env)

    # Configura il raggio del LiDAR in base all'ambiente se non specificato[cite: 14]
    if lidar_r_max is None:
        lidar_r_max = get_preset_env_defaults(path_name)["r_max"]

    # Inizializza l'orientamento di partenza evitando errori se l'array del percorso ha solo due colonne (X, Y)[cite: 14]
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

    # Prepara le strutture del simulatore per memorizzare i dati[cite: 14]
    sim = Simulator(robot=robot)
    sim.history = [robot.state().copy()]
    sim.commands = []
    sim.commands_applied = []

    # --- 3. Strutture dati per Localizzazione (ICP e (opzionalmente) Loop Closure) ---
    estimated_pose = robot.state().copy()
    previous_odom = robot.state().copy()
    keyframes = []
    last_loop_k = -10 ** 9
    n_loops = n_loop_rejected = n_icp_accepted = n_icp_rejected = 0

    # Variabili per gestire la logica di fine percorso nei tracciati chiusi ad anello[cite: 14]
    # Flag per gestire correttamente i percorsi chiusi (dove path[-1] ≈ path[0]):
    # la condizione di arrivo viene valutata solo dopo che il robot si è
    # effettivamente allontanato dal punto di partenza.
    start_pos = path[0, :2].copy()
    left_start = False
    estimated_history = []

    # --- 4. Ciclo di Simulazione Principale ---
    for step in range(total_steps):
        current_odom = robot.state()
        estimated_history.append(estimated_pose[:2].copy())

        # Effettua la lettura del LiDAR simulato basandosi sulla posizione reale del robot[cite: 14]
        scan_local = lidar.scan_hits(current_odom, env, frame='local')

        # === Fase di Localizzazione ===

        # A. Calcola lo spostamento del robot rispetto allo step precedente (Dead Reckoning)[cite: 14]
        R_delta, t_delta = compute_relative_transform_from_odometry(previous_odom, current_odom)

        # B. Applica lo spostamento misurato alla posizione stimata (Predizione)[cite: 14]
        cos_e, sin_e = np.cos(estimated_pose[2]), np.sin(estimated_pose[2])
        R_est_current = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        t_est_current = estimated_pose[:2]

        init_R = R_est_current @ R_delta
        init_t = t_est_current + (R_est_current @ t_delta)

        # Salva temporaneamente la nuova posa basata solo sull'odometria[cite: 14]
        estimated_pose[0] = init_t[0]
        estimated_pose[1] = init_t[1]
        estimated_pose[2] = np.arctan2(init_R[1, 0], init_R[0, 0])

        # Salva la predizione odometrica pura: verrà usata come riferimento per
        # giudicare se una correzione ICP successiva è fisicamente plausibile.
        predicted_pose = estimated_pose.copy()

        # C. Correzione ICP Scan-to-Map (solo se il LiDAR rileva abbastanza punti)[cite: 14]
        if len(scan_local) > 3:
            # Tenta di far collimare la scansione attuale con la mappa globale del mondo[cite: 14]
            icp_results = run_icp_scan_to_map_pair(
                map_world=map_world, curr_scan_local=scan_local,
                init_R=init_R, init_t=init_t, max_correspondence_distance=1.5
            )

            icp_init_res = icp_results['init']
            icp_t = icp_init_res['t']
            icp_theta = icp_init_res['alpha_rad']

            # Verifica la validità matematica dell'algoritmo ICP [cite: 14]
            # GATING DI QUALITÀ, in due parti:
            # (a) qualità intrinseca del fit ICP (corrispondenze, RMSE, convergenza)
            # (b) plausibilità fisica della correzione rispetto alla predizione
            #     odometrica: dato che qui l'odometria è pressoché perfetta, un
            #     salto enorme rispetto ad essa è quasi sempre un fit corrotto
            #     (es. "aperture problem" su corrispondenze poco diversificate
            #     angolarmente), non una correzione legittima.
            fit_quality_ok = (
                    icp_init_res['converged']
                    and icp_init_res['n_corr_last'] >= min_icp_correspondences
                    and icp_init_res['rmse'] <= max_icp_rmse
            )

            # Verifica la plausibilità fisica: scarta le correzioni che spostano il robot in modo irrealistico rispetto alla stima odometrica[cite: 14]
            angle_correction = angle_diff(icp_theta, predicted_pose[2])
            pos_correction = float(np.linalg.norm(np.asarray(icp_t) - predicted_pose[:2]))
            plausibility_ok = (
                        angle_correction <= max_icp_angle_correction and pos_correction <= max_icp_pos_correction)

            # Se i controlli sono superati, la posa viene corretta definitivamente[cite: 14]
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
                # Se il robot si è mosso abbastanza, salva lo stato attuale come "fotografia" (Keyframe)[cite: 14]
                if should_create_keyframe(estimated_pose, last_kf_pose):
                    kf = add_keyframe(keyframes, step, estimated_pose, scan_local)

                    # Verifica se è trascorso il periodo di cooldown e se ci sono sufficienti punti nella scansione
                    # scan corrente ha abbastanza punti da rendere RMSE/fitness statisticamente affidabili
                    if (step - last_loop_k) >= loop_cooldown and len(scan_local) >= min_loop_scan_points:
                        # Cerca nella cronologia un keyframe fisicamente vicino alla posa attuale[cite: 14]
                        candidate = find_loop_candidate(
                            estimated_pose,
                            keyframes[:-1], # esclude il keyframe appena aggiunto
                            step,
                            min_separation=loop_min_separation, search_radius=loop_search_radius,
                        )

                        if candidate:
                            # Tenta di sovrapporre la scansione attuale con quella del keyframe storico trovato[cite: 14]
                            loop_res = try_loop_closure(
                                curr_scan_local=scan_local, curr_pose_pred=estimated_pose,
                                candidate_kf=candidate, max_corr_dist=0.3, max_rmse=0.05, min_fitness=0.7,
                            )

                            if loop_res:
                                # Controlla che il loop closure non proponga correzioni assurde (perceptual aliasing)[cite: 14]
                                # GATE DI PLAUSIBILITÀ FISICA: RMSE e fitness misurano solo
                                # quanto bene i punti si allineano dato un abbinamento, non che
                                # l'abbinamento sia con il posto giusto ("perceptual aliasing" —
                                # ostacoli simili in punti diversi dell'ambiente possono produrre
                                # un fit numericamente ottimo ma geometricamente sbagliato). Un
                                # salto enorme rispetto alla stima corrente (qui affidabile,
                                # essendo l'odometria quasi perfetta) è quasi sempre un falso
                                # positivo, anche con RMSE/fitness eccellenti.
                                loop_angle_correction = angle_diff(loop_res["pose_corrected"][2], estimated_pose[2])
                                loop_pos_correction = float(
                                    np.linalg.norm(loop_res["pose_corrected"][:2] - estimated_pose[:2]))

                                loop_plausible = (
                                            loop_angle_correction <= max_loop_angle_correction and
                                            loop_pos_correction <= max_loop_pos_correction)

                                if loop_plausible:
                                    # Se tutto è coerente, applica la macro-correzione del loop closure[cite: 14]
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

        # Calcola la velocità lineare (v) e angolare (omega) affinché il robot segua il percorso[cite: 14]
        v, omega = controller.compute_commands(estimated_pose, path)

        # Invia i comandi ai motori e avanza la fisica di uno step di tempo (dt)[cite: 14]
        robot.set_command(v, omega)
        robot.step(dt)

        # Salva la cronologia reale per i grafici finali[cite: 14]
        # Salvataggio manuale dei dati nel simulatore ad ogni step
        sim.history.append(robot.state().copy())
        sim.commands.append([v, omega])
        sim.commands_applied.append([v, omega])

        # === Condizioni di Fine Simulazione ===

        # Verifica della condizione di arrivo al traguardo.
        # Aggiorna il flag "left_start" appena il robot si allontana a sufficienza
        # dal punto di partenza (necessario per i percorsi chiusi, dove l'ultimo
        # punto del percorso coincide con il primo).
        # Controlla se il robot si è staccato dal punto di partenza[cite: 14]
        if not left_start:
            if np.linalg.norm(current_odom[:2] - start_pos) > min_leave_start_dist:
                left_start = True

        # Interrompe la simulazione se il traguardo è stato raggiunto (motori fermi)[cite: 14]
        if left_start and (v == 0.0 and omega == 0.0):
            if verbose:
                label = "CON" if use_loop_closure else "SENZA"
                print(f"[{label} Loop Closure] Traguardo raggiunto al passo {step}!")
                print(f"  ICP scan-to-map: {n_icp_accepted} accettati, {n_icp_rejected} scartati per bassa qualità")
            break

    if verbose and (n_icp_accepted + n_icp_rejected) > 0:
        print(
            f"[Riepilogo ICP] {path_name} ({'CON' if use_loop_closure else 'SENZA'} LC): "
            f"{n_icp_accepted} accettati, {n_icp_rejected} scartati "
            f"({100 * n_icp_rejected / (n_icp_accepted + n_icp_rejected):.1f}% scartati)"
        )


    # Restituisce un riepilogo statistico della singola corsa[cite: 14]
    return {
        "path": path,
        "robot_history": np.array(sim.history),
        "estimated_history": np.array(estimated_history),
        "env": env,
        "n_loops": n_loops,
        "n_loop_rejected": n_loop_rejected,
        "n_icp_accepted": n_icp_accepted,
        "n_icp_rejected": n_icp_rejected,
    }


def plot_comparison(result_with_lc: dict, result_without_lc: dict, path_name: str = "tight_slalom") -> None:
    """
    Crea un grafico a due pannelli per confrontare visivamente i risultati
    della navigazione con e senza il sistema di loop closure[cite: 14].

    Args:
        result_with_lc: Dizionario dei risultati della simulazione con loop closure attivo[cite: 14].
        result_without_lc: Dizionario dei risultati della simulazione senza loop closure[cite: 14].
        path_name: Nome del preset usato, per scopi di titolazione[cite: 14].
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    # Prepara i titoli estraendo le metriche calcolate nelle simulazioni[cite: 14]
    titles = [
        f"CON Loop Closure ({result_with_lc['n_loops']} accettati, {result_with_lc['n_loop_rejected']} scartati "
        f"per implausibilità, ICP scartati: {result_with_lc['n_icp_rejected']}/"
        f"{result_with_lc['n_icp_accepted'] + result_with_lc['n_icp_rejected']})",
        f"SENZA Loop Closure (ICP scartati: {result_without_lc['n_icp_rejected']}/"
        f"{result_without_lc['n_icp_accepted'] + result_without_lc['n_icp_rejected']})",
    ]
    results = [result_with_lc, result_without_lc]

    # Itera sui due scenari per disegnare i rispettivi grafici[cite: 14]
    for ax, res, title in zip(axes, results, titles):
        # 1. Disegna gli ostacoli in grigio[cite: 14]
        env = res["env"]
        for obstacle in env.obstacles:
            x_obs, y_obs = obstacle.exterior.xy
            ax.fill(x_obs, y_obs, color='gray', alpha=0.5)

        path = res["path"]
        robot_history = res["robot_history"]
        estimated_history = res["estimated_history"]

        # 2. Disegna i tre strati informativi (percorso teorico, posizione reale, posizione stimata dal robot)[cite: 14]
        ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
        ax.plot(robot_history[:, 0], robot_history[:, 1], 'b-', label='Robot Traiettoria Reale')
        ax.plot(estimated_history[:, 0], estimated_history[:, 1], 'r.', markersize=3, label='Stima ICP + Odom')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(title)

    fig.suptitle(f"Pure Pursuit con ICP — Confronto su preset '{path_name}'")
    plt.tight_layout()
    plt.show()


def main():
    """
    Funzione d'ingresso principale (entry point) dello script.
    Avvia una suite di test comparativi su tutte le piste disponibili[cite: 14].
    Esegue il confronto CON/SENZA loop closure su tutti i preset di traiettoria disponibili.
    Ogni preset produce la propria pagina/figura con i due pannelli affiancati.
    """
    # Recupera tutti i nomi dei percorsi programmati (es. circolare, quadrato, slalom)[cite: 14]
    path_names = PrefabricatedPaths.list_presets()

    # Itera su ciascun percorso ed esegue il doppio test[cite: 14]
    for path_name in path_names:
        print(f"\n{'=' * 60}")
        print(f"PRESET: {path_name}")
        print(f"{'=' * 60}")

        # Esegue la simulazione CON loop closure
        print(f"=== Esecuzione CON Loop Closure ({path_name}) ===")
        result_with_lc = run_simulation(path_name=path_name, use_loop_closure=True)

        # Esegue la simulazione SENZA loop closure (solo odometria + ICP scan-to-map)
        print(f"\n=== Esecuzione SENZA Loop Closure ({path_name}) ===")
        result_without_lc = run_simulation(path_name=path_name, use_loop_closure=False)

        # Confronto grafico affiancato per questo preset
        plot_comparison(result_with_lc, result_without_lc, path_name=path_name)


if __name__ == "__main__":
    main()