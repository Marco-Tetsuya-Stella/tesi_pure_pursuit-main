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
    """Estrae i punti degli ostacoli per usarli come mappa globale (target) per l'ICP."""
    map_points = []
    for obstacle in env.obstacles:
        x_obs, y_obs = obstacle.exterior.xy
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
    Esegue la simulazione Pure Pursuit + ICP, con o senza loop closure.

    Args:
        path_name: nome del preset di traiettoria (vedi PrefabricatedPaths)
        use_loop_closure: se True, applica la logica di keyframe + loop closure;
            se False, la localizzazione si basa solo su odometria + ICP scan-to-map
        dt: passo temporale di integrazione
        total_steps: numero massimo di step di simulazione
        loop_cooldown: step minimi da attendere tra due loop closure accettati
        loop_min_separation: separazione temporale minima per considerare un keyframe candidato
        loop_search_radius: raggio massimo di ricerca del candidato
        lookahead_distance: distanza di lookahead L_d del controller Pure Pursuit (m)
        min_leave_start_dist: distanza minima dal punto di partenza che il robot deve
            percorrere prima che la condizione di "arrivo a destinazione" venga valutata.
            Necessario per i percorsi chiusi (es. 'eight', 'square', 'pista_f1'), dove
            l'ultimo punto del percorso coincide con il primo: senza questa soglia la
            simulazione terminerebbe immediatamente al primo step.
        min_icp_correspondences: numero minimo di corrispondenze punto-punto richieste
            perché la correzione ICP scan-to-map venga considerata affidabile e quindi
            applicata. Sotto questa soglia il fit è numericamente instabile (specialmente
            sulla rotazione) e viene scartato, mantenendo la sola predizione odometrica.
        max_icp_rmse: RMSE massimo accettabile perché la correzione ICP scan-to-map
            venga applicata. Sopra questa soglia il fit viene considerato inaffidabile
            e scartato.
        max_icp_angle_correction: correzione massima di orientamento (rad) che la
            correzione ICP può introdurre rispetto alla predizione odometrica. Dato
            che qui l'odometria è pressoché perfetta, un salto di orientamento più
            grande di questa soglia è quasi sempre un fit ICP numericamente instabile
            (es. per corrispondenze poco diversificate angolarmente, "aperture
            problem"), non una correzione legittima, e viene scartato.
        max_icp_pos_correction: correzione massima di posizione (m) analoga a
            max_icp_angle_correction, ma sulla componente (x, y).
        env_clearance: distanza minima (m) degli ostacoli dal percorso. Se None,
            usa il default specifico del preset (vedi PRESET_ENV_CONFIG in
            environment_presets_pure_pursuit.py).
        env_n_obstacles: numero di ostacoli nell'ambiente. Se None, usa il
            default specifico del preset.
        lidar_r_max: portata massima (m) del LiDAR. Se None, usa il default
            specifico del preset (vedi PRESET_ENV_CONFIG). Un r_max troppo
            grande aumenta il rischio di "perceptual aliasing" nella loop
            closure (il robot "vede" zone lontane simili ad altre già viste).
        max_loop_angle_correction: correzione massima di orientamento (rad)
            che una loop closure può introdurre rispetto alla stima corrente.
            RMSE e fitness misurano solo quanto bene i punti si allineano DATO
            un abbinamento, non che l'abbinamento sia con il posto giusto: con
            ostacoli di forma/dimensione simile in punti diversi dell'ambiente,
            un fit numericamente ottimo può comunque riferirsi al posto
            sbagliato (perceptual aliasing). Un salto enorme rispetto alla
            stima corrente (qui affidabile, essendo l'odometria quasi perfetta)
            è quasi sempre un falso positivo, anche con RMSE/fitness ottimi.
        max_loop_pos_correction: correzione massima di posizione (m), analoga
            a max_loop_angle_correction.
        min_loop_scan_points: numero minimo di punti nello scan corrente
            richiesti per tentare una loop closure. Con pochi punti, RMSE e
            fitness sono statisticamente inaffidabili anche quando appaiono
            ottimi (basta poca fortuna per un fit spurio).
        verbose: se True, stampa a schermo i messaggi di avanzamento/loop closure

    Returns:
        Dizionario con:
            "path": traiettoria di riferimento (Nx2 o Nx3)
            "robot_history": traiettoria reale del robot (Nx3)
            "estimated_history": traiettoria stimata (Nx2), da odometria+ICP(+loop closure)
            "env": ambiente usato
            "n_loops": numero di loop closure accettati (0 se use_loop_closure=False)
    """
    # 1. Inizializzazione Percorso
    path = PrefabricatedPaths.get_preset(path_name)

    # === AMBIENTE ===
    # Usa l'ambiente deterministico dedicato a questo preset (PRESET_ENV_CONFIG
    # in environment_presets_pure_pursuit.py), a meno che env_clearance/
    # env_n_obstacles non vengano specificati esplicitamente per sovrascriverlo.
    env = get_environment_for_preset(path_name, clearance=env_clearance, n_obstacles=env_n_obstacles)
    map_world = build_map_world(env)

    # Portata LiDAR: usa il default specifico del preset (proporzionato alla
    # sua estensione) a meno che non venga sovrascritto esplicitamente.
    if lidar_r_max is None:
        lidar_r_max = get_preset_env_defaults(path_name)["r_max"]

    # === RISOLUZIONE BUG INDEXERROR ===
    if path.shape[1] < 3:
        dx_init = path[1, 0] - path[0, 0]
        dy_init = path[1, 1] - path[0, 1]
        initial_theta = np.arctan2(dy_init, dx_init)
    else:
        initial_theta = path[0, 2]

    # 2. Inizializzazione Sensori, Controller e Robot
    robot = Robot(x=path[0, 0], y=path[0, 1], theta=initial_theta)
    controller = PurePursuitController( lookahead_distance=lookahead_distance, target_linear_velocity=0.4, stop_tolerance=0.1, max_index_gap=30)
    lidar = Lidar(n_rays=360, angle_span=2 * np.pi, r_max=lidar_r_max, add_noise=True)

    # 3. Inizializzazione del Simulatore
    sim = Simulator(robot=robot)
    sim.history = [robot.state().copy()]
    sim.commands = []
    sim.commands_applied = []

    # 4. Strutture dati per ICP e (opzionalmente) Loop Closure
    estimated_pose = robot.state().copy()
    previous_odom = robot.state().copy()
    keyframes = []
    last_loop_k = -10 ** 9
    n_loops = 0
    n_loop_rejected = 0
    n_icp_accepted = 0
    n_icp_rejected = 0

    # Flag per gestire correttamente i percorsi chiusi (dove path[-1] ≈ path[0]):
    # la condizione di arrivo viene valutata solo dopo che il robot si è
    # effettivamente allontanato dal punto di partenza.
    start_pos = path[0, :2].copy()
    left_start = False

    estimated_history = []

    for step in range(total_steps):
        current_odom = robot.state()
        estimated_history.append(estimated_pose[:2].copy())

        # B. Scansione LiDAR Reale nell'ambiente creato
        scan_local = lidar.scan_hits(current_odom, env, frame='local')

        # --- C. Localizzazione ---
        # 1. Calcolo del Delta Odometrico (Dead Reckoning)
        R_delta, t_delta = compute_relative_transform_from_odometry(previous_odom, current_odom)

        # 2. Applicazione del Delta alla Posa Stimata Attuale (Predizione)
        cos_e, sin_e = np.cos(estimated_pose[2]), np.sin(estimated_pose[2])
        R_est_current = np.array([[cos_e, -sin_e], [sin_e, cos_e]])
        t_est_current = estimated_pose[:2]

        init_R = R_est_current @ R_delta
        init_t = t_est_current + (R_est_current @ t_delta)

        # AGGIORNAMENTO DI BASE: Salvataggio della predizione odometrica
        estimated_pose[0] = init_t[0]
        estimated_pose[1] = init_t[1]
        estimated_pose[2] = np.arctan2(init_R[1, 0], init_R[0, 0])

        # Salva la predizione odometrica pura: verrà usata come riferimento per
        # giudicare se una correzione ICP successiva è fisicamente plausibile.
        predicted_pose = estimated_pose.copy()

        # 3. Correzione ICP solo se il LIDAR "vede" qualcosa
        if len(scan_local) > 3:
            icp_results = run_icp_scan_to_map_pair(
                map_world=map_world,
                curr_scan_local=scan_local,
                init_R=init_R,
                init_t=init_t,
                max_correspondence_distance=1.5
            )

            icp_init_res = icp_results['init']
            icp_t = icp_init_res['t']
            icp_theta = icp_init_res['alpha_rad']

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

            angle_correction = angle_diff(icp_theta, predicted_pose[2])
            pos_correction = float(np.linalg.norm(np.asarray(icp_t) - predicted_pose[:2]))
            plausibility_ok = (
                angle_correction <= max_icp_angle_correction
                and pos_correction <= max_icp_pos_correction
            )

            icp_reliable = fit_quality_ok and plausibility_ok

            if icp_reliable:
                # Sovrascrive la stima odometrica con i risultati ICP
                estimated_pose[0] = icp_t[0]
                estimated_pose[1] = icp_t[1]
                estimated_pose[2] = icp_theta
                n_icp_accepted += 1
            else:
                n_icp_rejected += 1

            # D. Logica di Loop Closure (solo se abilitata)
            if use_loop_closure:
                last_kf_pose = keyframes[-1].pose if keyframes else None
                if should_create_keyframe(estimated_pose, last_kf_pose):
                    kf = add_keyframe(keyframes, step, estimated_pose, scan_local)

                    # Prova la loop closure solo se è passato abbastanza tempo dall'ultimo loop accettato
                    # e se lo scan corrente ha abbastanza punti da rendere RMSE/fitness statisticamente
                    # affidabili (con pochi punti un fit spurio può comunque apparire ottimo).
                    if (step - last_loop_k) >= loop_cooldown and len(scan_local) >= min_loop_scan_points:
                        candidate = find_loop_candidate(
                            estimated_pose,
                            keyframes[:-1],  # esclude il keyframe appena aggiunto
                            step,
                            min_separation=loop_min_separation,
                            search_radius=loop_search_radius,
                        )

                        if candidate:
                            loop_res = try_loop_closure(
                                curr_scan_local=scan_local,
                                curr_pose_pred=estimated_pose,
                                candidate_kf=candidate,
                                max_corr_dist=0.3,
                                max_rmse=0.05,
                                min_fitness=0.7,
                            )

                            if loop_res:
                                # GATE DI PLAUSIBILITÀ FISICA: RMSE e fitness misurano solo
                                # quanto bene i punti si allineano dato un abbinamento, non che
                                # l'abbinamento sia con il posto giusto ("perceptual aliasing" —
                                # ostacoli simili in punti diversi dell'ambiente possono produrre
                                # un fit numericamente ottimo ma geometricamente sbagliato). Un
                                # salto enorme rispetto alla stima corrente (qui affidabile,
                                # essendo l'odometria quasi perfetta) è quasi sempre un falso
                                # positivo, anche con RMSE/fitness eccellenti.
                                loop_angle_correction = angle_diff(
                                    loop_res["pose_corrected"][2], estimated_pose[2]
                                )
                                loop_pos_correction = float(
                                    np.linalg.norm(loop_res["pose_corrected"][:2] - estimated_pose[:2])
                                )
                                loop_plausible = (
                                    loop_angle_correction <= max_loop_angle_correction
                                    and loop_pos_correction <= max_loop_pos_correction
                                )

                                if loop_plausible:
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

        # E. Inseguimento Traiettoria (Pure Pursuit) basato sulla Posa Stimata
        v, omega = controller.compute_commands(estimated_pose, path)

        # F. Aggiornamento fisico
        robot.set_command(v, omega)
        robot.step(dt)

        # Salvataggio manuale dei dati nel simulatore ad ogni step
        sim.history.append(robot.state().copy())
        sim.commands.append([v, omega])
        sim.commands_applied.append([v, omega])

        # Verifica della condizione di arrivo al traguardo.
        # Aggiorna il flag "left_start" appena il robot si allontana a sufficienza
        # dal punto di partenza (necessario per i percorsi chiusi, dove l'ultimo
        # punto del percorso coincide con il primo).
        if not left_start:
            if np.linalg.norm(current_odom[:2] - start_pos) > min_leave_start_dist:
                left_start = True

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
    """Disegna un confronto affiancato tra la simulazione CON e SENZA loop closure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    titles = [
        f"CON Loop Closure ({result_with_lc['n_loops']} accettati, {result_with_lc['n_loop_rejected']} scartati "
        f"per implausibilità, ICP scartati: {result_with_lc['n_icp_rejected']}/"
        f"{result_with_lc['n_icp_accepted'] + result_with_lc['n_icp_rejected']})",
        f"SENZA Loop Closure (ICP scartati: {result_without_lc['n_icp_rejected']}/"
        f"{result_without_lc['n_icp_accepted'] + result_without_lc['n_icp_rejected']})",
    ]
    results = [result_with_lc, result_without_lc]

    for ax, res, title in zip(axes, results, titles):
        env = res["env"]
        for obstacle in env.obstacles:
            x_obs, y_obs = obstacle.exterior.xy
            ax.fill(x_obs, y_obs, color='gray', alpha=0.5)

        path = res["path"]
        robot_history = res["robot_history"]
        estimated_history = res["estimated_history"]

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
    # Esegue il confronto CON/SENZA loop closure su tutti i preset di traiettoria disponibili.
    # Ogni preset produce la propria pagina/figura con i due pannelli affiancati.
    path_names = PrefabricatedPaths.list_presets()

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