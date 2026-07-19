import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, LineString
from typing import List, Optional
from shapely.errors import TopologicalError

# Importazioni dai moduli forniti
from robot import Robot
from lidar import Lidar
from environment import Environment
from prefabricated_paths import PrefabricatedPaths
from simulator import Simulator
from loop_closure import (
    should_create_keyframe,
    add_keyframe,
    find_loop_candidate,
    try_loop_closure
)

# Nuove importazioni da icp.py
from icp import (
    compute_relative_transform_from_odometry,
    run_icp_scan_to_map_pair
)

# Controller Pure Pursuit (ora in file dedicato)
from pure_pursuit import PurePursuitController


def build_environment() -> Environment:
    """Crea l'ambiente geometrico con ostacoli usato dalla simulazione."""
    env = Environment()
    env.set_bounds(-10.0, -10.0, 10.0, 10.0)

    # Aggiungiamo qualche ostacolo geometrico sparso per dare punti di riferimento al LIDAR
    env.add_circle(cx=1.5, cy=2.0, radius=0.4)
    env.add_circle(cx=-1.5, cy=4.0, radius=0.3)
    env.add_rectangle(xmin=2.0, ymin=4.0, xmax=3.0, ymax=5.0)
    env.add_wall(x0=-3.0, y0=1.0, x1=-2.0, y1=5.0, thickness=0.15)
    return env


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
        total_steps: int = 1500,
        loop_cooldown: int = 40,
        loop_min_separation: int = 40,
        loop_search_radius: float = 1.0,
        lookahead_distance: float = 0.3,
        min_leave_start_dist: float = 0.5,
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
    env = build_environment()
    map_world = build_map_world(env)

    # === RISOLUZIONE BUG INDEXERROR ===
    if path.shape[1] < 3:
        dx_init = path[1, 0] - path[0, 0]
        dy_init = path[1, 1] - path[0, 1]
        initial_theta = np.arctan2(dy_init, dx_init)
    else:
        initial_theta = path[0, 2]

    # 2. Inizializzazione Sensori, Controller e Robot
    robot = Robot(x=path[0, 0], y=path[0, 1], theta=initial_theta)
    controller = PurePursuitController(lookahead_distance=lookahead_distance, target_linear_velocity=0.5)
    lidar = Lidar(n_rays=360, angle_span=2 * np.pi, r_max=6.0, add_noise=True)

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

        # 3. Correzione ICP solo se il LIDAR "vede" qualcosa
        if len(scan_local) > 3:
            icp_results = run_icp_scan_to_map_pair(
                map_world=map_world,
                curr_scan_local=scan_local,
                init_R=init_R,
                init_t=init_t,
                max_correspondence_distance=1.5
            )

            # Sovrascrive la stima odometrica con i risultati ICP
            estimated_pose[0] = icp_results['init']['t'][0]
            estimated_pose[1] = icp_results['init']['t'][1]
            estimated_pose[2] = icp_results['init']['alpha_rad']

            # D. Logica di Loop Closure (solo se abilitata)
            if use_loop_closure:
                last_kf_pose = keyframes[-1].pose if keyframes else None
                if should_create_keyframe(estimated_pose, last_kf_pose):
                    kf = add_keyframe(keyframes, step, estimated_pose, scan_local)

                    # Prova la loop closure solo se è passato abbastanza tempo dall'ultimo loop accettato
                    if (step - last_loop_k) >= loop_cooldown:
                        candidate = find_loop_candidate(
                            estimated_pose,
                            keyframes[:-1],  # esclude il keyframe appena aggiunto
                            step,
                            min_separation=loop_min_separation,
                            search_radius=loop_search_radius,
                        )

                        if candidate:
                            loop_res = try_loop_closure(scan_local, estimated_pose, candidate)
                            if loop_res:
                                estimated_pose = loop_res["pose_corrected"]
                                last_loop_k = step
                                n_loops += 1

                                if verbose:
                                    print(
                                        f"[LOOP] step={step} | kf={candidate.k} | "
                                        f"rmse={loop_res['rmse']:.4f} | "
                                        f"fitness={loop_res['fitness']:.3f}"
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

        if left_start and np.linalg.norm(current_odom[:2] - path[-1, :2]) < 0.2:
            if verbose:
                label = "CON" if use_loop_closure else "SENZA"
                print(f"[{label} Loop Closure] Traguardo raggiunto al passo {step}!")
            break

    return {
        "path": path,
        "robot_history": np.array(sim.history),
        "estimated_history": np.array(estimated_history),
        "env": env,
        "n_loops": n_loops,
    }


def plot_comparison(result_with_lc: dict, result_without_lc: dict, path_name: str = "tight_slalom") -> None:
    """Disegna un confronto affiancato tra la simulazione CON e SENZA loop closure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    titles = [
        f"CON Loop Closure ({result_with_lc['n_loops']} loop accettati)",
        "SENZA Loop Closure",
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