"""

Esegue la simulazione Pure Pursuit
su un tracciato selezionato, mostrando l'animazione a schermo e salvando
il video MP4 nella cartella 'video_pure_pursuit' allo stesso livello di 'src'.
Evita la sovrascrittura generando percorsi univoci (es. file_1.mp4, file_2.mp4).
Aggiunto per tesi riguardante il Pure Pursuit


IMPORTANTE IMPORTANTE IMPORTANTE IMPORTANTE

per far funzionare FFMpegWriter scaricare FFmpeg, che è un programma/eseguibile di sistema
(installato tramite winget), non un pacchetto Python pip.

"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Importazioni dei moduli del progetto
from pure_pursuit_simulation import run_simulation
from visualizer import draw_robot
from visualizer_pure_pursuit import plot_environment, adjust_axis_limits, get_unique_filepath


def main():
    # ==========================================
    # 1. PARAMETRI DI CONFIGURAZIONE
    # ==========================================
    nome_tracciato = "pista_1"
    variante = "type2"
    lookahead = 0.4

    # Flag di controllo
    usa_odometria_rumorosa = True  # Modifica qui: True per rumore, False per ideale
    usa_loop_closure = False  # Modifica qui: True/False

    # ==========================================
    # 2. GENERAZIONE DINAMICA DEI NOMI E TITOLI
    # ==========================================
    str_odom = "noisy" if usa_odometria_rumorosa else "ideal"
    str_lc = "withlc" if usa_loop_closure else "nolc"

    titolo_odom = "Odometria Rumorosa" if usa_odometria_rumorosa else "Odometria Ideale"
    titolo_lc = "Con Loop Closure" if usa_loop_closure else "Senza Loop Closure"

    print(f"{'=' * 75}")
    print(f"AVVIO SIMULAZIONE SU: '{nome_tracciato}' (Look-Ahead: {lookahead}m)")
    print(f"Modalità: {titolo_odom} | {titolo_lc}")
    print(f"{'=' * 75}\n")

    # ==========================================
    # 3. ESECUZIONE SIMULAZIONE
    # ==========================================
    risultato = run_simulation(
        path_name=nome_tracciato,
        variant=variante,
        use_loop_closure=usa_loop_closure,
        add_odom_noise=usa_odometria_rumorosa,
        lookahead_distance=lookahead,
        verbose=True
    )

    env = risultato['env']
    path = risultato['path']
    real_history = risultato['robot_history']

    if len(real_history) == 0:
        print("[ERRORE] La traiettoria generata è vuota. Impossibile continuare.")
        return

    # ==========================================
    # 4. CARTELLA E NOME FILE VIDEO AUTOMATICO
    # ==========================================
    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parent / "video_pure_pursuit"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Il nome file ora include dinamicamente 'noisy'/'ideal' e 'withlc'/'nolc'
    nome_base_video = f"{nome_tracciato}_{variante}_ld{lookahead}_{str_odom}_{str_lc}.mp4"
    percorso_video_originale = base_dir / nome_base_video
    percorso_video = get_unique_filepath(percorso_video_originale)

    # ==========================================
    # 5. IMPOSTAZIONE GRAFICA E ANIMAZIONE
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f'Pure Pursuit - {nome_tracciato}')

    plot_environment(ax, env)
    ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
    trail_line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Traiettoria Reale')

    # Titolo del grafico dinamico
    ax.set_title(f"Simulazione Pure Pursuit: '{nome_tracciato}'\n{titolo_odom} | {titolo_lc}", fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True)
    adjust_axis_limits(ax, path, env=env, real=real_history, min_range=5.0)

    robot_artists = []

    def init():
        trail_line.set_data([], [])
        return trail_line,

    def update(frame):
        nonlocal robot_artists

        for artist in robot_artists:
            try:
                artist.remove()
            except Exception:
                pass
        robot_artists.clear()

        trail_line.set_data(real_history[:frame + 1, 0], real_history[:frame + 1, 1])

        xlim = ax.get_xlim()
        scale = max(0.08, (xlim[1] - xlim[0]) * 0.02)

        state = real_history[frame]
        robot_artists = draw_robot(ax, state, robot_radius=scale, color='tab:blue')

        return [trail_line] + robot_artists

    num_frames = len(real_history)
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40, repeat=False)

    # ==========================================
    # 6. SALVATAGGIO VIDEO ED ESECUZIONE
    # ==========================================
    print(f"Generazione e salvataggio del video MP4 in corso...")
    print(f"Destinazione: {percorso_video.name}")
    try:
        writer = FFMpegWriter(fps=25, metadata=dict(artist='Pure Pursuit Simulation'), bitrate=1800)
        ani.save(str(percorso_video), writer=writer)
        print(f"[SUCCESS] Video salvato con successo in:\n -> {percorso_video}\n")
    except Exception as e:
        print(f"[ERRORE] Salvataggio MP4 fallito: {e}")

    print("Apertura finestra grafica a schermo...")
    plt.show()


if __name__ == "__main__":
    main()