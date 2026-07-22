import matplotlib.pyplot as plt


def plot_comparison(res_ideal_no_lc: dict, res_ideal_lc: dict,
                    res_noisy_no_lc: dict, res_noisy_lc: dict,
                    path_name: str = "tight_slalom",
                    lookahead_distance: float = 0.2) -> None:
    """
    Crea un grafico a griglia 2x2 per confrontare visivamente i risultati
    della navigazione testando tutte e quattro le combinazioni possibili.
    Mostra anche il Look Ahead Distance utilizzato per l'esperimento.
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    results = [res_ideal_no_lc, res_ideal_lc, res_noisy_no_lc, res_noisy_lc]

    titles = [
        f"IDEALE | SENZA Loop Closure\n(ICP scartati: {res_ideal_no_lc['n_icp_rejected']}/{res_ideal_no_lc['n_icp_accepted'] + res_ideal_no_lc['n_icp_rejected']})",
        f"IDEALE | CON Loop Closure\n(Loop: {res_ideal_lc['n_loops']} | ICP scart: {res_ideal_lc['n_icp_rejected']}/{res_ideal_lc['n_icp_accepted'] + res_ideal_lc['n_icp_rejected']})",
        f"RUMOROSA | SENZA Loop Closure\n(ICP scartati: {res_noisy_no_lc['n_icp_rejected']}/{res_noisy_no_lc['n_icp_accepted'] + res_noisy_no_lc['n_icp_rejected']})",
        f"RUMOROSA | CON Loop Closure\n(Loop: {res_noisy_lc['n_loops']} | ICP scart: {res_noisy_lc['n_icp_rejected']}/{res_noisy_lc['n_icp_accepted'] + res_noisy_lc['n_icp_rejected']})"
    ]

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

        ax.legend(loc='lower right', fontsize='small')
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight="bold")

    variant_name = res_ideal_lc.get("variant", "Sconosciuta").upper()

    # Aggiunto il valore della lookahead_distance al titolo del grafico
    fig.suptitle(
        f"Pure Pursuit ICP — Odometria IDEALE vs RUMOROSA su '{path_name}'\nVariante: {variant_name} | Look Ahead Distance: {lookahead_distance} m",
        fontsize=16, y=0.96)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()