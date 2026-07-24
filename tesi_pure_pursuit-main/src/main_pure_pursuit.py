from prefabricated_paths import PrefabricatedPaths
from pure_pursuit_simulation import run_simulation
# Importata la funzione per il salvataggio automatico del singolo gruppo
from visualizer_pure_pursuit import InteractiveVisualizer, export_group_plots


def main():
    """
    Funzione d'ingresso principale dello script.
    Avvia la suite di test, salva i grafici su disco ad ogni iterazione (Opzione B)
    e raggruppa i risultati per poi aprirli nel visualizzatore interattivo.
    """
    path_names = PrefabricatedPaths.list_presets()
    variants = ["type1"]
    lookahead_distances = [0.2, 0.4, 0.6]  # array di valori di lookahead

    # Lista che conterrà tutti i setup. Ogni setup raggruppa le 4 categorie.
    all_experiment_groups = []

    for path_name in path_names:
        for variant in variants:
            for ld in lookahead_distances:
                print(f"\n{'=' * 75}")
                print(f"PRESET: {path_name} | VARIANTE: {variant.upper()} | LOOK AHEAD: {ld}m")
                print(f"{'=' * 75}")

                print(f"\n[1/4] Esecuzione ODOMETRIA IDEALE (SENZA Loop Closure)...")
                res_ideal_no_lc = run_simulation(
                    path_name=path_name, variant=variant, use_loop_closure=False,
                    add_odom_noise=False, lookahead_distance=ld, verbose=False
                )

                print(f"[2/4] Esecuzione ODOMETRIA IDEALE (CON Loop Closure)...")
                res_ideal_lc = run_simulation(
                    path_name=path_name, variant=variant, use_loop_closure=True,
                    add_odom_noise=False, lookahead_distance=ld, verbose=False
                )

                print(f"[3/4] Esecuzione ODOMETRIA RUMOROSA (SENZA Loop Closure)...")
                res_noisy_no_lc = run_simulation(
                    path_name=path_name, variant=variant, use_loop_closure=False,
                    add_odom_noise=True, lookahead_distance=ld, verbose=False
                )

                print(f"[4/4] Esecuzione ODOMETRIA RUMOROSA (CON Loop Closure)...")
                res_noisy_lc = run_simulation(
                    path_name=path_name, variant=variant, use_loop_closure=True,
                    add_odom_noise=True, lookahead_distance=ld, verbose=False
                )

                # Raggruppa l'esperimento corrente
                group = {
                    "path_name": path_name,
                    "variant": variant,
                    "ld": ld,
                    "ideal_no_lc": res_ideal_no_lc,
                    "ideal_lc": res_ideal_lc,
                    "noisy_no_lc": res_noisy_no_lc,
                    "noisy_lc": res_noisy_lc
                }

                # ------------------------------------------------------------------
                # OPZIONE B: Salvataggio automatico ed immediato del gruppo su disco
                # ------------------------------------------------------------------
                print(f"Salvataggio automatico grafici per {path_name}_{variant}_ld{ld}...")
                export_group_plots(group)
                # ------------------------------------------------------------------

                all_experiment_groups.append(group)

    print("\n>>> Tutte le simulazioni e i salvataggi su disco sono stati completati.")
    print(">>> Avvio della visualizzazione interattiva...")

    # Avvia la classe visualizer e mostra la schermata
    visualizer = InteractiveVisualizer(all_experiment_groups)
    visualizer.show()


if __name__ == "__main__":
    main()