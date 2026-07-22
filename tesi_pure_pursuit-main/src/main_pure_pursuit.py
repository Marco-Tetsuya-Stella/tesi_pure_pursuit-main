from prefabricated_paths import PrefabricatedPaths

# IMPORTANTE: Assicurati di importare run_simulation dal file corretto dove è definita
from pure_pursuit_simulation import run_simulation
from visualizer_pure_pursuit import plot_comparison


def main():
    """
    Funzione d'ingresso principale (entry point) dello script.
    Avvia una suite di test comparativi completi su tutte le piste e varianti disponibili,
    testando 3 diversi valori di look ahead distance.
    """
    path_names = PrefabricatedPaths.list_presets()
    variants = ["type1", "type2", "type3"]

    # Definiamo i 3 valori differenti per la Look Ahead Distance
    lookahead_distances = [0.2, 0.4, 0.6]

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

                print(
                    f"\n>>> Simulazioni completate per {path_name} - {variant} (Look Ahead: {ld}m). Generazione grafico...")

                # Chiamata al nuovo modulo di visualizzazione, passando il valore ld
                plot_comparison(
                    res_ideal_no_lc, res_ideal_lc, res_noisy_no_lc, res_noisy_lc,
                    path_name=path_name, lookahead_distance=ld
                )


if __name__ == "__main__":
    main()