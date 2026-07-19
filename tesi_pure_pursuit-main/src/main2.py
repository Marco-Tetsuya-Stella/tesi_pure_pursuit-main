import visualizer
from prefabricated_paths import PrefabricatedPaths, convert_2d_path_to_3d_states


def main():
    print("=== Visualizzazione Percorsi Geometrici Singoli ===")

    nomi_preset = [
        "circle_medium",
        "tight_slalom",
        "eight",
        "pista_f1"
    ]

    # 1. Recupero dei punti 2D e conversione in pose 3D [x, y, theta]
    path_list_2d = [PrefabricatedPaths.get_preset(nome) for nome in nomi_preset]
    histories = [convert_2d_path_to_3d_states(p) for p in path_list_2d]

    print("[Viewer] Lancio del visualizzatore a pannello singolo...")
    # Usiamo la nuova funzione creata appositamente
    visualizer.show_path_only_carousel(
        histories=histories,
        titles=nomi_preset,
        show_orient_every=10,
        _dts=0.05,
        environment=None,  # Passando None, NON compariranno ostacoli o confini
        _fit_to='trajectory'
    )


if __name__ == "__main__":
    main()