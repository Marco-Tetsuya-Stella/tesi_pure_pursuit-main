"""
Ambienti deterministici per ciascun preset di PrefabricatedPaths.

Ogni preset ha un ambiente FISSO e RIPRODUCIBILE:
stesso preset in ingresso => stessi ostacoli,
nelle stesse posizioni, ad ogni esecuzione. Non c'è alcun elemento casuale.

Gli ostacoli sono distribuiti su tutta l'area dei bounds usando una sequenza
di Halton (quasi-random ma completamente deterministica): questo evita che
gli ostacoli si dispongano tutti "a corridoio" lungo i due lati del percorso
(pattern troppo prevedibile e comodo per l'ICP), dando invece una copertura
naturale e ben distribuita di tutta l'area, con forme e dimensioni variate
(cerchi, rettangoli, poligoni).

Ogni candidato viene comunque verificato prima di essere aggiunto (distanza
minima dal percorso, nessuna sovrapposizione con altri ostacoli, dentro i
bounds).

Ogni ambiente viene costruito una sola volta e poi cache-ato: chiamate
successive con lo stesso preset (e stessi parametri) ritornano sempre lo
stesso oggetto, senza rigenerarlo.



Aggiunto per la tesi riguradante il pure pursuit
"""

import numpy as np
from shapely.geometry import LineString, Point, box, Polygon as ShapelyPolygon

from environment import Environment
from prefabricated_paths import PrefabricatedPaths


# --- Configurazione per-preset di clearance e numero di ostacoli e varianti di difficoltà ---
#
# Ogni percorso ha estensione e complessità diverse , quindi invece di
# un unico valore fisso per tutti, ogni preset ha i propri default sensati.
# Questi valori sono comunque sempre sovrascrivibili passando esplicitamente
# 'clearance' e/o 'n_obstacles' alle funzioni sottostanti.


# Ogni percorso ha diverse varianti (es. "type1", "type2", "type3") con
# configurazioni specifiche. Questo permette di testare lo stesso tracciato
# con densità di ostacoli e spazi di manovra differenti.

PRESET_ENV_CONFIG: dict = {
    "straight_short": {
        "type1":   {"clearance": 0.8, "n_obstacles": 0,  "r_max": 4.0},
        "type2":   {"clearance": 0.7, "n_obstacles": 2,  "r_max": 8.0},
        "type3":   {"clearance": 0.6, "n_obstacles": 5,  "r_max": 4.0},
    },
    "straight_long": {
        "type1":   {"clearance": 0.9, "n_obstacles": 0,  "r_max": 6.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 3, "r_max": 12.0},
        "type3":   {"clearance": 0.7, "n_obstacles": 10, "r_max": 6.0},
    },
    "circle_medium": {
        "type1":   {"clearance": 0.9, "n_obstacles": 0,  "r_max": 5.0},
        "type2":   {"clearance": 0.8, "n_obstacles": 2, "r_max": 10.0},
        "type3":   {"clearance": 0.7, "n_obstacles": 8, "r_max": 5.0},
    },
    "circle_large": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 7.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 3, "r_max": 14.0},
        "type3":   {"clearance": 0.8, "n_obstacles": 10, "r_max": 7.0},
    },
    "tight_slalom": {
        "type1":   {"clearance": 0.7, "n_obstacles": 0,  "r_max": 4.5},
        "type2":   {"clearance": 0.7, "n_obstacles": 3, "r_max": 5.5},
        "type3":   {"clearance": 0.5, "n_obstacles": 8, "r_max": 4.5},
    },
    "wide_slalom": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 6.5},
        "type2":   {"clearance": 0.9, "n_obstacles": 3, "r_max": 10},
        "type3":   {"clearance": 0.8, "n_obstacles": 10, "r_max": 6.5},
    },
    "eight": {
        "type1":   {"clearance": 0.9, "n_obstacles": 0,  "r_max": 6.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 3, "r_max": 8.0},
        "type3":   {"clearance": 0.7, "n_obstacles": 9, "r_max": 6.0},
    },
    "square": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 6.5},
        "type2":   {"clearance": 0.9, "n_obstacles": 4, "r_max": 12.5},
        "type3":   {"clearance": 0.8, "n_obstacles": 10, "r_max": 6.5},
    },
    "pista_f1": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 7.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 4, "r_max": 14.0},
        "type3":   {"clearance": 0.8, "n_obstacles": 10, "r_max": 7.0},
    },
    "pista_2": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 6.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 5,  "r_max": 10.0},
        "type3":   {"clearance": 0.8, "n_obstacles": 12, "r_max": 8.0},
    },
    "pista_3_(2_giri)": {
        "type1":   {"clearance": 1.0, "n_obstacles": 0,  "r_max": 6.0},
        "type2":   {"clearance": 0.9, "n_obstacles": 3,  "r_max": 9.0},
        "type3":   {"clearance": 0.8, "n_obstacles": 8, "r_max": 6.0},
    },
}


# Default di fallback per preset non presenti nella tabella sopra (es. nuovi
# preset aggiunti in futuro a PrefabricatedPaths senza una voce dedicata qui)
_DEFAULT_CLEARANCE = 0.8
_DEFAULT_N_OBSTACLES = 15
_DEFAULT_R_MAX = 6.0


def get_preset_env_defaults(name: str, variant: str = "type2") -> dict:
    """
    Ritorna {'clearance': ..., 'n_obstacles': ..., 'r_max': ...} per il preset
    e la variante richiesta ("type1", "type2", "type3").
    Usa default generici se il preset o la variante non esistono.
    """
    key = name.lower().strip()
    variant_key = variant.lower().strip()

    # 1. Cerca il tracciato. Se non esiste, ritorna un dizionario vuoto
    track_variants = PRESET_ENV_CONFIG.get(key, {})

    # 2. Cerca la variante dentro il tracciato. Se non esiste (o se il tracciato
    # era vuoto), ritorna i valori di fallback universali.
    return track_variants.get(
        variant_key,
        {
            "clearance": _DEFAULT_CLEARANCE,
            "n_obstacles": _DEFAULT_N_OBSTACLES,
            "r_max": _DEFAULT_R_MAX
        }
    )


def _halton(index: int, base: int) -> float:
    """
    Genera un numero quasi-casuale nell'intervallo [0, 1) usando la sequenza di Halton.

    La sequenza di Halton è una sequenza a bassa discrepanza usata per distribuire
    punti nello spazio in modo uniforme e naturale, evitando "grumi" (clustering).
    A differenza di random.uniform(), è al 100% deterministica: a pari 'index' e 'base'
    restituirà sempre identico valore.

    Args:
        index: L'indice del punto nella sequenza (un intero positivo, es. 1, 2, 3...).
        base: Un numero primo usato come base di conversione (es. 2, 3, 5, 7).
            Basi diverse generano dimensioni spaziali tra loro indipendenti.

    Returns:
        float: Un valore compreso tra 0.0 (incluso) e 1.0 (escluso).
    """
    # Variabile accumulatore per il risultato finale (inizialmente 0.0)
    result = 0.0

    # Peso della cifra corrente: parte come (1 / base) e viene diviso ad ogni ciclo
    # (es. con base 2: 0.5, poi 0.25, poi 0.125, ...)
    f = 1.0 / base

    # Copia locale dell'indice da scomporre
    i = index

    # Ciclo che "riflette" le cifre dell'indice convertito nella base scelta
    while i > 0:
        # 1. (i % base) estrae l'ultima cifra di 'i' nella base 'base'
        # 2. Moltiplica la cifra per il peso corrente 'f' e la somma al risultato
        result += f * (i % base)

        # Riduce 'i' eliminando l'ultima cifra appena elaborata (divisione intera)
        i //= base

        # Riduce il peso per la cifra successiva
        f /= base

    return result


def _is_safe(env: Environment, geom, path_line: LineString, clearance: float) -> bool:
    """
    Verifica se un ostacolo candidato rispetta tutti i vincoli di sicurezza dell'ambiente.

    Controlla tre condizioni fondamentali:
    1. Distanza di sicurezza (clearance) minima dal percorso del robot.
    2. Assenza di sovrapposizioni (collisioni) con ostacoli già esistenti.
    3. Completo contenimento all'interno dei confini (bounds) dell'ambiente.

    Args:
        env: L'oggetto Environment corrente contenente gli ostacoli già piazzati e i confini.
        geom: La geometria Shapely dell'ostacolo candidato (es. Polygon, Point.buffer, box).
        path_line: La traiettoria di riferimento rappresentata come LineString.
        clearance: Distanza minima (m) che l'ostacolo deve mantenere dalla traiettoria.

    Returns:
        bool: True se l'ostacolo è valido e sicuro da posizionare, False altrimenti.
    """
    # 1. CONTROLLO DISTANZA DAL PERCORSO
    # Calcola la distanza minima tra la forma dell'ostacolo (geom) e la linea del percorso (path_line).
    # Se è inferiore alla soglia di sicurezza 'clearance', il candidato viene scartato per evitare ostruzioni.
    if geom.distance(path_line) < clearance:
        return False

    # 2. CONTROLLO SOVRAPPOSIZIONE CON ALTRI OSTACOLI
    # Scorre tutti gli ostacoli già confermati e presenti nell'ambiente.
    for ob in env.obstacles:
        # Se la nuova geometria si interseca o sovrappone a un ostacolo esistente, la scarta.
        if geom.intersects(ob):
            return False

    # 3. CONTROLLO CONFINI DELL'AMBIENTE (BOUNDS)
    # Se sono definiti dei limiti geografici per la mappa...
    if env.bounds is not None:
        try:
            # Verifica che l'ostacolo sia INTERAMENTE contenuto all'interno del rettangolo dei bounds.
            if not env.bounds.contains(geom):
                return False
        # Gestisce eventuali incompatibilità o errori nei tipi geometrici di Shapely
        except (ValueError, AttributeError, TypeError):
            return False

    # Se supera tutti e tre i controlli, l'ostacolo è ritenuto sicuro e valido
    return True


def _make_polygon_geom(cx: float, cy: float, r: float, n_sides: int, rotation: float):
    """
    Costruisce la geometria di un poligono regolare a due dimensioni (es. triangolo, pentagono, esagono).

    Calcola la posizione trigonometrica di ciascun vertice distribuendolo
    equidistante lungo una circonferenza di raggio 'r' centrata in '(cx, cy)'.

    Args:
        cx: Coordinata X del centro del poligono.
        cy: Coordinata Y del centro del poligono.
        r: Raggio del cerchio circoscritto al poligono (distanza centro-vertice).
        n_sides: Numero di lati (e di vertici) del poligono regolare.
        rotation: Angolo di rotazione iniziale applicato alla figura, espresso in radianti.

    Returns:
        tuple: Una tupla contenente due elementi:
            - verts (list): Lista di tuple con le coordinate float [(x1, y1), (x2, y2), ...] dei vertici.
            - ShapelyPolygon: Oggetto geometrico 2D utilizzabile per i calcoli spaziali di Shapely.
    """
    # 1. CALCOLO DEGLI ANGOLI
    # Genera 'n_sides' angoli distribuiti in modo uniforme nell'intervallo [0, 2π) (360 gradi).
    # 'endpoint=False' evita di raddoppiare l'angolo finale 2π (che coincide con 0).
    # Aggiunge 'rotation' per ruotare l'intera figura.
    angles = rotation + np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)

    # 2. CALCOLO DELLE COORDINATE CARTESIANE DEI VERTICI
    # Per ogni angolo 'a', converte la posizione polare (r, a) in coordinate cartesiane (x, y)
    # usando la trigonometria di base (coseno per X, seno per Y) e trasla il tutto rispetto al centro (cx, cy).
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

    # 3. CREAZIONE E RITORNO DELLA GEOMETRIA
    # Restituisce sia la lista grezza delle coordinate vertici, sia l'oggetto Polygon della libreria Shapely.
    return verts, ShapelyPolygon(verts)


def _build_environment_for_path(
        path: np.ndarray,
        clearance: float = 0.8,
        bounds_pad: float = 2.5,
        n_obstacles: int = 15,
        min_size: float = 0.25,
        max_size: float = 0.55,
        max_attempts_factor: int = 25,
) -> Environment:
    """
    Costruisce deterministicamente un Environment per un dato percorso.

    Args:
        path: array Nx2 o Nx3 con la traiettoria di riferimento.
        clearance: distanza minima (m) che ogni ostacolo deve mantenere dal percorso.
        bounds_pad: margine (m) oltre l'estensione del percorso per i bounds dell'ambiente.
        n_obstacles: numero di ostacoli da tentare di piazzare nell'ambiente.
        min_size, max_size: intervallo di raggio/mezza-dimensione degli ostacoli.
        max_attempts_factor: numero massimo di candidati testati, come multiplo
            di n_obstacles (per garantire terminazione anche se molti candidati
            vengono scartati).

    Returns:
        Environment con bounds e ostacoli fissati deterministicamente.
    """
    env = Environment()
    path_xy = np.asarray(path, dtype=float)[:, :2]

    x_min, y_min = float(np.min(path_xy[:, 0])), float(np.min(path_xy[:, 1]))
    x_max, y_max = float(np.max(path_xy[:, 0])), float(np.max(path_xy[:, 1]))
    b_left, b_bottom = x_min - bounds_pad, y_min - bounds_pad
    b_right, b_top = x_max + bounds_pad, y_max + bounds_pad
    env.set_bounds(b_left, b_bottom, b_right, b_top)

    path_line = LineString(path_xy.tolist())
    if path_line.length < 1e-6:
        return env

    span_x = b_right - b_left
    span_y = b_top - b_bottom

    placed = 0
    attempt = 0
    max_attempts = max(50, n_obstacles * max_attempts_factor)

    # Offset dell'indice di Halton per ciascuna dimensione: evita che le prime
    # iterazioni (a bassa discrepanza ma vicine all'origine della sequenza)
    # producano cluster nello stesso angolo per ogni preset.
    halton_offset = 17

    while placed < n_obstacles and attempt < max_attempts:
        idx = halton_offset + attempt
        attempt += 1

        # Posizione candidata distribuita su TUTTA l'area dei bounds (non
        # agganciata al percorso): la sequenza di Halton (basi 2 e 3) copre
        # l'area in modo uniforme e ben distribuito, senza clustering né
        # pattern ripetitivi, restando comunque deterministica.
        fx = _halton(idx, 2)
        fy = _halton(idx, 3)
        cx = b_left + fx * span_x
        cy = b_bottom + fy * span_y

        # Dimensione e forma variate deterministicamente da altre due
        # dimensioni della sequenza di Halton (basi 5 e 7), indipendenti
        # dalla posizione, per avere varietà di feature per l'ICP.
        size_frac = _halton(idx, 5)
        r = min_size + size_frac * (max_size - min_size)

        shape_frac = _halton(idx, 7)
        if shape_frac < 0.34:
            shape_kind = "circle"
        elif shape_frac < 0.67:
            shape_kind = "rectangle"
        else:
            shape_kind = "polygon"

        if shape_kind == "circle":
            candidate = Point(cx, cy).buffer(r, resolution=24)
            if _is_safe(env, candidate, path_line, clearance):
                env.add_circle(cx, cy, r)
                placed += 1

        elif shape_kind == "rectangle":
            # Rettangolo non quadrato (rapporto d'aspetto variato) per
            # aggiungere ulteriore diversità geometrica
            aspect = 0.6 + 0.8 * _halton(idx, 11)  # ~0.6 - 1.4
            w, h = r * 2.0, r * 2.0 * aspect
            xmin_r, ymin_r = cx - w / 2, cy - h / 2
            xmax_r, ymax_r = cx + w / 2, cy + h / 2
            candidate = box(xmin_r, ymin_r, xmax_r, ymax_r)
            if _is_safe(env, candidate, path_line, clearance):
                env.add_rectangle(xmin_r, ymin_r, xmax_r, ymax_r)
                placed += 1

        else:  # poligono regolare (triangolo, pentagono o esagono)
            n_sides = 3 + (int(idx) % 3) * 2  # alterna 3, 5, 7 lati
            rotation = 2 * np.pi * _halton(idx, 13)
            verts, candidate = _make_polygon_geom(cx, cy, r, n_sides, rotation)
            if _is_safe(env, candidate, path_line, clearance):
                env.add_polygon(verts)
                placed += 1

    return env


# --- Registro degli ambienti: costruiti una sola volta, poi cache-ati ---

_ENVIRONMENTS_CACHE: dict = {}


def get_environment_for_preset(
    name: str,
    variant: str = "type2",
    clearance: float = None,
    n_obstacles: int = None,
) -> Environment:
    """
    Ritorna l'Environment deterministico associato al preset e alla variante richiesta.

    Se 'clearance' e/o 'n_obstacles' non vengono specificati, si usano i
    default definiti in PRESET_ENV_CONFIG per quel preset e variante (ogni percorso
    ha la propria configurazione sensata in base a estensione, complessità e difficoltà).
    Passa esplicitamente i parametri per sovrascrivere i default.

    Costruito una sola volta al primo utilizzo (per una data combinazione di
    nome/variante/clearance/n_obstacles) e poi cache-ato: chiamate successive con gli
    stessi parametri ritornano sempre lo stesso oggetto, senza rigenerarlo.

    Args:
        name: nome del preset (vedi PrefabricatedPaths.list_presets()).
        variant: livello di difficoltà o variante (es. "type1", "type2", "type3").
            Default è "type2".
        clearance: distanza minima (m) degli ostacoli dal percorso. Se None,
            usa il default specifico del preset e della variante.
        n_obstacles: numero di ostacoli da piazzare nell'ambiente. Se None,
            usa il default specifico del preset e della variante.
    """
    # Recupera i default specifici per preset e variante
    defaults = get_preset_env_defaults(name, variant=variant)

    # Se non sono sovrascritti manualmente, usa i valori della variante
    if clearance is None:
        clearance = defaults["clearance"]
    if n_obstacles is None:
        n_obstacles = defaults["n_obstacles"]

    # La chiave di cache include anche la variante per evitare collisioni
    key = (name.lower().strip(), variant.lower().strip(), float(clearance), int(n_obstacles))

    if key not in _ENVIRONMENTS_CACHE:
        path = PrefabricatedPaths.get_preset(name)
        _ENVIRONMENTS_CACHE[key] = _build_environment_for_path(
            path, clearance=clearance, n_obstacles=n_obstacles
        )

    return _ENVIRONMENTS_CACHE[key]


def build_all_environments(overrides: dict = None) -> dict:
    """
    Costruisce (o recupera dalla cache) l'ambiente per ogni preset disponibile,
    usando per ciascuno i propri default (PRESET_ENV_CONFIG) salvo diversa
    indicazione.

    Args:
        overrides: dizionario opzionale {nome_preset: {"clearance": ..., "n_obstacles": ...}}
            per personalizzare uno o più preset specifici, lasciando gli altri
            ai valori di default. Non serve specificare entrambe le chiavi:
            si può sovrascrivere solo 'clearance' o solo 'n_obstacles'.

    Esempio:
        # Tutti i preset con i loro default, tranne circle_large che vuole
        # più ostacoli e square che vuole clearance più ampio
        build_all_environments(overrides={
            "circle_large": {"n_obstacles": 30},
            "square": {"clearance": 1.0},
        })
    """
    overrides = overrides or {}
    result = {}
    for name in PrefabricatedPaths.list_presets():
        defaults = get_preset_env_defaults(name)
        cfg = {**defaults, **overrides.get(name, {})}
        result[name] = get_environment_for_preset(
            name, clearance=cfg["clearance"], n_obstacles=cfg["n_obstacles"]
        )
    return result


def plot_all_environments(overrides: dict = None) -> None:
    """
    Utility di verifica visiva che crea un'unica finestra interattiva per scorrere
    tra i diversi tracciati. Per ogni tracciato selezionato, mostra in una griglia
    1x3 le sue tre varianti ("type1", "type2", "type3") affiancate.

    Args:
        overrides: Dizionario opzionale con struttura {nome_preset: {"clearance": ..., "n_obstacles": ...}}
            per personalizzare le configurazioni di uno o più ambienti specifici.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # 1. RECUPERO DEI PRESET E DELLE VARIANTI
    presets = PrefabricatedPaths.list_presets()

    # Raccoglie e ordina le varianti disponibili (type1 -> type2 -> type3)
    all_variants = list(
        {variant for track in PRESET_ENV_CONFIG.values() for variant in track.keys()}
    )
    order_map = {"type1": 0, "type2": 1, "type3": 2}
    all_variants.sort(key=lambda v: order_map.get(v.lower(), 99))

    overrides = overrides or {}

    # 2. INIZIALIZZAZIONE DELLA FIGURA (1 RIGA x 3 COLONNE PER LE 3 VARIANTI)
    fig, axes = plt.subplots(1, len(all_variants), figsize=(16, 5), layout="constrained")
    axes = np.atleast_1d(axes).flatten()

    # Margini di sicurezza per distanziare il titolo in alto e i bottoni in basso
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, hspace=0.05, wspace=0.05)

    # Stato locale per tracciare l'indice del TRACCIATO attivo
    state = {"current_preset_idx": 0}

    # 3. FUNZIONE DI AGGIORNAMENTO DEL GRAFICO PER IL TRACCIATO CORRENTE
    def update_plot():
        preset_name = presets[state["current_preset_idx"]]
        path = PrefabricatedPaths.get_preset(preset_name)

        # Mostra affiancate le 3 varianti (type1, type2, type3) del tracciato corrente
        for i, variant in enumerate(all_variants):
            ax = axes[i]
            ax.clear()  # Pulisce il riquadro prima di ridisegnare

            # Recupera i parametri di default per la variante e applica eventuali override
            defaults = get_preset_env_defaults(preset_name, variant=variant)
            cfg = {**defaults, **overrides.get(preset_name, {})}

            # Carica l'ambiente per il tracciato e la variante specifica
            env = get_environment_for_preset(
                preset_name,
                variant=variant,
                clearance=cfg["clearance"],
                n_obstacles=cfg["n_obstacles"],
            )

            # Disegna gli ostacoli e il percorso
            env.plot(ax=ax)
            ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, label="Percorso")

            # Configura il titolo del singolo sotto-grafico
            ax.set_title(
                f"Variante: {variant.upper()}\n"
                f"(clearance={cfg['clearance']}, n={cfg['n_obstacles']}, piazzati={len(env.obstacles)})",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.set_aspect('equal', 'box')

        # Titolo generale con l'indicazione del tracciato corrente e del progresso
        fig.suptitle(
            f"Tracciato: {preset_name.upper()} ({state['current_preset_idx'] + 1}/{len(presets)})",
            fontsize=14,
            fontweight='bold',
        )
        fig.canvas.draw_idle()

    # 4. CREAZIONE DEI BOTTONI INTERATTIVI (INDIETRO / AVANTI TRACCIATO)
    # Posizionati con uno spazio dedicato in basso per evitare sovrapposizioni
    ax_prev = fig.add_axes([0.38, 0.02, 0.10, 0.05])
    ax_next = fig.add_axes([0.52, 0.02, 0.10, 0.05])

    btn_prev = Button(ax_prev, '◄ Indietro', color='lightgray', hovercolor='0.9')
    btn_next = Button(ax_next, 'Avanti ►', color='lightgray', hovercolor='0.9')

    # Callback per passare al tracciato successivo
    def next_preset(event):
        state["current_preset_idx"] = (state["current_preset_idx"] + 1) % len(presets)
        update_plot()

    # Callback per passare al tracciato precedente
    def prev_preset(event):
        state["current_preset_idx"] = (state["current_preset_idx"] - 1) % len(presets)
        update_plot()

    btn_next.on_clicked(next_preset)
    btn_prev.on_clicked(prev_preset)

    # Disegna il primo tracciato all'apertura
    update_plot()
    plt.show()


if __name__ == "__main__":
    plot_all_environments()