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
"""

import numpy as np
from shapely.geometry import LineString, Point, box, Polygon as ShapelyPolygon

from environment import Environment
from prefabricated_paths import PrefabricatedPaths


# --- Configurazione per-preset di clearance e numero di ostacoli ---
#
# Ogni percorso ha estensione e complessità diverse , quindi invece di
# un unico valore fisso per tutti, ogni preset ha i propri default sensati.
# Questi valori sono comunque sempre sovrascrivibili passando esplicitamente
# 'clearance' e/o 'n_obstacles' alle funzioni sottostanti.
PRESET_ENV_CONFIG: dict = {
    "straight_short":  {"clearance": 0.6, "n_obstacles": 5,  "r_max": 4.0},
    "straight_long":   {"clearance": 0.7, "n_obstacles": 10, "r_max": 6.0},
    "circle_medium":   {"clearance": 0.7, "n_obstacles": 8,  "r_max": 5.0},
    "circle_large":    {"clearance": 0.8, "n_obstacles": 10, "r_max": 7.0},
    "tight_slalom":    {"clearance": 0.5, "n_obstacles": 8,  "r_max": 4.5},
    "wide_slalom":     {"clearance": 0.8, "n_obstacles": 10, "r_max": 6.5},
    "eight":           {"clearance": 0.7, "n_obstacles": 9,  "r_max": 6.0},
    "square":          {"clearance": 0.8, "n_obstacles": 8,  "r_max": 6.5},
    "pista_f1":        {"clearance": 0.8, "n_obstacles": 10, "r_max": 7.0},
}

# Default di fallback per preset non presenti nella tabella sopra (es. nuovi
# preset aggiunti in futuro a PrefabricatedPaths senza una voce dedicata qui)
_DEFAULT_CLEARANCE = 0.8
_DEFAULT_N_OBSTACLES = 15
_DEFAULT_R_MAX = 6.0


def get_preset_env_defaults(name: str) -> dict:
    """Ritorna {'clearance': ..., 'n_obstacles': ..., 'r_max': ...} per il preset
    richiesto, usando i default generici se il preset non ha una voce dedicata."""
    key = name.lower().strip()
    return PRESET_ENV_CONFIG.get(
        key,
        {"clearance": _DEFAULT_CLEARANCE, "n_obstacles": _DEFAULT_N_OBSTACLES, "r_max": _DEFAULT_R_MAX},
    )


def _halton(index: int, base: int) -> float:
    """
    Sequenza di Halton: genera numeri in [0, 1) a bassa discrepanza (ben
    distribuiti, senza clustering) in modo completamente deterministico —
    stesso index e base => stesso valore, sempre. Usata al posto di
    random.uniform per ottenere una distribuzione spaziale naturale e
    riproducibile degli ostacoli.
    """
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _is_safe(env: Environment, geom, path_line: LineString, clearance: float) -> bool:
    """
    Verifica che 'geom' sia un candidato valido per essere aggiunto all'ambiente:
    - a distanza almeno 'clearance' dal percorso (nessuna intersezione, nessun
      restringimento pericoloso del corridoio di sicurezza attorno al path);
    - non sovrapposto a nessun ostacolo già presente;
    - contenuto nei bounds dell'ambiente.
    """
    if geom.distance(path_line) < clearance:
        return False
    for ob in env.obstacles:
        if geom.intersects(ob):
            return False
    if env.bounds is not None:
        try:
            if not env.bounds.contains(geom):
                return False
        except (ValueError, AttributeError, TypeError):
            return False
    return True


def _make_polygon_geom(cx: float, cy: float, r: float, n_sides: int, rotation: float):
    """Costruisce un poligono regolare (n_sides lati) centrato in (cx, cy),
    con raggio r e rotazione iniziale 'rotation' (radianti)."""
    angles = rotation + np.linspace(0.0, 2 * np.pi, n_sides, endpoint=False)
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
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
        clearance: float = None,
        n_obstacles: int = None,
) -> Environment:
    """
    Ritorna l'Environment deterministico associato al preset richiesto.

    Se 'clearance' e/o 'n_obstacles' non vengono specificati, si usano i
    default definiti in PRESET_ENV_CONFIG per quel preset (ogni percorso ha
    la propria configurazione sensata in base a estensione e complessità).
    Passa esplicitamente i parametri per sovrascrivere i default.

    Costruito una sola volta al primo utilizzo (per una data combinazione di
    nome/clearance/n_obstacles) e poi cache-ato: chiamate successive con gli
    stessi parametri ritornano sempre lo stesso oggetto, senza rigenerarlo.

    Args:
        name: nome del preset (vedi PrefabricatedPaths.list_presets()).
        clearance: distanza minima (m) degli ostacoli dal percorso. Se None,
            usa il default specifico del preset.
        n_obstacles: numero di ostacoli da piazzare nell'ambiente. Se None,
            usa il default specifico del preset.
    """
    defaults = get_preset_env_defaults(name)
    if clearance is None:
        clearance = defaults["clearance"]
    if n_obstacles is None:
        n_obstacles = defaults["n_obstacles"]

    key = (name.lower().strip(), float(clearance), int(n_obstacles))
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
    """Utility di verifica visiva: disegna ogni ambiente (con i propri default
    o eventuali override) insieme al proprio percorso."""
    import matplotlib.pyplot as plt

    presets = PrefabricatedPaths.list_presets()
    cols = 3
    rows = (len(presets) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    overrides = overrides or {}
    i = 0
    for i, name in enumerate(presets):
        ax = axes[i]
        defaults = get_preset_env_defaults(name)
        cfg = {**defaults, **overrides.get(name, {})}

        path = PrefabricatedPaths.get_preset(name)
        env = get_environment_for_preset(name, clearance=cfg["clearance"], n_obstacles=cfg["n_obstacles"])
        env.plot(ax=ax)
        ax.plot(path[:, 0], path[:, 1], 'g--', linewidth=2, label="Percorso")
        ax.set_title(
            f"{name} (clearance={cfg['clearance']}, n={cfg['n_obstacles']}, piazzati={len(env.obstacles)})",
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_aspect('equal', 'box')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_all_environments()