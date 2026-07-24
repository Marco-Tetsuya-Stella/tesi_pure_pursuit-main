import gc  # Garbage Collector per liberare la memoria bitmap di Windows
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.spatial.distance import cdist
from contextlib import suppress

from visualizer import draw_robot


# =======================================================
# FUNZIONI HELPER INDIPENDENTI
# =======================================================

def get_unique_filepath(filepath: Path) -> Path:
    """
    Ritorna un percorso file univoco per non sovrascrivere file esistenti.

    crea dei file con questo ordine nomeOriginale, nomeOriginale_1, nomeOriginale_2,
    Args:
        filepath (Path): Il percorso del file originale da verificare.

    Returns:
        Path: Il percorso originale se non esiste, oppure un percorso
              modificato con un suffisso numerico (es. nome_1.ext).
    """
    # Verifica se il percorso originale fornito NON esiste ancora sul disco.
    if not filepath.exists():
        # Se il file non esiste, il percorso originale va bene e viene restituito immediatamente.
        return filepath

    # Estrae solo il nome del file senza l'estensione (es. da "grafico.png" ottiene "grafico").
    stem = filepath.stem

    # Estrae l'estensione del file, incluso il punto (es. ".png").
    suffix = filepath.suffix

    # Ottiene il percorso della cartella che contiene il file (es. "cartella/di/destinazione").
    parent = filepath.parent

    # Inizializza un contatore numerico a 1. Questo numero verrà usato per differenziare i file.
    counter = 1

    # Avvia un ciclo infinito che continuerà finché non trova un nome di file disponibile.
    while True:
        # Crea un nuovo percorso "candidato" unendo cartella, nome originale, contatore ed estensione.
        # Al primo ciclo diventerà ad esempio: "cartella/di/destinazione/grafico_1.png".
        candidate = parent / f"{stem}_{counter}{suffix}"

        # Controlla se questo nuovo percorso candidato è libero (cioè se NON esiste sul disco).
        if not candidate.exists():
            # Se il file non esiste, abbiamo trovato un nome univoco e la funzione restituisce questo nuovo percorso.
            return candidate

        # Se invece il file esiste già, incrementa il contatore di 1 (es. da _1 passa a _2) e il ciclo ricomincia.
        counter += 1


def calc_errors(real, est, path):
    """
    Calcola la distanza minima dal percorso di riferimento (path)
    per ogni punto della traiettoria reale e di quella stimata.

    NOTA: Se la traiettoria ha 100 punti e il percorso ha 500 punti, la funzione cdist calcola in un solo passaggio
    vettorializzato le 100 * 500 = 50.000 distanze incrociate. Successivamente, np.min(..., axis=1) isola per ciascuno
    dei 100 punti del robot la distanza minima assoluta verso la linea di riferimento.
    Args:
        real (np.ndarray): Matrice con i punti della traiettoria reale del robot (N, >=2).
        est (np.ndarray): Matrice con i punti della traiettoria stimata (ICP + Odometria) (M, >=2).
        path (np.ndarray): Matrice con i punti del percorso di riferimento ideale (K, >=2).

    Returns:
        tuple: (dev_real, dev_est) contenenti i vettori delle distanze minime (in metri)
               per ciascuno step temporale delle due traiettorie.
    """
    # Inizializza due liste vuote per contenere gli errori (deviazioni) di percorso
    dev_real, dev_est = [], []

    # --- 1. CALCOLO ERRORE TRAIETTORIA REALE ---
    # Controlla che sia la traiettoria reale sia il path contengano almeno un punto
    if len(real) > 0 and len(path) > 0:
        # Calcola la matrice delle distanze euclidee tra ogni punto di 'real' e ogni punto di 'path'.
        # 'real[:, :2]' estrae solo le coordinate (x, y) ignorando l'orientamento (theta) o z.
        # Se 'real' ha N punti e 'path' ha K punti, 'dists_r' sarà una matrice di forma (N, K).
        dists_r = cdist(real[:, :2], path[:, :2])

        # Per ciascun punto N della traiettoria reale (lungo axis=1, cioè per ogni riga),
        # trova la distanza MINIMA tra quel punto e l'intero insieme di punti del path.
        dev_real = np.min(dists_r, axis=1)

    # --- 2. CALCOLO ERRORE TRAIETTORIA STIMATA ---
    # Controlla che sia la traiettoria stimata sia il path contengano almeno un punto
    if len(est) > 0 and len(path) > 0:
        # Calcola la matrice delle distanze euclidee tra ogni punto di 'est' e ogni punto di 'path' (M x K)
        dists_e = cdist(est[:, :2], path[:, :2])

        # Trova la distanza minima dal path per ciascun punto della traiettoria stimata
        dev_est = np.min(dists_e, axis=1)

    # Restituisce i due array di numpy con le deviazioni calcolate punto per punto
    return dev_real, dev_est


def adjust_axis_limits(ax, path, env=None, real=None, est=None, min_range=5.0):
    """
    Calcola i limiti degli assi per un grafico Matplotlib includendo il percorso di riferimento,
    l'ambiente, gli ostacoli e le traiettorie reali/stimate.

    Args:
        ax (matplotlib.axes.Axes): L'asse del grafico da ridimensionare.
        path (np.ndarray): Matrice con i punti del percorso di riferimento.
        env (object, optional): Oggetto ambiente contenente i confini (bounds) e/o ostacoli.
        real (np.ndarray, optional): Traiettoria reale del robot.
        est (np.ndarray, optional): Traiettoria stimata del robot.
        min_range (float, optional): Dimensione minima garantita della finestra di visualizzazione in metri.
    """
    # Inizializza due liste per raccogliere TUTTE le coordinate X e Y di tutti gli elementi da mostrare
    xs, ys = [], []

    # Se il percorso di riferimento esiste e non è vuoto, ne aggiunge tutte le X e le Y alle liste
    if path is not None and len(path) > 0:
        xs.extend(path[:, 0])
        ys.extend(path[:, 1])

    # Se la traiettoria reale del robot esiste e non è vuota, aggiunge tutte le sue coordinate X e Y
    if real is not None and len(real) > 0:
        xs.extend(real[:, 0])
        ys.extend(real[:, 1])

    # Se la traiettoria stimata esiste e non è vuota, aggiunge tutte le sue coordinate X e Y
    if est is not None and len(est) > 0:
        xs.extend(est[:, 0])
        ys.extend(est[:, 1])

    # Se è stato fornito un ambiente virtuale (env)
    if env is not None:
        # Controlla se l'oggetto 'env' possiede una proprietà 'bounds' definita
        if getattr(env, 'bounds', None) is not None:
            try:
                # Estrae i limiti minimi e massimi della mappa dell'ambiente (x_min, y_min, x_max, y_max)
                bx_min, by_min, bx_max, by_max = env.bounds.bounds
                xs.extend([bx_min, bx_max])
                ys.extend([by_min, by_max])
            except Exception:
                pass  # Se l'estrazione fallisce, ignora l'errore e prosegue

        # Controlla se l'ambiente ha un elenco di ostacoli
        if hasattr(env, 'obstacles'):
            for ob in env.obstacles:
                try:
                    # Per ciascun ostacolo, estrae i confini della sua bounding box e li aggiunge alle liste
                    ox_min, oy_min, ox_max, oy_max = ob.bounds
                    xs.extend([ox_min, ox_max])
                    ys.extend([oy_min, oy_max])
                except Exception:
                    pass  # Ignora ostacoli mal formattati

    # Se non è stata trovata alcuna coordinata valida, interrompe la funzione senza modificare il grafico
    if not xs or not ys:
        return

    # Calcola il valore minimo e massimo assoluto tra tutte le coordinate X e Y raccolte
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Calcola l'ampiezza dell'ingombro totale lungo l'asse X e l'asse Y
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Calcola il punto centrale dell'intera scena (centroide del bounding box totale)
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    # Determina il semi-raggio (metà ampiezza): sceglie il valore massimo tra l'ampiezza X, l'ampiezza Y
    # e il parametro 'min_range', per poi dividerlo per 2
    max_r = max(x_range, y_range, min_range) / 2.0

    # Calcola un piccolo margine di bordo (padding dell'8%) per evitare che gli elementi tocchino i bordi
    pad = max_r * 0.08

    # Imposta i limiti di visualizzazione sull'asse X centrando la vista e applicando il semi-raggio con margine
    ax.set_xlim(x_center - max_r - pad, x_center + max_r + pad)

    # Imposta i limiti di visualizzazione sull'asse Y usando la stessa ampiezza usata per X
    ax.set_ylim(y_center - max_r - pad, y_center + max_r + pad)

    # Imposta il rapporto di forma (aspect ratio) 1:1 affinché i metri su X siano visivamente identici a Y
    ax.set_aspect('equal', adjustable='box')


def plot_environment(ax, env):
    """
    Disegna gli ostacoli dell'ambiente sul grafico Matplotlib fornito.

    Args:
        ax (matplotlib.axes.Axes): L'asse del grafico su cui disegnare gli ostacoli.
        env (object): L'oggetto ambiente contenente la lista/collezione degli ostacoli.
    """
    # Verifica di sicurezza: se l'ambiente non è definito (None) o non possiede
    # l'attributo 'obstacles', interrompe l'esecuzione ed esce dalla funzione.
    if env is None or not hasattr(env, 'obstacles'):
        return

    # Cicla attraverso ciascun ostacolo presente nella lista degli ostacoli dell'ambiente
    for obstacle in env.obstacles:
        # Estrae le coordinate X e Y del perimetro esterno dell'ostacolo.
        # Presuppone che gli ostacoli siano oggetti geometrici di tipo Shapely (Polygon).
        x_obs, y_obs = obstacle.exterior.xy

        # Disegna l'ostacolo come una figura geometrica piena (poligono):
        # - color='gray': riempimento di colore grigio
        # - alpha=0.5: trasparenza al 50% per non coprire del tutto lo sfondo
        # - edgecolor='black': bordo esterno di colore nero
        # - linewidth=0.8: spessore del bordo pari a 0.8 punti
        ax.fill(x_obs, y_obs, color='gray', alpha=0.5, edgecolor='black', linewidth=0.8)


# =======================================================
# SALVATAGGIO INCREMENTALE (GRUPPO PER GRUPPO)
# =======================================================

import gc
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def export_group_plots(group, base_dir=None):
    """
    Esporta i grafici di un SINGOLO gruppo di esperimenti liberando immediatamente
    la memoria RAM e GDI bitmap per prevenire 'Fail to allocate bitmap'.

    Args:
        group (dict): Dizionario contenente i dati e i risultati dell'esperimento.
                      Deve contenere i metadati ('path_name', 'variant', 'ld')
                      e i risultati delle 4 varianti di simulazione ('ideal_no_lc',
                      'ideal_lc', 'noisy_no_lc', 'noisy_lc').
        base_dir (str | Path, optional): Il percorso della cartella principale in cui
                                         salvare le immagini esportate. Se None, viene
                                         creata una cartella di default.

    Returns:
        None: La funzione non restituisce valori, esegue solo il salvataggio dei file
              su disco come effetto collaterale.
    """
    # Se non viene fornita una cartella di base, ne crea una di default chiamata "img_pure_pursuit"
    # salendo di due livelli rispetto a dove si trova questo script
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "img_pure_pursuit"
    else:
        # Altrimenti usa la cartella fornita convertendola in oggetto Path
        base_dir = Path(base_dir)

    # Definisce i percorsi per 5 sotto-cartelle dove verranno smistati i vari tipi di grafici
    dir_completa = base_dir / "paths + stima + reale"
    dir_stima = base_dir / "paths + stima"
    dir_reale = base_dir / "paths + reale"
    dir_scostamenti = base_dir / "analisi scostamenti"
    dir_riepilogo = base_dir / "riepilogo degli esperimenti"

    # Crea fisicamente tutte le sotto-cartelle sul disco (se non esistono già)
    for d in [dir_completa, dir_stima, dir_reale, dir_scostamenti, dir_riepilogo]:
        d.mkdir(parents=True, exist_ok=True)

    # Chiavi usate per estrarre i dati dei 4 test dal dizionario 'group'
    exp_keys = ['ideal_no_lc', 'ideal_lc', 'noisy_no_lc', 'noisy_lc']

    # Etichette usate per comporre i nomi dei file salvati
    exp_labels = [
        "1_Odometria_Ideale_No_LC",
        "2_Odometria_Ideale_Con_LC",
        "3_Odometria_Rumorosa_No_LC",
        "4_Odometria_Rumorosa_Con_LC"
    ]

    # Titoli descrittivi che verranno stampati in cima a ogni grafico
    exp_titles = [
        "1. ODOMETRIA IDEALE | SENZA Loop Closure",
        "2. ODOMETRIA IDEALE | CON Loop Closure",
        "3. ODOMETRIA RUMOROSA | SENZA Loop Closure",
        "4. ODOMETRIA RUMOROSA | CON Loop Closure"
    ]

    # Estrae i metadati dell'esperimento: nome del percorso, variante e distanza di Look-Ahead
    path_name = group['path_name']
    variant = group['variant']
    ld = group['ld']

    # Crea un prefisso testuale che accomunerà tutti i file di questo gruppo (es. "cubo_pp_ld2.0")
    prefix = f"{path_name}_{variant}_ld{ld}"

    # Salva lo stato attuale della modalità interattiva di Matplotlib per ripristinarla alla fine
    was_interactive = plt.isinteractive()

    # Disattiva la modalità interattiva: i grafici vengono disegnati in background senza aprire finestre,
    # risparmiando moltissima RAM e CPU.
    plt.ioff()

    # Blocco try-finally per garantire che la memoria venga SEMPRE liberata in caso di errori
    try:
        # --- 1, 2, 3, 4: GRAFICI PER SINGOLO ESPERIMENTO ---
        # Cicla attraverso i 4 test (ideal no-lc, ideal lc, noisy no-lc, noisy lc)
        for k_idx, k in enumerate(exp_keys):
            # Estrae i dati specifici del test corrente
            res = group[k]
            env = res['env']
            path = res['path']
            est = res['estimated_history']
            real = res['robot_history']
            exp_label = exp_labels[k_idx]
            exp_title = exp_titles[k_idx]

            # ==========================================
            # 1. GRAFICO: PATH + STIMA + REALE
            # ==========================================
            fig, ax = plt.subplots(figsize=(10, 8))  # Crea figura 10x8 pollici
            plot_environment(ax, env)  # Disegna gli ostacoli
            ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')  # Path in verde tratteggiato

            if len(real) > 0:  # Se c'è la traiettoria reale, la disegna come linea blu
                ax.plot(real[:, 0], real[:, 1], 'b-', linewidth=1.5, label='Traiettoria Reale')
            if len(est) > 0:  # Se c'è la stima, la disegna come punti rossi
                ax.plot(est[:, 0], est[:, 1], 'r.', markersize=4, label='Stima ICP + Odom')

            # Imposta titolo, legenda, griglia e adatta l'inquadratura del grafico
            ax.set_title(
                f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\n{exp_title}\nVista: Path + Stima + Reale",
                fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True)
            adjust_axis_limits(ax, path, env=env, real=real, est=est, min_range=5.0)
            fig.tight_layout()  # Ottimizza gli spazi interni del grafico

            # Ottiene un nome file sicuro per non sovrascrivere, salva a 120 DPI, e CHIUDE la figura
            save_p = get_unique_filepath(dir_completa / f"{prefix}_{exp_label}_completa.png")
            fig.savefig(save_p, dpi=120)
            plt.close(fig)

            # ==========================================
            # 2. GRAFICO: PATH + STIMA
            # ==========================================
            # Ripete le stesse identiche operazioni di prima, ma omette la traiettoria Reale (blu)
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_environment(ax, env)
            ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
            if len(est) > 0:
                ax.plot(est[:, 0], est[:, 1], 'r.', markersize=4, label='Stima ICP + Odom')
            ax.set_title(
                f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\n{exp_title}\nVista: Path + Stima",
                fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True)
            adjust_axis_limits(ax, path, env=env, real=real, est=est, min_range=5.0)
            fig.tight_layout()
            save_p = get_unique_filepath(dir_stima / f"{prefix}_{exp_label}_stima.png")
            fig.savefig(save_p, dpi=120)
            plt.close(fig)

            # ==========================================
            # 3. GRAFICO: PATH + REALE
            # ==========================================
            # Ripete le stesse operazioni, ma omette la stima ICP (rossa)
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_environment(ax, env)
            ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
            if len(real) > 0:
                ax.plot(real[:, 0], real[:, 1], 'b-', linewidth=1.5, label='Traiettoria Reale')
            ax.set_title(
                f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\n{exp_title}\nVista: Path + Reale",
                fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True)
            adjust_axis_limits(ax, path, env=env, real=real, est=est, min_range=5.0)
            fig.tight_layout()
            save_p = get_unique_filepath(dir_reale / f"{prefix}_{exp_label}_reale.png")
            fig.savefig(save_p, dpi=120)
            plt.close(fig)

            # ==========================================
            # 4. GRAFICI: ANALISI SCOSTAMENTI
            # ==========================================
            # Calcola l'errore metrico istante per istante usando la funzione calc_errors
            dev_real, dev_est = calc_errors(real, est, path)

            # --- Sotto-Grafico: Errore Logaritmico ---
            fig, ax = plt.subplots(figsize=(10, 6))
            # Sostituisce eventuali errori pari a 0 con 1e-6 (un milionesimo) per evitare errori matematici
            # nel calcolo del logaritmo (log(0) è indefinito)
            dev_real_log = np.where(np.array(dev_real) == 0, 1e-6, dev_real)
            dev_est_log = np.where(np.array(dev_est) == 0, 1e-6, dev_est)

            if len(dev_real_log) > 0:
                ax.plot(dev_real_log, 'b-', label='Errore Traiettoria Reale', alpha=0.8, linewidth=1.5)
            if len(dev_est_log) > 0:
                ax.plot(dev_est_log, 'r--', label='Errore Stima ICP+Odom', alpha=0.8, linewidth=1.5)

            ax.set_yscale('log')  # Imposta l'asse Y in scala logaritmica
            ax.set_xlabel("Step di Simulazione", fontsize=11)
            ax.set_ylabel("Distanza dal Path (metri)", fontsize=11)
            ax.set_title(
                f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\n{exp_title}\nVista: Analisi Scostamenti (Scala Logaritmica)",
                fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, which="both", ls="--", alpha=0.6)
            fig.tight_layout()
            save_p = get_unique_filepath(dir_scostamenti / f"{prefix}_{exp_label}_errore_log.png")
            fig.savefig(save_p, dpi=120)
            plt.close(fig)

            # --- Sotto-Grafico: Errore Lineare ---
            fig, ax = plt.subplots(figsize=(10, 6))
            if len(dev_real) > 0:
                ax.plot(dev_real, 'b-', label='Errore Traiettoria Reale', alpha=0.8, linewidth=1.5)
            if len(dev_est) > 0:
                ax.plot(dev_est, 'r--', label='Errore Stima ICP+Odom', alpha=0.8, linewidth=1.5)

            ax.set_xlabel("Step di Simulazione", fontsize=11)
            ax.set_ylabel("Distanza dal Path (metri)", fontsize=11)
            ax.set_title(
                f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\n{exp_title}\nVista: Analisi Scostamenti (Scala Lineare)",
                fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            save_p = get_unique_filepath(dir_scostamenti / f"{prefix}_{exp_label}_errore_lineare.png")
            fig.savefig(save_p, dpi=120)
            plt.close(fig)

        # ==========================================
        # 5. RIEPILOGO DEGLI ESPERIMENTI (Bar Charts)
        # ==========================================
        means_real, means_est = [], []
        sums_real, sums_est = [], []

        # Raccoglie media e somma degli errori per i 4 esperimenti
        for k in exp_keys:
            r, e = calc_errors(group[k]['robot_history'], group[k]['estimated_history'], group[k]['path'])
            means_real.append(np.mean(r) if len(r) > 0 else 0)
            means_est.append(np.mean(e) if len(e) > 0 else 0)
            sums_real.append(np.sum(r) if len(r) > 0 else 0)
            sums_est.append(np.sum(e) if len(e) > 0 else 0)

        # Prepara gli indici e le etichette per gli assi X dei grafici a barre
        x = np.arange(4)
        width = 0.35
        labels_bar = [
            "Odometria Ideale\nNo LC",
            "Odometria Ideale\nCon LC",
            "Odometria Rumorosa\nNo LC",
            "Odometria Rumorosa\nCon LC"
        ]

        # --- Grafico a Barre: Riepilogo Media ---
        fig, ax = plt.subplots(figsize=(10, 6))
        # Crea le barre affiancate spostandole di "width/2" a sinistra (reale) e a destra (stima)
        rects1 = ax.bar(x - width / 2, means_real, width, label='Media Errore Reale', color='blue', alpha=0.7)
        rects2 = ax.bar(x + width / 2, means_est, width, label='Media Errore Stima ICP+Odom', color='red', alpha=0.7)
        ax.set_title(
            f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\nRIEPILOGO ESPERIMENTI: Media degli Errori",
            fontweight='bold', fontsize=13)
        ax.set_ylabel("Errore Medio (metri)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Aggiunge i numeretti esatti (annotate) sopra ogni barra
        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

        fig.tight_layout()
        save_p = get_unique_filepath(dir_riepilogo / f"{prefix}_riepilogo_media.png")
        fig.savefig(save_p, dpi=120)
        plt.close(fig)

        # --- Grafico a Barre: Riepilogo Somma ---
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, sums_real, width, label='Somma Errore Reale', color='blue', alpha=0.7)
        rects2 = ax.bar(x + width / 2, sums_est, width, label='Somma Errore Stima ICP+Odom', color='red', alpha=0.7)
        ax.set_title(
            f"Path: '{path_name}' | Var: {variant.upper()} | Look-Ahead: {ld}m\nRIEPILOGO ESPERIMENTI: Somma Totale degli Errori",
            fontweight='bold', fontsize=13)
        ax.set_ylabel("Errore Totale Cumulato (metri)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_bar, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        fig.tight_layout()
        save_p = get_unique_filepath(dir_riepilogo / f"{prefix}_riepilogo_somma.png")
        fig.savefig(save_p, dpi=120)
        plt.close(fig)

    finally:
        # Questa sezione viene ESEGUITA SEMPRE, anche se il codice sopra si interrompe per un errore
        plt.close('all')  # Forza la chiusura di tutte le figure Matplotlib rimaste aperte in memoria
        gc.collect()  # Chiama il Garbage Collector (spazzino della memoria) per liberare la RAM/GDI

        # Se la modalità interattiva era attiva all'inizio, la riaccende
        if was_interactive:
            plt.ion()


# =======================================================
# CLASSE VISUALIZZATORE INTERATTIVO
# =======================================================

class InteractiveVisualizer:
    def __init__(self, experiment_groups):
        """
        Inizializza l'interfaccia grafica interattiva per navigare tra i risultati delle simulazioni.

        Args:
            experiment_groups (list o dict): Struttura dati (lista o dizionario) che contiene
                                         tutti i risultati, i metadati e i parametri degli
                                         esperimenti raggruppati per tipologia di simulazione.
        """
        # Salva i dati di tutti i gruppi di esperimenti passati in input
        self.groups = experiment_groups

        # Indice per tenere traccia di quale "gruppo" stiamo visualizzando (es. cubo, random, ecc.)
        self.current_group_idx = 0

        # Indice per l'esperimento corrente all'interno del gruppo:
        # 0=Odometria Ideale No LC, 1=Odometria Ideale LC, 2=Odometria Rumorosa No LC,
        # 3=Odometria Rumorosa LC, 4=Grafici di RIEPILOGO finali
        self.current_exp_idx = 0

        # Indice per la vista specifica (come visualizzare i dati dell'esperimento scelto):
        # Per exp 0-3: 0=Completa, 1=Scomposta, 2=Errore log, 3=Errore lineare, 4=Auto Animata
        # Per exp 4 (Riepilogo): 0=Media, 1=Somma
        self.current_view_idx = 0

        # Crea la finestra principale di Matplotlib definendone le dimensioni (18x10 pollici)
        self.fig = plt.figure(figsize=(18, 10))
        # Imposta il titolo della finestra del sistema operativo
        self.fig.canvas.manager.set_window_title('Pure Pursuit - Simulazioni (Interattivo)')

        # --- Configurazione dei bottoni UI per la navigazione ---
        # Definisce le posizioni e le dimensioni dei pulsanti [sinistra, basso, larghezza, altezza]
        ax_prev_grp = plt.axes([0.02, 0.02, 0.08, 0.05])
        ax_next_grp = plt.axes([0.11, 0.02, 0.08, 0.05])

        ax_prev_exp = plt.axes([0.22, 0.02, 0.09, 0.05])
        ax_next_exp = plt.axes([0.32, 0.02, 0.09, 0.05])

        ax_prev_viw = plt.axes([0.79, 0.02, 0.08, 0.05])
        ax_next_viw = plt.axes([0.88, 0.02, 0.08, 0.05])

        # Crea fisicamente i pulsanti Matplotlib assegnando loro l'area appena definita e un'etichetta
        self.btn_prev_grp = Button(ax_prev_grp, '< Gruppo')
        self.btn_next_grp = Button(ax_next_grp, 'Gruppo >')

        self.btn_prev_exp = Button(ax_prev_exp, '< Esperimento')
        self.btn_next_exp = Button(ax_next_exp, 'Esperimento >')

        self.btn_prev_viw = Button(ax_prev_viw, '< Vista')
        self.btn_next_viw = Button(ax_next_viw, 'Vista >')

        # Collega il click di ciascun pulsante alla rispettiva funzione della classe (es. prev_group)
        self.btn_prev_grp.on_clicked(self.prev_group)
        self.btn_next_grp.on_clicked(self.next_group)
        self.btn_prev_exp.on_clicked(self.prev_exp)
        self.btn_next_exp.on_clicked(self.next_exp)
        self.btn_prev_viw.on_clicked(self.prev_view)
        self.btn_next_viw.on_clicked(self.next_view)

        # --- Bottoni dedicati esclusivamente alla vista "Auto Animata" ---
        ax_play = plt.axes([0.44, 0.02, 0.06, 0.05])
        ax_pause = plt.axes([0.51, 0.02, 0.06, 0.05])
        ax_step = plt.axes([0.58, 0.02, 0.06, 0.05])

        self.btn_play = Button(ax_play, 'Play >')
        self.btn_pause = Button(ax_pause, 'Pausa')
        self.btn_step = Button(ax_step, 'Step >')

        # Collega i comandi di riproduzione animazione alle relative funzioni
        self.btn_play.on_clicked(self.play_animation)
        self.btn_pause.on_clicked(self.pause_animation)
        self.btn_step.on_clicked(self.step_animation)

        # Nasconde di default i bottoni dell'animazione (verranno mostrati solo quando serve)
        for anim_ax in (ax_play, ax_pause, ax_step):
            anim_ax.set_visible(False)

        # --- Inizializzazione delle variabili di stato per l'animazione ---
        self.anim_hist = None  # Conterrà la storia della traiettoria da animare
        self.anim_frame = 0  # Fotogramma corrente (step di simulazione attuale)
        self.anim_playing = False  # Flag booleano: True se l'animazione è in corso, False se in pausa
        self.anim_ax = None  # L'asse specifico su cui si sta disegnando l'animazione
        self.robot_artists = []  # Lista per tenere traccia delle figure del robot disegnate a schermo
        self.anim_text = None  # Oggetto testo (es. "Step 5/100") mostrato sul grafico

        # Crea un timer di Matplotlib che scatterà ogni 40 millisecondi (circa 25 FPS)
        self.anim_timer = self.fig.canvas.new_timer(interval=40)
        # Collega il timer alla funzione che disegna il frame successivo
        self.anim_timer.add_callback(self._on_anim_timer)

        # Lista vuota in cui verranno salvati tutti i grafici (assi) attivi sullo schermo
        # per poterli cancellare comodamente quando si cambia vista
        self.ax_grid = []

    # --- Logica per la vista "Auto Animata" ---
    def _stop_animation(self):
        """
        Ferma l'animazione se è in esecuzione, arrestando il timer in modo sicuro.

        """
        # Controlla se l'animazione è attualmente considerata "in riproduzione" (True)
        if self.anim_playing:
            # Aggiorna immediatamente lo stato interno impostandolo a False (animazione ferma)
            self.anim_playing = False

            # Apre un blocco "sicuro". suppress(Exception) intercetta e ignora
            # qualsiasi errore si verifichi al suo interno, evitando che il programma vada in crash.
            with suppress(Exception):
                # Tenta di fermare fisicamente il timer di Matplotlib responsabile
                # di chiamare la funzione di aggiornamento dei fotogrammi.
                self.anim_timer.stop()

    def _setup_animation(self, ax, env, path, est, real, exp_name):
        """
        Inizializza lo scenario e gli elementi grafici necessari per l'animazione.

        Args:
            ax (matplotlib.axes.Axes): L'asse di Matplotlib su cui predisporre l'animazione.
            env (object): L'oggetto ambiente contenente gli ostacoli.
            path (np.ndarray): Punti del percorso di riferimento.
            est (np.ndarray): Punti della traiettoria stimata.
            real (np.ndarray): Punti della traiettoria reale usati per muovere il robot.
            exp_name (str): Titolo o descrizione dell'esperimento corrente.

        Returns:
            None: Modifica lo stato dell'oggetto ed esegue il setup sul grafico.
        """
        # Memorizza l'asse del grafico su cui verrà disegnata l'animazione
        self.anim_ax = ax

        # Se la traiettoria reale contiene dati la usa per l'animazione,
        # altrimenti assegna un array di ripiego di 1 punto nullo (x=0, y=0, theta=0)
        self.anim_hist = real if len(real) > 0 else np.zeros((1, 3))

        # Resetta l'indice del fotogramma corrente all'inizio (step 0)
        self.anim_frame = 0

        # Inizializza/svuota la lista per i riferimenti ai componenti visivi del robot
        self.robot_artists = []

        # Disegna gli ostacoli dell'ambiente sull'asse
        plot_environment(ax, env)

        # Traccia il percorso ideale in verde tratteggiato
        ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')

        # Se presente, traccia l'intera traiettoria stimata con puntini rossi semitrasparenti
        if len(est) > 0:
            ax.plot(est[:, 0], est[:, 1], 'r.', markersize=3, alpha=0.5, label='Stima ICP + Odom')

        # Traccia l'intera traiettoria reale (lo "storico") con una linea blu sottile e sfumata
        ax.plot(real[:, 0], real[:, 1], 'b-', linewidth=0.8, alpha=0.4, label='Traiettoria Reale')

        # Imposta titolo, legenda e griglia dell'area di lavoro
        ax.set_title(f"{exp_name}\nVista: Auto Animata", fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True)

        # Scala e centra gli assi in modo da inquadrare correttamente l'intera scena (1:1)
        adjust_axis_limits(ax, path, env=env, real=real, est=est, min_range=5.0)

        # Crea una casella di testo vuota nell'angolo in alto a sinistra (2% x, 98% y in coordinate relative)
        # che verrà aggiornata durante l'animazione con i dettagli dello step corrente
        self.anim_text = ax.text(
            0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Disegna lo stato iniziale del robot al fotogramma 0
        self._draw_robot_at_frame()

    def _draw_robot_at_frame(self):
        """
        Ridisegna la posizione del robot e aggiorna l'interfaccia per il fotogramma (frame) corrente.

        """
        # Controlla che gli oggetti essenziali dell'animazione esistano; se mancano, esce per evitare crash
        if self.anim_ax is None or self.anim_hist is None:
            return

        # Cancella i componenti grafici del robot disegnati nel fotogramma precedente
        for artist in self.robot_artists:
            with suppress(Exception):
                artist.remove()  # Rimuove l'elemento visivo dall'asse Matplotlib
        self.robot_artists = []  # Svuota la lista delle referenze

        # Determina la lunghezza della traiettoria e calcola l'indice di frame valido (evita 'out of index')
        n = len(self.anim_hist)
        f = min(self.anim_frame, n - 1)

        # Estrae lo stato del robot al frame corrente (es. coordinata [x, y, theta])
        state = self.anim_hist[f]

        # Calcola una scala dinamica per le dimensioni del robot basata sulla larghezza del grafico,
        # garantendo che il robot mantenga sempre una dimensione leggibile e proporzionata allo zoom
        xlim = self.anim_ax.get_xlim()
        scale = max(0.08, (xlim[1] - xlim[0]) * 0.02)

        # Disegna il robot nella nuova posizione richiamando la funzione helper 'draw_robot'
        # e memorizza i nuovi oggetti grafici ritornati per poterli rimuovere al frame successivo
        self.robot_artists = draw_robot(self.anim_ax, state, robot_radius=scale, color='tab:blue')

        # Aggiorna il testo informativo in alto a sinistra mostrando lo step attuale sul totale
        if self.anim_text is not None:
            self.anim_text.set_text(f"Step {f + 1} / {n}")

        # Notifica a Matplotlib di aggiornare la finestra grafica non appena il sistema è inattivo (idle),
        # garantendo un rendering fluido ed efficiente senza bloccare l'interfaccia
        self.fig.canvas.draw_idle()

    def _on_anim_timer(self):
        """
        Funzione di callback (tick) del timer: fa avanzare l'animazione di un fotogramma.

        """
        # Controlla se la traiettoria da animare esiste; se è assente, interrompe l'esecuzione
        if self.anim_hist is None:
            return

        # Calcola il numero totale di passi (frame) della traiettoria
        n = len(self.anim_hist)

        # Se il frame attuale NON è ancora l'ultimo dello storico:
        if self.anim_frame < n - 1:
            # Passa al fotogramma successivo
            self.anim_frame += 1
            # Ridisegna il robot nella nuova posizione e aggiorna l'interfaccia
            self._draw_robot_at_frame()
        else:
            # Se si è raggiunto l'ultimo frame della simulazione, arresta l'animazione
            self._stop_animation()

    def play_animation(self, event):
        """
        Gestisce l'evento di click sul pulsante "Play", avviando la riproduzione dell'animazione.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Controlla le condizioni di guardia:
        # 1. current_exp_idx >= 4 indica che ci troviamo nella schermata di RIEPILOGO (non un test singolo)
        # 2. current_view_idx != 4 indica che NON siamo nella vista "Auto Animata"
        # Se una delle due condizioni è vera, interrompe la funzione senza fare nulla.
        if self.current_exp_idx >= 4 or self.current_view_idx != 4:
            return

        # Se l'animazione al momento NON è in esecuzione (anim_playing è False):
        if not self.anim_playing:
            # Aggiorna lo stato interno per indicare che l'animazione è ora attiva
            self.anim_playing = True
            # Fai partire il timer di Matplotlib, che inizierà a chiamare 'self._on_anim_timer' a intervalli regolari
            self.anim_timer.start()

    def pause_animation(self, event):
        """
        Gestisce l'evento di click sul pulsante "Pausa", mettendo in pausa l'animazione.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Richiama il metodo di utilità interno per fermare il timer
        # e impostare lo stato 'anim_playing' a False in modo sicuro
        self._stop_animation()

    def step_animation(self, event):
        """
        Gestisce l'evento di click sul pulsante "Step", facendo avanzare l'animazione
        manuale di un singolo fotogramma.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Controlla le condizioni di guardia:
        # Se ci troviamo nella schermata di RIEPILOGO (idx >= 4) oppure NON siamo nella vista animata (idx != 4),
        # interrompe l'esecuzione ed esce senza compiere alcuna azione.
        if self.current_exp_idx >= 4 or self.current_view_idx != 4:
            return

        # Mette in pausa l'animazione automatica (se era in esecuzione) arrestando il timer
        self._stop_animation()

        # Verifica che la storia della traiettoria da animare sia valida e contenga dati
        if self.anim_hist is not None:
            # Calcola il numero totale di passi (frame) della traiettoria
            n = len(self.anim_hist)

            # Incrementa il frame di 1 e applica l'operatore modulo (%) rispetto alla lunghezza totale.
            # Questo fa avanzare il robot e lo riporta automaticamente allo step 0 quando raggiunge la fine.
            self.anim_frame = (self.anim_frame + 1) % n

            # Ridisegna il robot nella posizione corrispondente al nuovo frame calcolato
            self._draw_robot_at_frame()

    def show(self):
        """
        Renderizza i grafici/interfaccia corrente e apre la finestra interattiva di Matplotlib.
        """
        # Chiama il metodo di rendering interno della classe per costruire/disegnare
        # la griglia dei grafici e l'interfaccia corrente in base agli indici selezionati
        self.render()

        # Apre fisicamente la finestra grafica di Matplotlib a schermo e blocca l'esecuzione
        # dello script finché l'utente non chiude la finestra principale
        plt.show()

    # --- Logica di navigazione ---
    def prev_group(self, event):
        """
        Gestisce l'evento di click sul pulsante "< Gruppo", passando al gruppo di esperimenti precedente.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Decrementa l'indice del gruppo corrente di 1.
        # L'operatore modulo garantisce la navigazione circolare all'indietro.
        self.current_group_idx = (self.current_group_idx - 1) % len(self.groups)

        # Resetta l'indice dell'esperimento sul primo della nuova serie (0)
        self.current_exp_idx = 0

        # Resetta l'indice della vista sulla vista principale (0)
        self.current_view_idx = 0

        # Aggiorna l'interfaccia grafica per mostrare i dati del nuovo gruppo
        self.render()

    def next_group(self, event):
        """
        Gestisce l'evento di click sul pulsante "Gruppo >", passando al gruppo di esperimenti successivo.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Incrementa l'indice del gruppo corrente di 1.
        # L'operatore modulo (% len(self.groups)) garantisce la navigazione circolare:
        # arrivati all'ultimo gruppo della lista, il click successivo riparte dal primo (indice 0).
        self.current_group_idx = (self.current_group_idx + 1) % len(self.groups)

        # Resetta l'indice dell'esperimento sul primo della nuova serie (0)
        self.current_exp_idx = 0

        # Resetta l'indice della vista sulla vista principale (0)
        self.current_view_idx = 0

        # Aggiorna l'interfaccia grafica per mostrare i dati del nuovo gruppo
        self.render()

    def prev_exp(self, event):
        """
        Gestisce l'evento di click sul pulsante "< Esperimento", passando
        all'esperimento precedente all'interno dello stesso gruppo.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Decrementa l'indice dell'esperimento corrente di 1.
        # L'operatore modulo (% 5) gestisce i 5 stati possibili (0=Ideale No LC, 1=Ideale LC,
        # 2=Rumorosa No LC, 3=Rumorosa LC, 4=RIEPILOGO) in modo circolare:
        # se si passa indietro dallo stato 0, si finisce direttamente al RIEPILOGO (stato 4).
        self.current_exp_idx = (self.current_exp_idx - 1) % 5

        # Resetta l'indice della vista sulla prima vista disponibile (0) dell'esperimento selezionato
        self.current_view_idx = 0

        # Aggiorna l'interfaccia grafica per mostrare i dati dell'esperimento precedente
        self.render()

    def next_exp(self, event):
        """
        Gestisce l'evento di click sul pulsante "Esperimento >", passando all'esperimento successivo
        del gruppo corrente (o alla schermata di riepilogo).

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Incrementa l'indice dell'esperimento corrente di 1.
        # L'operatore modulo (% 5) gestisce i 5 stati possibili (0=Ideale No LC, 1=Ideale LC,
        # 2=Rumorosa No LC, 3=Rumorosa LC, 4=RIEPILOGO) in modo circolare:
        # se si passa indietro dallo stato 0, si finisce direttamente al RIEPILOGO (stato 4).
        self.current_exp_idx = (self.current_exp_idx + 1) % 5

        # Resetta l'indice della vista sulla prima vista disponibile (0) per il nuovo esperimento scelto
        self.current_view_idx = 0

        # Aggiorna l'interfaccia grafica per mostrare il nuovo esperimento selezionato
        self.render()

    def prev_view(self, event):
        """
        Gestisce l'evento di click sul pulsante "< Vista", passando alla vista precedente
        per l'esperimento o riepilogo corrente.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Determina il numero massimo di viste disponibili per la modalità corrente:
        # - Se current_exp_idx < 4 (singolo esperimento): ci sono 5 viste (0..4)
        #   [0=Completa, 1=Scomposta, 2=Errore log, 3=Errore lineare, 4=Auto Animata]
        # - Se current_exp_idx == 4 (schermata di RIEPILOGO): ci sono solo 2 viste (0..1)
        #   [0=Media, 1=Somma]
        max_v = 5 if self.current_exp_idx < 4 else 2

        # Decrementa l'indice della vista di 1.
        # L'operatore modulo (% max_v) gestisce la navigazione circolare all'indietro:
        # se l'utente si trova sulla prima vista (0), passa automaticamente all'ultima vista disponibile (max_v - 1).
        self.current_view_idx = (self.current_view_idx - 1) % max_v

        # Aggiorna la schermata ridisegnando i grafici relativi alla nuova vista selezionata
        self.render()

    def next_view(self, event):
        """
        Gestisce l'evento di click sul pulsante "Vista >", passando alla vista successiva
        per l'esperimento o il riepilogo corrente.

        Args:
            event (matplotlib.backend_bases.MouseEvent): L'oggetto evento generato
                                                         dal click del mouse sul bottone.
        """
        # Determina il numero massimo di viste disponibili in base alla schermata attuale:
        # - Se current_exp_idx < 4 (singolo esperimento): ci sono 5 viste (0..4)
        #   [0=Completa, 1=Scomposta, 2=Errore log, 3=Errore lineare, 4=Auto Animata]
        # - Se current_exp_idx == 4 (schermata di RIEPILOGO): ci sono solo 2 viste (0..1)
        #   [0=Media, 1=Somma]
        max_v = 5 if self.current_exp_idx < 4 else 2

        # Incrementa l'indice della vista di 1.
        # L'operatore modulo (% max_v) gestisce la navigazione circolare in avanti:
        # arrivati all'ultima vista disponibile (max_v - 1), il click successivo riparte dalla prima (0).
        self.current_view_idx = (self.current_view_idx + 1) % max_v

        # Aggiorna l'interfaccia grafica per mostrare la nuova vista selezionata
        self.render()

    # --- Render Principale ---
    def render(self):
        """
        Ridisegna completamente l'interfaccia grafica in base allo stato corrente
        (gruppo, esperimento e vista selezionati).

        Args:
            self: Riferimento all'istanza della classe.

        Returns:
            None: Pulisce la finestra grafica, rigenera i sotto-grafici (axes),
                  aggiorna i controlli dell'animazione e richiede il ridisegno del canvas.
        """
        # Se non ci sono gruppi di esperimenti caricati nella struttura dati, interrompe l'esecuzione
        if not self.groups:
            return

        # Ferma qualsiasi animazione in corso prima di distruggere o ridisegnare la scena
        self._stop_animation()

        # Determina se mostrare i pulsanti di controllo dell'animazione (Play, Pausa, Step):
        # visibili SOLO se stiamo analizzando un singolo test (exp_idx < 4) E ci troviamo nella vista "Auto Animata" (view_idx == 4)
        show_anim_buttons = (self.current_exp_idx < 4 and self.current_view_idx == 4)
        self.btn_play.ax.set_visible(show_anim_buttons)
        self.btn_pause.ax.set_visible(show_anim_buttons)
        self.btn_step.ax.set_visible(show_anim_buttons)

        # Rimuove fisicamente tutti i sotto-grafici (Axes) esistenti dalla figura Matplotlib per evitare sovrapposizioni
        for ax in self.ax_grid:
            ax.remove()
        self.ax_grid.clear()  # Svuota la lista dei riferimenti ai grafici

        # Recupera i dati del gruppo di esperimenti attualmente selezionato
        group = self.groups[self.current_group_idx]
        # Costruisce la prima riga del titolo con le informazioni generali del percorso e dei parametri
        title_base = f"Path: '{group['path_name']}' | Variante: {group['variant'].upper()} | Look Ahead: {group['ld']} m"

        # Chiavi e nomi descrittivi per accedere ai 4 esperimenti del gruppo
        exp_keys = ['ideal_no_lc', 'ideal_lc', 'noisy_no_lc', 'noisy_lc']
        exp_labels = [
            "1. ODOMETRIA IDEALE | SENZA Loop Closure",
            "2. ODOMETRIA IDEALE | CON Loop Closure",
            "3. ODOMETRIA RUMOROSA | SENZA Loop Closure",
            "4. ODOMETRIA RUMOROSA | CON Loop Closure"
        ]

        # =========================================================================
        # MODALITÀ 1: ANALISI DI UN SINGOLO ESPERIMENTO (current_exp_idx da 0 a 3)
        # =========================================================================
        if self.current_exp_idx < 4:
            # Estrae i dati e il nome dell'esperimento corrente
            res = group[exp_keys[self.current_exp_idx]]
            exp_name = exp_labels[self.current_exp_idx]

            env = res['env']
            path = res['path']
            est = res['estimated_history']
            real = res['robot_history']

            # --- VISTA 0: Traiettoria Completa (Percorso, Reale e Stima sullo stesso grafico) ---
            if self.current_view_idx == 0:
                ax = self.fig.add_subplot(1, 1, 1)
                self.ax_grid.append(ax)
                plot_environment(ax, env)
                ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
                ax.plot(real[:, 0], real[:, 1], 'b-', linewidth=1.5, label='Traiettoria Reale')
                ax.plot(est[:, 0], est[:, 1], 'r.', markersize=4, label='Stima ICP + Odom')

                ax.set_title(f"{exp_name}\nVista: Traiettoria Completa", fontweight='bold', fontsize=14)
                ax.legend(loc='lower right')
                ax.grid(True)
                adjust_axis_limits(ax, path, env=env, real=real, est=est, min_range=5.0)
                view_name = "Completa"

            # --- VISTA 1: Traiettoria Scomposta (Due grafici affiancati con assi sincronizzati) ---
            elif self.current_view_idx == 1:
                ax1 = self.fig.add_subplot(1, 2, 1)
                ax2 = self.fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)  # sharex/y blocca gli zoom insieme
                self.ax_grid.extend([ax1, ax2])

                # Riquadro Sinistro: Solo Stima ICP vs Percorso
                plot_environment(ax1, env)
                ax1.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
                ax1.plot(est[:, 0], est[:, 1], 'r.', markersize=4, label='Stima ICP + Odom')
                ax1.set_title("Solo Path + Stima")
                ax1.legend(loc='lower right')
                ax1.grid(True)

                # Riquadro Destro: Solo Traiettoria Reale vs Percorso
                plot_environment(ax2, env)
                ax2.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')
                ax2.plot(real[:, 0], real[:, 1], 'b-', linewidth=1.5, label='Traiettoria Reale')
                ax2.set_title("Solo Path + Reale")
                ax2.legend(loc='lower right')
                ax2.grid(True)

                adjust_axis_limits(ax1, path, env=env, real=real, est=est, min_range=5.0)
                view_name = "Scomposta"

            # --- VISTA 2: Analisi Scostamenti / Errore Temporale (Scala Logaritmica) ---
            elif self.current_view_idx == 2:
                ax = self.fig.add_subplot(1, 1, 1)
                self.ax_grid.append(ax)

                # Calcola gli scostamenti euclidei rispetto al tracciato ideale
                dev_real, dev_est = calc_errors(real, est, path)
                # Sostituisce gli zeri con 1e-6 per evitare errori matematici log(0)
                dev_real_log = np.where(np.array(dev_real) == 0, 1e-6, dev_real)
                dev_est_log = np.where(np.array(dev_est) == 0, 1e-6, dev_est)

                if len(dev_real_log) > 0:
                    ax.plot(dev_real_log, 'b-', label='Errore Traiettoria Reale', alpha=0.8, linewidth=1.5)
                if len(dev_est_log) > 0:
                    ax.plot(dev_est_log, 'r--', label='Errore Stima ICP+Odom', alpha=0.8, linewidth=1.5)

                ax.set_yscale('log')  # Imposta l'asse Y in scala logaritmica
                ax.set_xlabel("Step di Simulazione", fontsize=12)
                ax.set_ylabel("Distanza dal Path (metri)", fontsize=12)
                ax.set_title(f"{exp_name}\nVista: Analisi Scostamenti (Scala Logaritmica)", fontweight='bold',
                             fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, which="both", ls="--", alpha=0.6)
                view_name = "Errore (Log)"

            # --- VISTA 3: Analisi Scostamenti / Errore Temporale (Scala Lineare) ---
            elif self.current_view_idx == 3:
                ax = self.fig.add_subplot(1, 1, 1)
                self.ax_grid.append(ax)

                dev_real, dev_est = calc_errors(real, est, path)

                if len(dev_real) > 0:
                    ax.plot(dev_real, 'b-', label='Errore Traiettoria Reale', alpha=0.8, linewidth=1.5)
                if len(dev_est) > 0:
                    ax.plot(dev_est, 'r--', label='Errore Stima ICP+Odom', alpha=0.8, linewidth=1.5)

                ax.set_xlabel("Step di Simulazione", fontsize=12)
                ax.set_ylabel("Distanza dal Path (metri)", fontsize=12)
                ax.set_title(f"{exp_name}\nVista: Analisi Scostamenti (Scala Lineare)", fontweight='bold', fontsize=14)
                ax.legend(fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)
                view_name = "Errore (Lineare)"

            # --- VISTA 4: Auto Animata (Riproduzione interattiva del robot) ---
            elif self.current_view_idx == 4:
                ax = self.fig.add_subplot(1, 1, 1)
                self.ax_grid.append(ax)
                # Delega l'inizializzazione dell'animazione al metodo interno dedicato
                self._setup_animation(ax, env, path, est, real, exp_name)
                view_name = "Auto Animata"

            # Compone la stringa di stato visibile nel sottotitolo
            status_text = f"Gruppo [{self.current_group_idx + 1}/{len(self.groups)}] | Esp [{self.current_exp_idx + 1}/5] ({exp_name}) | Vista [{self.current_view_idx + 1}/5] ({view_name})"

        # =========================================================================
        # MODALITÀ 2: SCHERMATA DI RIEPILOGO COMPARA TIVI (current_exp_idx == 4)
        # =========================================================================
        else:
            ax = self.fig.add_subplot(1, 1, 1)
            self.ax_grid.append(ax)

            means_real, means_est = [], []
            sums_real, sums_est = [], []

            # Calcola le metriche di errore (media e somma) per tutti e 4 i test del gruppo
            for k in exp_keys:
                r, e = calc_errors(group[k]['robot_history'], group[k]['estimated_history'], group[k]['path'])
                means_real.append(np.mean(r) if len(r) > 0 else 0)
                means_est.append(np.mean(e) if len(e) > 0 else 0)
                sums_real.append(np.sum(r) if len(r) > 0 else 0)
                sums_est.append(np.sum(e) if len(e) > 0 else 0)

            x = np.arange(4)  # Posizioni delle 4 barre
            width = 0.35  # Larghezza di ciascuna barra
            labels_bar = [
                "Odometria Ideale\nNo LC",
                "Odometria Ideale\nCon LC",
                "Odometria Rumorosa\nNo LC",
                "Odometria Rumorosa\nCon LC"
            ]

            # --- RIEPILOGO - VISTA 0: Grafico a Barre delle MEDIE ---
            if self.current_view_idx == 0:
                rects1 = ax.bar(x - width / 2, means_real, width, label='Media Errore Reale', color='blue', alpha=0.7)
                rects2 = ax.bar(x + width / 2, means_est, width, label='Media Errore Stima ICP+Odom', color='red',
                                alpha=0.7)
                ax.set_title("5. RIEPILOGO ESPERIMENTI: Media degli Errori", fontweight='bold', fontsize=16)
                ax.set_ylabel("Errore Medio (metri)", fontsize=12)
                view_name = "Grafico a Barre (Media)"
            # --- RIEPILOGO - VISTA 1: Grafico a Barre delle SOMME TOTALI ---
            else:
                rects1 = ax.bar(x - width / 2, sums_real, width, label='Somma Errore Reale', color='blue', alpha=0.7)
                rects2 = ax.bar(x + width / 2, sums_est, width, label='Somma Errore Stima ICP+Odom', color='red',
                                alpha=0.7)
                ax.set_title("5. RIEPILOGO ESPERIMENTI: Somma Totale degli Errori", fontweight='bold', fontsize=16)
                ax.set_ylabel("Errore Totale Cumulato (metri)", fontsize=12)
                view_name = "Grafico a Barre (Somma)"

            ax.set_xticks(x)
            ax.set_xticklabels(labels_bar, fontsize=11)
            ax.legend(fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Aggiunge le etichette numeriche sopra ogni barra del grafico
            for rect in rects1 + rects2:
                height = rect.get_height()
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

            status_text = f"Gruppo [{self.current_group_idx + 1}/{len(self.groups)}] | Esp [5/5] (Riepilogo) | Vista [{self.current_view_idx + 1}/2] ({view_name})"

        # Imposta il titolo generale in cima alla finestra (unendo base e status)
        self.fig.suptitle(f"{title_base}\n{status_text}", fontsize=13)
        # Aggiusta i margini della figura per lasciare spazio sufficiente ai pulsanti in basso e ai titoli in alto
        plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.15)
        # Richiede il ridisegno non bloccante della finestra grafica
        self.fig.canvas.draw_idle()