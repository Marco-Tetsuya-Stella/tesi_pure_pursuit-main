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
from pathlib import Path  # classe Path per manipolare agevolmente i percorsi dei file e delle cartelle
import matplotlib.pyplot as plt  # modulo pyplot di matplotlib per creare grafici e finestre interattive
from matplotlib.animation import FuncAnimation, FFMpegWriter  # Importa le classi per creare animazioni e salvarle in MP4

# Importazioni dei moduli del progetto
# funzione core che esegue effettivamente la logica della simulazione
from pure_pursuit_simulation import run_simulation
# funzione che disegna geometricamente il robot sul grafico
from visualizer import draw_robot
# utility per visualizzare gli ostacoli, gestire gli assi e creare nomi file univoci
from visualizer_pure_pursuit import plot_environment, adjust_axis_limits, get_unique_filepath


def main():
    # ==========================================
    # 1. PARAMETRI DI CONFIGURAZIONE
    # ==========================================
    nome_tracciato = "straight_short"  # Imposta il nome della mappa/percorso da utilizzare per la simulazione
    variante = "type2"  # Specifica tipologia del percorso
    lookahead = 0.4  # Imposta la distanza di look-ahead in metri

    # Flag di controllo
    usa_odometria_rumorosa = True  # True per rumore, False per ideale
    usa_loop_closure = False  # True uso loop closure, False non uso loop closure

    # ==========================================
    # 2. GENERAZIONE DINAMICA DEI NOMI E TITOLI
    # ==========================================
    str_odom = "noisy" if usa_odometria_rumorosa else "ideal"  # Crea stringa per il nome del file in base alla scelta sull'odometria
    str_lc = "withlc" if usa_loop_closure else "nolc"  # Crea stringa per il nome del file in base alla scelta sul loop closure

    #stringhe che vengono stampate per capire con quali parametri stò lavorando
    titolo_odom = "Odometria Rumorosa" if usa_odometria_rumorosa else "Odometria Ideale"
    titolo_lc = "Con Loop Closure" if usa_loop_closure else "Senza Loop Closure"

    print(f"{'=' * 75}")  # print di = per 75 volte
    print(f"AVVIO SIMULAZIONE SU: '{nome_tracciato}' (Look-Ahead: {lookahead}m)")  # nome_tracciato è la striga definita all inizio
    print(f"Modalità: {titolo_odom} | {titolo_lc}")
    print(f"{'=' * 75}\n")  # \n vado a capo

    # ==========================================
    # 3. ESECUZIONE SIMULAZIONE
    # ==========================================
    risultato = run_simulation(  # Richiama la funzione di simulazione vera e propria e ne salva l'output
        path_name=nome_tracciato,  # nome del percorso
        variant=variante,  # variante percorso
        use_loop_closure=usa_loop_closure,  # utilizzo o non utilizzo loopclosure
        add_odom_noise=usa_odometria_rumorosa,  # utilizzo o non utilizzo odometria rumorosa
        lookahead_distance=lookahead,  # distanza di look ahead
        verbose=True  # Abilita le stampe di debug e avanzamento da parte del simulatore
    )

    env = risultato['env']  # Estrae dal dizionario restituito l'oggetto che modella l'ambiente
    path = risultato['path']  # Estrae l'elenco dei punti formanti la traiettoria di riferimento ideale (path)
    real_history = risultato['robot_history']  # Estrae le reali pose assunte dal robot ad ogni passo temporale

    # Controlla se la lista delle posizioni risulta vuota (indice di errore critico o simulazione fallita)
    if len(real_history) == 0:
        print("[ERRORE] La traiettoria generata è vuota. Impossibile continuare.")
        return  # Interrompe l'esecuzione uscendo dalla funzione main anticipatamente

    # ==========================================
    # 4. CARTELLA E NOME FILE VIDEO AUTOMATICO
    # ==========================================
    src_dir = Path(__file__).resolve().parent  # Risale in modo robusto alla cartella esatta contenente lo script attualmente in esecuzione
    base_dir = src_dir.parent / "video_pure_pursuit"  # Calcola il percorso per una directory destinata ai video, posizionandola un livello sopra ai sorgenti
    base_dir.mkdir(parents=True, exist_ok=True)  # Genera la cartella sul disco (e le sue parenti se necessario), evitando errori se essa è già esistente

    # Il nome file ora include dinamicamente 'noisy'/'ideal' e 'withlc'/'nolc'
    nome_base_video = f"{nome_tracciato}_{variante}_ld{lookahead}_{str_odom}_{str_lc}.mp4"  # Costruisce la stringa rappresentante il nome del file MP4 con i dettagli dell'esperimento
    percorso_video_originale = base_dir / nome_base_video  # Concatena il percorso della cartella e il nome del file calcolato
    percorso_video = get_unique_filepath(percorso_video_originale)  # Passa il percorso a una funzione che aggiunge indici (_1, _2) nel caso in cui il nome esista già

    # ==========================================
    # 5. IMPOSTAZIONE GRAFICA E ANIMAZIONE
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))  # Crea una figura Matplotlib e il suo asse principale con larghezza 10 e altezza 8 pollici
    fig.canvas.manager.set_window_title(f'Pure Pursuit - {nome_tracciato}')  # Modifica il titolo della finestra di sistema che conterrà il grafico

    plot_environment(ax, env)  # Disegna sull'asse tutti gli ostacoli e geometrie dell'oggetto ambiente
    ax.plot(path[:, 0], path[:, 1], 'g--', label='Percorso Riferimento')  # Disegna la curva del tracciato di riferimento con una linea verde tratteggiata, con etichetta
    trail_line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Traiettoria Reale')  # Crea l'oggetto vuoto per il rendering dinamico della scia del robot (blu e semi-trasparente)

    # Titolo del grafico dinamico
    ax.set_title(f"Simulazione Pure Pursuit: '{nome_tracciato}'\n{titolo_odom} | {titolo_lc}", fontweight='bold')  # Inserisce il titolo all'interno del grafico Matplotlib con le modalità
    ax.legend(loc='lower right')  # Chiede a Matplotlib di posizionare un box legenda nell'angolo inferiore destro
    ax.grid(True)  # comparsa della griglia cartesiana sullo sfondo dell'asse
    adjust_axis_limits(ax, path, env=env, real=real_history, min_range=5.0)  # Sistema e zooma gli assi X e Y per visualizzare in modo proporzionato ambiente e traiettorie

    robot_artists = []  # Prepara una lista vuota per memorizzare i poligoni e geometrie che compongono il corpo del robot al fine di ripulirli

    #
    # funzione di inizializzazione dell'animazione. Viene chiamata una volta sola all'inizio (o ad ogni reset della simulazione).
    #
    def init():  # Funzione inizializzatrice richiamata dalla logica interna di FuncAnimation
        trail_line.set_data([], [])  # Garantisce che al tempo iniziale la linea della traiettoria sia un array vuoto
        return trail_line,  # Ritorna l'oggetto visivo resettato

    #
    # parte di animazione eseguita ad ogni fotogramma
    #
    def update(frame):  # Definisce il nucleo dell'animazione: viene eseguita una volta per ogni passo temporale ('frame')

        #
        # Informa Python di usare la variabile robot_artists della funzione genitrice (scope esterno), non creandone una locale
        #
        # poichè robot_artists viene completamente sovrascritta ad ogni frame con non local specifico di non creare una nuova variabile locale ma di modificare quella che è già stata definita nel main
        nonlocal robot_artists

        #
        # nel caso che venga lanciata l'eccezione da artist.remove() significa che l'elemento visivo è già stato rimosso o non è presente quindi l'obbiettivo di non avere l'oggetto è già stato raggiunto
        # di conseguenza possimo ignorare l'eccezione attraverso pass
        #
        for artist in robot_artists:  # Cicla su ciascun pezzo grafico disegnato appartenente al corpo del robot nel fotogramma appena trascorso
            try:  # Tenta il blocco in via sicura
                artist.remove()  # Elimina fisicamente il disegno della posa vecchia dal grafico di Matplotlib
            except Exception:  # Intercetta qualsivoglia errore durante la cancellazione
                pass  # Prosegue il ciclo ignorando gli errori di rimozione
        robot_artists.clear()  # Svuota definitivamente la lista per preparare l'inserimento della posa aggiornata

        trail_line.set_data(real_history[:frame + 1, 0], real_history[:frame + 1, 1])  # Ricostruisce le stringhe X e Y della scia fino all'istante temporale 'frame' corrente

        xlim = ax.get_xlim()  # Ricava il limite visivo in uso sull'asse delle ascisse

        # Adatta le proporzioni di disegno del robot rispetto allo zoom del grafico affinché rimanga della misura consona
        scale = max(0.08, (xlim[1] - xlim[0]) * 0.02)

        state = real_history[frame]  # Pesca le coordinate di stato (X, Y, orientamento) corrispondenti al frame presente
        robot_artists = draw_robot(ax, state, robot_radius=scale, color='tab:blue')  # Chiama l'helper per renderizzare il modello geometrico del robot e immagazzina i riferimenti agli "artist"

        return [trail_line] + robot_artists  # Ritorna l'intera batteria di oggetti grafici che Matplotlib necessita di aggiornare su schermo per questo frame

    num_frames = len(real_history)  # Conta quanti fotogrammi totali occorrono calcolando la lunghezza della cronologia delle coordinate

    #
    # oggetto di Matplotlib che orchestra e fa partire l'animazione
    #
    ani = FuncAnimation(fig, # finestra del grafico
                        update, # funzione da richiamare ad ogni ciclo
                        frames=num_frames, # numero totale di passaggi (pari al numero di passi salvati in real_history)
                        init_func=init, # funzione di azzeramento iniziale
                        blit=False, # indica a Matplotlib di ridisegnare l'intera figura ad ogni frame (utile se si ridimensionano gli assi o gli sfondi)
                        interval=40, # intervallo di tempo tra un frame e il successivo in millisecondi (40 ms corrisponde a circa 25 frame al secondo
                        repeat=False) # evita che l'animazione ricominci da capo una volta terminata

    # ==========================================
    # 6. SALVATAGGIO VIDEO ED ESECUZIONE
    # ==========================================
    print(f"Generazione e salvataggio del video MP4 in corso...")
    print(f"Destinazione: {percorso_video.name}")  # Esplicita il solo nome del file finale in elaborazione
    #
    # esecuzione bloccante = programma si ferma e "congela" l'avanzamento dello script su quella riga di codice finché l'operazione non è stata completata al 100%
    #
    # ani richiama init e save
    try:  # Prova l'esecuzione bloccante di salvataggio file
        writer = FFMpegWriter(fps=25, metadata=dict(artist='Pure Pursuit Simulation'), bitrate=1800)  # Crea lo scrittore per il formato mp4, configurando i frame al secondo e la qualità (bitrate)
        ani.save(str(percorso_video), writer=writer)  # Lancia l'effettiva esportazione fotogramma per fotogramma verso il file indicato in 'percorso_video'
        print(f"[SUCCESS] Video salvato con successo in:\n -> {percorso_video}\n")
    except Exception as e:  # Cattura eccezioni
        print(f"[ERRORE] Salvataggio MP4 fallito: {e}")

    print("Apertura finestra grafica a schermo...")
    plt.show()  # Richiama il processo della libreria grafica per eseguire l'esposizione della simulazione su interfaccia monitor bloccante


if __name__ == "__main__":
    main()