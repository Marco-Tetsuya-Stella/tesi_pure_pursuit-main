# Pure Pursuit Robot System

Il progetto prosegue ed estende un lavoro di tesi incentrato sull'implementazione, la simulazione e l'analisi sperimentale dell'algoritmo di inseguimento di traiettoria **Pure Pursuit** per veicoli robotici autonomi.

---

## 🛠️ Struttura del Progetto e Moduli

### 🔹 Moduli Principali (Algoritmo e Simulazione)
* **`pure_pursuit.py`**: Implementa la logica dell'algoritmo di guida Pure Pursuit per il calcolo dei comandi di guida.
* **`pure_pursuit_simulation.py`**: Gestisce il ciclo di simulazione. Permette di testare **12 diverse tipologie di tracciati** (aperti e chiusi) utilizzando la stima di posizione tramite **ICP con odometria** (sia *ideale* sia *rumorosa*) ed abilitando/disabilitando la *loop closure*. Gestisce inoltre esperimenti parametrici variando la distanza di look-ahead ($L_d$).
* **`noisy_odometry.py`**: Simula un modello di odometria con rumore realistico e intermittente per valutare la robustezza del sistema.

### 🔹 Moduli di Generazione Ambiente e Percorsi
* **`path_generator.py`**: Fornisce i metodi per la generazione discreta di percorsi custom.
* **`prefabricated_paths.py`**: Mette a disposizione una serie di percorsi predefiniti pronti all'uso e funzioni per la loro visualizzazione.
* **`environment_presets_pure_pursuit.py`**: Consente la generazione deterministica di ambienti con ostacoli che non interferiscono con il percorso. Supporta 3 livelli di configurazione per tracciato:
  * `type1`: Nessun ostacolo.
  * `type2`: Densità media di ostacoli.
  * `type3`: Massimo numero di ostacoli.

### 🔹 Visualizzazione ed Esecuzione
* **`visualizer_pure_pursuit.py`**: Modulo dedicato all'analisi grafica dei risultati:
  * Confronto tra la traiettoria di riferimento e la traiettoria effettivamente percorsa dal veicolo.
  * Analisi quantitativa degli errori di inseguimento e dei parametri di controllo.
  * Salvataggio automatico dei grafici generati.
* **`main_pure_pursuit.py`**: Script principale che orchestra ed esegue l'intero flusso di simulazione ed esperimenti parametrici.
* **`main2.py`**: Script per l'esecuzione di una singola simulazione su un tracciato selezionato; mostra l'animazione in tempo reale a schermo e salva il video in formato `.mp4` nella cartella `video_pure_pursuit/`.

---

## 💻 Requisiti di Sistema e Dipendenze

> ⚠️ **IMPORTANTE:** È **necessario utilizzare Python 3.11**. Versioni successive causano problemi di compatibilità con alcune librerie.

### Dipendenze (`requirements.txt`)

```text
numpy==2.2.5
matplotlib==3.10.3
shapely==2.1.2
tqdm==4.67.1
scipy==1.16.3
open3d==0.19.0
```

Per installare le dipendenze:
```bash
pip install -r requirements.txt
```

---

## 🚀 Guida all'Esecuzione

1. **Esecuzione della suite di test completa (Esperimenti e Grafici)**:
   ```bash
   python main_pure_pursuit.py
   ```

2. **Visualizzazione di una singola simulazione e salvataggio video**:
   ```bash
   python main2.py
   ```

3. **Visualizzazione preventiva dei tracciati e degli ambienti con ostacoli**:
   ```bash
   python environment_presets_pure_pursuit.py
   ```

