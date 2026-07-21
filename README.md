# README - Pure Pursuit Robot System

> **Aggiornato al 22/07/2026:** Gli aggiornamenti sono scritti in fondo al testo.
>
> **Nota:** File README non definitivo.

## Descrizione del Progetto

Il programma prosegue un lavoro di tesi già iniziato da altri studenti, implementando un sistema di **Pure Pursuit** per un robot.

---

## Struttura dei Nuovi File Implementati

- **`path_generator`**: fornisce i metodi per la generazione di percorsi discreti.
- **`prefabricated_paths`**: mette a disposizione una serie di percorsi predefiniti, pronti all'uso, e ne consente la visualizzazione.
- **`environment_presets_pure_pursuit`**: permette la generazione deterministica di ambienti con ostacoli, garantendo che questi non interferiscano con il percorso.
- **`pure_pursuit`**: implementa l'algoritmo di Pure Pursuit.
- **`pure_pursuit_simulation`**: esegue le simulazioni e ne gestisce la visualizzazione. Attualmente consente di testare **nove diverse tipologie di tracciati** (sia aperti sia chiusi), utilizzando **ICP con odometria**, con la possibilità di abilitare o meno la *loop closure*.

> **Nota aggiuntiva:** È inoltre presente una classe denominata `main2`, realizzata inizialmente per implementare una visualizzazione animata dei tracciati, ma al momento ancora incompleta.

---

## Sviluppi Futuri (Next Steps)

Nei prossimi giorni intendo completare i seguenti aspetti:

1. **Migliorare la visualizzazione grafica** dei risultati, aggiungendo ulteriori grafici.
2. **Ampliare il numero di test** eseguiti.
3. **Implementare un sistema per il salvataggio automatico** dei grafici prodotti.
4. **Valutare l'aggiunta di un'animazione** del movimento del robot durante le simulazioni.

---

## Esecuzione dei Programmi

- Il programma principale è eseguibile attraverso:
  ```bash
  python pure_pursuit_simulation.py
  ```
- Attraverso il seguente script è invece possibile visualizzare i tracciati e gli ambienti prefabbricati:
  ```bash
  python environment_presets_pure_pursuit.py
  ```

---

## Requisiti di Sistema e Dipendenze

Per eseguire i programmi è **necessaria la versione Python 3.11** (versioni successive causano problemi di compatibilità con alcune librerie Python presenti nei requirments).

### Dipendenze (`requirements.txt`)

```text
numpy==2.2.5
matplotlib==3.10.3
shapely==2.1.2
tqdm==4.67.1
scipy==1.16.3
open3d==0.19.0
```

---

## Aggiornamento 22/07/2026

- **`environment_presets_pure_pursuit.py`**: aggiunta di tre tipi di configurazione per tracciato (`type1` = nessun ostacolo, `type3` = massimo numero di ostacoli).
- **`path_generator.py`**: aggiunta di una nuova pista (la quale ha un punto di intersezione così da poter sfruttare la loop closure).
- **`pure_pursuit.py`**:
  - Aggiunta di una finestra su cui andare a vedere i punti del percorso.
  - Correzione del bug di terminazione prematura.
- **`pure_pursuit_simulation.py`**:
  - Permette ora di eseguire 4 tipi di test per tipologia di tracciato, scegliendo se attivare o meno il rumore all'odometria e se usare o meno la loop closure.
- **`noisy_odometry.py`**: simula un'odometria con rumore realistico e intermittente.
