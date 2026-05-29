# Vesper

Vesper è un agente AI specializzato nella ricerca e recupero di informazioni (RAG). È progettato dal principio come agente multi-specialista per inferenza offline a partire da 6GB di VRAM. Si crea il suo spazio nel tuo vault Obsidian e puoi espandere la sua knowledge base con tutte le note che vuoi.

Nessun cloud, niente costi API. Accessibile via Telegram e browser.

## Features

### Conoscenza personale

- 🧠 **Knowledge base nativa con Obsidian**: le tue note Obsidian diventano il cervello dell'agente. Ricerca semantica nella wiki, graph traversal sui `[[link]]` interni, re-ranker neurale. Più curi la tua LLM wiki e più ti sarà utile.
- 📂 **Watchdog**: ogni nota salvata nella wiki viene re-indicizzata automaticamente entro pochi secondi, senza intervento manuale. La knowledge base è sempre aggiornata senza che tu debba pensarci.
- 📝 **Sintesi automatiche**: ogni risposta basata su contenuti della tua wiki viene salvata automaticamente in `synthesis/` con tag e link alle fonti arricchendo la knowledge base nel tempo. Per le ricerche web sarai tu a deciderlo. Le sintesi salvate vengono indicizzate e recuperate con priorità nelle ricerche future.
- 🕸️ **Knowledge graph**: entità e relazioni vengono estratte automaticamente dai contenuti della wiki e persistitono in un grafo locale, abilitandoti richieste più complesse come "dammi tutto quello che ho scritto su X in relazione a Y" pure quando gli argomenti non compaiono nello stesso documento.

### Memoria

- 💾 **Memoria indipendente dal modello**: la cronologia delle tue conversazioni è agnostica al LLM che usi.
- 🗜️ **Compressione automatica della cronologia**: sopra una certa soglia di messaggi la cronologia viene compressa automaticamente senza perdere il filo della conversazione. Nessun limite artificiale a quanto a lungo puoi conversare.
- 👤 **Profilo utente**: l'agente impara le tue preferenze dalla conversazione e le salva dentro il tuo vault Obsidian nel documento `USER.md` che potrai facilmente consultare, correggere e osservarne i risultati già con l'inferenza successiva.

### Privacy e controllo

- 🔒 **100% locale, davvero**: nessun dato esce dal PC, nessun costo API, nessun account richiesto, nessun cloud. Non parzialmente, ma interamente locale: LLM, embedding e re-ranker girano tutti sul tuo hardware. Le tue conversazioni non lasciano mai il tuo PC.
- 🤏 **Costruito per i modelli locali**: progettato dal principio per modelli da 9–27B parametri e costruito attorno ai loro limiti reali. Ogni chiamata LLM ha un task preciso con uno schema di output fisso e facile da seguire, ottenendo risultati sorprendenti anche con modelli quantizzati a 3 bit.
- ⚡ **Specialisti in parallelo**: researcher, analyzer, reasoner e generator lavorano in parallelo quando possibile, ognuno con il contesto strettamente necessario al suo ruolo. Le risposte sono più rapide e coerenti di un singolo prompt generico, anche su modelli compatti.

### Produttività

- 🔍 **Ricerca web parallela**: quando la risposta richiede sia la wiki che il web, vault e web vengono interrogati in contemporanea. Il recupero delle pagine web dalle fonti trovate avviene in parallelo per design.
- ⏰ **Task proattivi**: imposta task one-off o ricorrenti che Vesper eseguirà in autonomia — briefing mattutini, promemoria, monitoraggi periodici. Li scrive come file YAML nel vault, dove puoi modificarli o cancellarli direttamente.
- 📎 **Analisi documenti**: carica PDF e DOCX direttamente in chat — l'agente li legge e ragiona sul contenuto. I file caricati vengono indicizzati automaticamente in `vault/raw/` e restano consultabili anche nelle sessioni future tramite ricerca semantica, non solo durante il turno di upload.

### Accessibilità e hardware

- 🖥️ **Multi-GPU automatico**: tensor split e assegnazione GPU senza configurazione, a partire da soli 6 GB di VRAM. Se hai più GPU le usa tutte proporzionalmente, senza impostare nulla.
- 💬 **Telegram + Web**: Telegram per admin con accesso completo alle funzionalità agentiche, Web per condividere l'agente (con accesso limitato configurabile).

### Architettura

- 🧩 **Pipeline a tre livelli**: Orchestrator, Manager e specialisti hanno ruoli rigidamente separati. L'Orchestrator pianifica, il Manager coordina senza mai chiamare l'LLM, gli specialisti eseguono il loro singolo task. Nessun componente fa più di quello che gli compete.
- 🔇 **Specialisti stateless**: ogni specialista riceve solo il contesto strettamente necessario al suo ruolo — niente cronologia, niente dati degli altri task. Meno rumore nel prompt significa risposte più precise, specialmente su modelli locali piccoli.
- 🔌 **Interfacce intercambiabili**: la pipeline core è implementata una volta sola e condivisa da tutte le interfacce. Aggiungere un nuovo canale (es. API REST, Discord) richiede solo di implementare tre metodi: come inviare un messaggio, come inviare un errore, quali azioni sono permesse.

<br>

> **Work in progress**:
> - Interfaccia web
> - Scheduler cron per task one-shot/ricorrenti

## Indice

- [Requisiti](#requisiti)
- [Installazione](#installazione)
- [Configurazione `.env`](#configurazione-env)
- [Setup consigliato per fasce di VRAM](#setup-consigliato-per-fasce-di-vram)
- [Come funziona](#come-funziona)
- [Task proattivi](#task-proattivi)
- [Interfacce](#interfacce)
- [Struttura del vault](#struttura-del-vault)
- [Licenza](#licenza)
- [Riconoscimenti](#riconoscimenti)


## Requisiti

- **Python 3.10+**
- GPU NVIDIA con driver recenti.
- Almeno 6 GB di VRAM. Vedi [Setup consigliato per fasce di VRAM](#setup-consigliato-per-fasce-di-vram).
- Min. ~20 GB, max. ~52 GB di spazio libero su disco per i pesi dei modelli (secondo i setup consigliati) e il virtualenv.

> **Windows**: consigliato eseguire Vesper in WSL (Windows Subsystem for Linux). I binari CUDA di llama.cpp e le dipendenze Python sono distribuiti e testati principalmente per Linux — su Windows nativo alcune dipendenze richiedono configurazione manuale aggiuntiva.


## Installazione

### 1. Clona il repository

```bash
git clone <repo-url>
cd vesper/agent
```

### 2. Esegui l'installer

```bash
python install.py
```

L'installer procederà con:
- creare il virtualenv con la versione Python più recente disponibile sul tuo PC (3.10+)
- installare le dipendenze
- generare il file `.env`
- scaricare il modello LLM, embedding e reranking
- scaricare il binario nativo `llama-server.exe` o il binario Ubuntu CUDA su Linux/WSL

### 3. Configura `.env`

Apri il file `.env` e compila almeno queste tre variabili:

| Variabile | Descrizione |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token del bot Telegram (da [@BotFather](https://t.me/BotFather)) |
| `TELEGRAM_ADMIN_CHAT_ID` | Il tuo Chat ID Telegram |
| `VAULT_PATH` | Path assoluto al tuo vault Obsidian (es. `/home/user/vault`) |

Per ulteriori variabili, vedi [configurazione `.env`](#configurazione-env)

### 4. Avvia Vesper

Con `LLM_MANAGED=true` (default), llama-server parte automaticamente insieme a `main.py`.

Se hai già un server LLM attivo o vuoi usare un'istanza remota, imposta `LLM_MANAGED=false` nel `.env` e configura `LLM_BASE_URL` con l'endpoint corretto. In questo caso `install.py` non scaricherà i modelli LLM.

Come scelta di design, Vesper usa tutte le GPU disponibili. Se hai più GPU, i layer del modello principale di inferenza vengono distribuiti a quote proporzionalmente maggiori sulle GPU con il bus PCIe più veloce. I modelli di embedding e re-ranking vengono invece assegnati alla GPU con il bus PCIe più lento.

```bash
cd vesper/agent
python main.py
```

---

## Configurazione `.env`

Le variabili obbligatorie sono nel [passo 3](#3-configura-env). Queste sono quelle che potresti voler toccare:

| Variabile | Default | Descrizione |
|---|---|---|
| `LLM_MANAGED` | `true` | `false`: gestisci llama-server manualmente. `true`: avvia e ferma llama-server automaticamente insieme a `main.py`. Con `false`, `install.py` non scarica i modelli LLM. |
| `HF_TOKEN` | _(blank)_ | Token HuggingFace. Velocizza i download e sblocca modelli privati. |
| `LLM_MULTI_GPU` | `true` | Distribuisce il modello su tutte le GPU disponibili. `false` = GPU singola |
| `LLM_N_GPU_LAYERS` | `-1` | Layer da caricare in GPU. `-1` = automatico; `0` = solo CPU |
| `LLM_N_CTX` | _(auto)_ | Lunghezza del contesto in token. Blank (default) = rilevato automaticamente |
| `LLM_MODEL_FILE` | `Qwen3.6-27B-Q6_K.gguf` | Nome del file GGUF del modello LLM |
| `RERANKER_ENABLED` | `true` | `false` = disabilita il re-ranker (ricerca meno precisa ma più veloce, e libera ~2.3 GB VRAM). Il re-ranker gira in VRAM su entrambe le piattaforme (subprocess `llama-server` su `RERANKER_PORT`). |
| `VAULT_WATCH_DEBOUNCE_SECONDS` | `5` | Secondi di attesa dopo l'ultimo salvataggio prima di re-indicizzare un file |

> Cambiando `EMBEDDING_MODEL`, l'indice RAG esistente verrà ricreato.


## Setup consigliato per fasce di VRAM

I valori di VRAM sono approssimativi. Considera solo la VRAM fisica (non shared) per evitare inferenze lente.

Per cambiare il LLM aggiorna `LLM_MODEL_REPO` e `LLM_MODEL_FILE` nel file `.env` — il percorso locale viene derivato automaticamente come `data/models/{LLM_MODEL_REPO}/{LLM_MODEL_FILE}`. Il re-ranker si configura con `RERANKER_MODEL_REPO` e `RERANKER_MODEL_FILE` — stessa convenzione: `data/models/{RERANKER_MODEL_REPO}/{RERANKER_MODEL_FILE}`.

> Prima di aggiornare il file `.env`, verifica su HuggingFace che il repo e il file esistano. Le naming convention variano tra provider GGUF e possono cambiare nel tempo.
> <br>Ultimo controllo: 17.05.2026

#### 6 GB

Con 6 GB la somma dei tre modelli supera la VRAM disponibile. Disabilita il re-ranker con `RERANKER_ENABLED=false` e aspettati che l'embedding possa girare su CPU (indicizzazione più lenta).

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | UD-Q3_K_XL | 5.05 | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-UD-Q3_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 8 GB

Con 8 GB la somma dei tre modelli supera la VRAM disponibile. Disabilita il re-ranker con `RERANKER_ENABLED=false`.

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | UD-Q3_K_XL | 5.05 | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-UD-Q3_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 12 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | UD-Q4_K_XL | 5.97 | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 16 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | UD-Q6_K_XL | 8.76 | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-UD-Q6_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 24 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | UD-Q4_K_XL | 17.6 | `unsloth/Qwen3.6-27B-GGUF` | `Qwen3.6-27B-UD-Q4_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 32 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | UD-Q5_K_XL | 20 | `unsloth/Qwen3.6-27B-GGUF` | `Qwen3.6-27B-UD-Q5_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

#### 40 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | UD-Q6_K_XL | 25.6 | `unsloth/Qwen3.6-27B-GGUF` | `Qwen3.6-27B-UD-Q6_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |


#### >48 GB

| Ruolo | Modello | Quant. | GB | Repo HuggingFace | File |
|---|---|---|---|---|---|
| LLM | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | UD-Q8_K_XL | 35.3 | `unsloth/Qwen3.6-27B-GGUF` | `Qwen3.6-27B-UD-Q8_K_XL.gguf` |
| Embedding | [bge-m3](https://huggingface.co/BAAI/bge-m3) | — | 2.11 | `BAAI/bge-m3` | — |
| Re-ranker | [Qwen3-Reranker-4B](https://huggingface.co/Mungert/Qwen3-Reranker-4B-GGUF) | Q4_K_M | 2.3 | `Mungert/Qwen3-Reranker-4B-GGUF` | `Qwen3-Reranker-4B-q4_k_m.gguf` |

---

## Come funziona

Quando mandi un messaggio, Vesper:

1. **Cerca nel vault** — prima ancora di pianificare, verifica se ci sono contenuti rilevanti nel vault (ricerca vettoriale leggera, nessun LLM). Passa il segnale all'Orchestrator.
2. **Pianifica** — l'Orchestrator legge la richiesta con la cronologia della conversazione e decide quali azioni servono producendo un piano JSON.
3. **Esegue** — il Manager crea il piano ordinando le azioni e lanciandole in parallelo. Raccoglie i risultati e prepara l'output finale. Zero inferenza.
4. **Risponde** — ogni specialista riceve il suo task con il contesto strettamente necessario fornito dal Manager e produce l'output (1 chiamata LLM ciascuno).
5. **Salva la sintesi** — se la risposta si è basata su contenuti del vault, salva automaticamente una nota in `vault/synthesis/` con tag e link alle fonti consultate. Per le ricerche web compare il pulsante "salva" che fa la stessa cosa su richiesta.
6. **Aggiorna il tuo profilo** — dopo ogni risposta, controlla se hai espresso nuove preferenze e aggiorna `USER.md`.

### Architettura

La pipeline a tre livelli tiene l'LLM lontano da tutto ciò che il codice può fare meglio:

- **Orchestrator** — l'unico componente che conosce la cronologia della conversazione. Pianifica le azioni in JSON (max 1024 token di output). Non tocca strumenti, non legge documenti.
- **Manager** — zero LLM. Risolve i gruppi di dipendenza, lancia le azioni indipendenti in parallelo, valida e corregge il JSON di output. Non può allucinare perché non fa inferenza.
- **Specialisti** — stateless. Ogni chiamata riceve esattamente il contesto che serve al suo ruolo, niente di più. Schema di output obbligatorio. Generano fino al completamento naturale senza cap artificiali di token.

**Ottimizzato per modelli piccoli**: 
Il modello LLM risponde esclusivamente a prompt concisi. Ogni chiamata LLM ha un task preciso con schema fisso. Il risultato è un'affidabilità strutturale che un approccio generalista non riesce a raggiungere su modelli piccoli (attorno ai 9B parametri).

Per i flussi di lavoro consulta il documento [ARCHITECTURE.md](ARCHITECTURE.md) 

### Specialisti interni

Gli specialisti (o "subagents") non sono nient'altro che prompt di sistema eseguiti sullo stesso modello, ciascuno senza accesso alla cronologia conversazione:
ogni chiamata riceve solo il contesto strettamente necessario al suo ruolo, ottenendo così un LLM singolo "stateless" che assume ruoli diversi tramite system prompt dedicati.

| Specialista | Cosa fa |
|---|---|
| `RESEARCHER` | Aggrega risultati da vault e web in una sintesi citando le fonti |
| `ANALYZER` | Analizza documenti, email, codice, thread — produce findings e confidence |
| `REASONER` | Ragionamento logico con extended thinking, valutazione di opzioni e rischi |
| `GENERATOR` | Genera la risposta finale tenendo conto del tuo profilo in `USER.md` |
| `STORAGE_MANAGER` | Decide cosa scrivere, dove e perché — non esegue ciecamente |

**Gruppi di esecuzione parallela**: `RETRIEVE`, `ANALYZE` e `REASON` partono contemporaneamente. `GENERATE` aspetta i loro output. `STORE` aspetta `GENERATE`.

### Azioni disponibili

Le azioni disponibili vengono scelte da Orchestrator come primo step della pipeline di esecuzione.

| Azione | Cosa fa |
|---|---|
| `RETRIEVE` | Recupera dati dal vault o dal web, senza interpretarli |
| `ANALYZE` | Analisi approfondita di un documento o di un insieme di fonti |
| `REASON` | Ragionamento critico su un problema, con extended thinking abilitato |
| `GENERATE` | Crea la risposta o un documento da mostrare all'utente |
| `STORE` | Salva o aggiorna informazioni nel vault |

### Tool disponibili

Gli strumenti vengono invocati automaticamente dalla pipeline in base al piano dell'Orchestrator.

| Tool | Descrizione |
|---|---|
| `vault_search` | Ricerca semantica + graph traversal sui `[[link]]` Obsidian + re-ranker neurale |
| `web_search` | Ricerca via DuckDuckGo — recupera le pagine in parallelo e pulisce l'HTML |
| `read_document` | Legge un documento dal vault o caricato dall'utente |
| `write_document` | Scrive un nuovo documento (non sovrascrive file esistenti) |
| `update_document` | Aggiorna una sezione di un documento esistente |

### Ricerca nel vault Obsidian

La ricerca nel vault è delegata al tool `vault_search` che combina più segnali in sequenza:

1. **Embedding semantico** — i file in `wiki/`, `synthesis/` e `raw/` (PDF e DOCX) vengono indicizzati con chunking a due livelli (split per header Markdown → sliding window con overlap) e ricerca vettoriale su ChromaDB. Usa la strategia *retrieve small, generate large*: embedding su chunk granulari per precisione di retrieval, poi chunk adiacenti riuniti automaticamente per dare allo specialista il passaggio completo. Trova contenuti rilevanti anche senza corrispondenza lessicale esatta. Le sintesi in `synthesis/` vengono recuperate con priorità (score boost ×1.2).
2. **Graph traversal** — espande i risultati seguendo i `[[link]]` interni di Obsidian in entrambe le direzioni. Se una nota è rilevante, anche le note che la citano o che essa cita vengono incluse come contesto. Maggiore cura metterai nella manutenzione della wiki, maggiore sarà la qualità dei risultati.
3. **Re-ranker** — riordina i candidati per rilevanza semantica fine prima di passare i risultati allo specialista.
4. **Knowledge graph** (opzionale) — espande ulteriormente con documenti correlati via entità condivise, abilitando retrieval multi-hop tra note che non si linkano direttamente.

### Memoria e conoscenza

- **Memoria di contesto** — la cronologia delle tue conversazioni viene resa persistente su SQLite. Solo Orchestrator la vede; gli specialisti operano stateless sul loro singolo task. Con `/compact` puoi compattarla in qualsiasi momento; superati 35 messaggi si compatta automaticamente.
- **Memoria di sessione** — dopo ogni turno con un recupero riuscito dal vault o dal web, Vesper salva automaticamente una capsule strutturata (richiesta, fonti consultate, risposta). Nei turni successivi le capsule rilevanti vengono recuperate e usate come contesto di sfondo, così Vesper ricorda cosa ha trovato anche quando il contenuto originale non è più nella finestra di contesto.
- **Profilo utente** — `vault/USER.md` contiene le preferenze apprese dalla conversazione. Viene aggiornato automaticamente quando Vesper rileva informazioni nuove su di te.
- **Watchdog** — monitora `vault/wiki/`, `vault/synthesis/` e `vault/raw/` in tempo reale con debounce per file. Ogni nota o documento salvato viene re-indicizzato automaticamente entro pochi secondi.
- **Wiki** — i file in `vault/wiki/` vengono indicizzati automaticamente. Vesper li usa per rispondere a domande sulla tua knowledge base.

---

## Task proattivi (**Work in progress**)

Vesper è proattivo **solo** se hai definito esplicitamente un task in tal senso. Non manda notifiche non richieste.

Quando crei un task ricorrente tramite chat, Vesper pianifica le azioni necessarie e le scrive nel frontmatter YAML del file task nel vault. Da quel momento in poi, esegue quelle azioni in autonomia senza doverti chiedere nulla.

### Task ricorrente

File in `vault/agenda/recurring/`, con schedule in formato cron:

```yaml
---
schedule: "0 8 * * *"
output_target: "telegram://123456789"   # il tuo TELEGRAM_ADMIN_CHAT_ID viene inserito qui
actions:
  - action: RETRIEVE
    input: {query: "ultime notizie tech", source: "web"}
  - action: GENERATE
    input: {format: "briefing mattutino"}
---
Briefing mattutino con le novità tech.
```

### Task one-off

File in `vault/agenda/one-off/`, con data di esecuzione in formato ISO 8601. Non si ripete dopo l'esecuzione:

```yaml
---
execute_at: "2025-05-20T09:00"
output_target: "telegram://123456789"   # il tuo TELEGRAM_ADMIN_CHAT_ID viene inserito qui
actions:
  - action: GENERATE
    input: {format: "promemoria"}
---
Promemoria scadenza progetto X.
```

### Se un task fallisce

Vesper ci riprova una volta in silenzio. Se fallisce di nuovo, ti manda un messaggio di errore con il nome del task, il timestamp e il motivo.

---

## Interfacce

### Telegram (Admin)

Accesso completo a tutte le funzionalità agentiche. È l'interfaccia principale.

| Comando | Funzione |
|---|---|
| `/start` | Messaggio di benvenuto |
| `/status` | Statistiche dettagliate sull'agente |
| `/reindex` | Re-indicizza i file della wiki |
| `/compact` | Compatta la memoria di contesto (produce un summary degli scambi più vecchi) |
| `/clear` | Cancella la cronologia e le capsule di memoria (SQLite + ChromaDB) |
| `/help` | Mostra la lista dei comandi disponibili |

Supporta documenti in upload fino a 20MB (limite tecnico di Telegram). Al termine di ogni risposta compaiono uno o due pulsanti inline: **Riprova** (sempre presente) e **Salva sintesi** (solo per risposte basate su ricerche web). Premendo Salva sintesi la risposta viene archiviata in `vault/synthesis/` e il pulsante diventa inerte.

### Web (pensato per uso da terzi) (**Work in progress**)

Accesso limitato. Espone alcune sezioni della wiki e permette upload di documenti per analisi o generazione di contenuti. La memoria è temporanea (si azzera alla chiusura del browser).

Le sezioni della wiki accessibili si configurano con `WEB_WIKI_ALLOWED_SECTIONS` nel `.env`.

| Endpoint | Descrizione |
|---|---|
| `POST /chat` | Invia un messaggio, ricevi la risposta via SSE |
| `POST /upload` | Carica un documento in `vault/raw/` |
| `GET /wiki/{section}/{filename}` | Legge un documento dalla wiki (solo sezioni permesse) |

---

## Struttura del vault

```
vault/
├── AGENT.md          ← regole e comportamento dell'agente
├── USER.md           ← il tuo profilo utente e le preferenze apprese
├── wiki/             ← knowledge base — indicizzata in ChromaDB (file .md)
│   └── _index.md     ← punto di partenza per la navigazione interna
├── synthesis/        ← sintesi auto (vault) e manuali (web) — indicizzate con priorità boost ×1.2
├── raw/              ← PDF e DOCX caricati via upload — indicizzati in ChromaDB
└── agenda/
    ├── recurring/    ← task ricorrenti
    └── one-off/      ← task singoli con data di esecuzione
```

`wiki/`, `synthesis/` e `raw/` (PDF/DOCX) sono indicizzati dal RAG e ricercabili semanticamente. Le altre cartelle sono accessibili all'agente tramite path diretto.

---

## Licenza

MIT. Libero uso, modifica e distribuzione con attribuzione. Vedi [LICENSE](LICENSE) per il testo completo.

© 2026 Lucas Schneider

---

## Riconoscimenti

- **[tokenjuice](https://github.com/vincentkoc/tokenjuice)** di [@vincentkoc](https://github.com/vincentkoc): la logica di riduzione del testo in `src/core/token_optimizer.py` è stata riadattata dal port Python standalone [`tokenjuice-py`](https://github.com/lschd/tokenjuice-py) del motore algoritmico di tokenjuice. Licenza MIT.
