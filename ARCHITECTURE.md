# Pipeline Architetturale

## 1. Pipeline principale (richiesta utente → risposta)

In breve:
```
  ┌────────────────────┐
  │ Interfaccia utente │ ◄── L'utente interagisce
  └─────────┬──────────┘
            ├─ <richiesta>
  ┌─────────▼──────────┐
  │    ORCHESTRATOR    │ ◄── Pianifica le azioni necessarie per rispondere all'utente
  └─────────┬──────────┘
            ├─ <piano d'azione, argomenti>
  ┌─────────▼──────────┐
  │      MANAGER       │ ◄── Organizza le risorse e raccoglie i risultati
  └─────────┬──────▲───┘
            │      ├─ <risultato>
            │      │     ┌────────────────────┐
            │      └──┬──►    Specialisti     │ ◄── Eseguono la pianificazione
            │      <task>└─────────┬──────────┘
            │                      ├─ <preferenza>
            │            ┌─────────▼──────────┐
            │            │ Aggiorna `USER.md` │ ◄── Aggiorna le preferenze dell'utente
            │            └────────────────────┘
            ├─ <risultati>
  ┌─────────▼──────────┐
  │     GENERATOR      │ ◄── Specialista che genera la risposta finale
  └─────────┬──────────┘
            ├─ <risposta>
  ┌─────────▼──────────┐
  │ Invio risp. finale │ ◄── L'utente riceve la risposta
  └────────────────────┘
```

Esteso:
```
  ┌────────────────────────────────────────────────────────────────────┐
  │                          INTERFACCIA UTENTE                        │
  └─────────────┬────────────────────────────────────────┬─────────────┘
  ┌─────────────▼──────────────┐          ┌──────────────▼─────────────┐
  │    TELEGRAM  (admin)       │          │      WEB  ("consumer")     │
  │                            │          │                            │
  │  Accesso completo          │          │  RETRIEVE ANALYZE GENERATE │ ◄── Permessi
  │                            │          │                            │
  │  cronologia → SQLite       │          │  sessione → RAM  (TTL)     │ ◄── Privacy policy
  │  upload ≤ 20 MB (limite)   │          │  upload ≤ 20 MB (default)  │
  └─────────────┬──────────────┘          └──────────────┬─────────────┘
                └────────────────────┬───────────────────┘
                                     │
                      user_message + available_actions
                         + history  (solo Telegram)
                                     │
                      ┌──────────────▼─────────────┐
                      │   AbstractInterface        │  pipeline condivisa
                      │   process_request()        │  src/interfaces/base.py
                      └──────────────┬─────────────┘
                                     │
                      ┌──────────────▼─────────────┐
                      │      SESSION MEMORY        │  ChromaDB collection
                      │      session_memory.py     │  session_id = str(chat_id)
                      │                            │             o session_id web
                      │  query(session_id, text)   │ ◄── pre-Orchestrator
                      └──────────────┬─────────────┘
                                     │
                      context_memory (capsule turni precedenti)
                                     │
                      ┌──────────────▼──────────────┐
                      │    VAULT PRE-SEARCH         │  zero LLM
                      │    vault_search.py          │  top_k=3, no rerank
                      │                             │  no link traversal
                      │  se score ≥ 0.6:            │
                      │    vault_hint = "Trovati N  │
                      │    doc rilevanti (max: X)"  │
                      │  altrimenti vault_hint=None │
                      └──────────────┬──────────────┘
                                     │
                          vault_hint (str | None)
                                     │
                      ┌──────────────▼─────────────┐
                      │        ORCHESTRATOR        │ ◄── LLM call
                      │       orchestrator.py      │     max_tokens = 1024
                      │                            │
                      │  • vede le conversazioni   │
                      │  • pianifica le azioni     │
                      │  • definisce gli argomenti │
                      └──────────────┬─────────────┘
                                     │
                               piano JSON
                  { "AZIONE": { "argomento": "valore" } }
                                     │
                      ┌──────────────▼─────────────┐
                      │      _post_plan_hook()     │  hook platform-specific
                      │  (es. filtra sezioni wiki  │  default: identità
                      │   per Web su vault_search) │  override in WebInterface
                      └──────────────┬─────────────┘
                                     │
                      ┌──────────────▼─────────────┐
                      │          MANAGER           │  zero LLM
                      │         manager.py         │  _resolve_groups()
                      │                            │  asyncio.gather
                      └───┬──────────┬─────────┬───┘
                          │          │         │
                   ┌──────▼──┐  ┌────▼────┐  ┌─▼──────┐
                   │GRUPPO 0 │  │GRUPPO 1 │  │GRUPPO 2│
                   │parallelo│  │attende 0│  │attende1│
                   ├─────────┤  ├─────────┤  ├────────┤
                   │RETRIEVE │  │GENERATE │  │ STORE  │
                   │ANALYZE  │  └────┬────┘  └───┬────┘
                   │REASON   │       │           │
                   └────┬────┘       ├─attende   ├─salta se GRUPPO 1 fallisce
                        │            │ GRUPPO 0  │
                        │            │           │
              ┌─────────▼────────────▼───────────▼───────────┐
              │          per ogni azione attiva              │
              │                                              │
              │  1. Manager invoca i TOOL Python             │
              │     (vault_search, web_search, …)            │
              │                                              │
              │  2. Manager passa i risultati al SUBAGENT    │
              │     context = {                              │
              │       "request":      richiesta utente,      │
              │       "tool_results": { vault_results, … }   │
              │     }                                        │
              │                                              │
              │  3. Subagent.run() → LLM call                │
              │     output: { "success": bool, "output": … } │
              └──────────────────────────────────────────────┘
                                      │
                             risultati accumulati
                           in RequestContext.results
                                      │
                       ┌──────────────▼──────────────┐
                       │      _save_synthesis()      │  zero LLM
                       │   (dentro Manager.execute)  │
                       │                             │
                       │ trigger: vault_search OK    │
                       │       + GENERATE success    │
                       │                             │
                       │ • estrae tag dai sorgenti   │
                       │ • scrive vault/synthesis/   │
                       │   YYYY-MM-DD-<slug>.md      │
                       └──────────────┬──────────────┘
                                      │
                       ┌──────────────▼──────────────┐
                       │       POLICY USER.md        │ ◄── LLM call
                       │  (dentro Manager.execute)   │
                       │                             │
                       │ •legge results["request"]   │
                       │ •rileva nuove preferenze    │
                       │ •aggiorna vault/USER.md     │
                       └──────────────┬──────────────┘
                                      │
                        ┌─────────────▼─────────────┐
                        │         GENERATOR         │
                        │        specialista        │
                        │                           │
                        │ Genera la risposta finale │
                        │ con i risultati raccolti  │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                               risposta finale
                            all'interfaccia utente
                                      │
                      ┌───────────────▼────────────┐
                      │      SESSION MEMORY        │
                      │                            │
                      │  store(session_id, ctx)    │ ◄── post-turno, fire-and-forget
                      │  (asyncio.create_task)     │     salva capsule strutturata
                      └────────────────────────────┘
```

## 2. Specialisti (subagent) e tool per azione

```
  AZIONE       TOOL INVOCATO DAL MANAGER         SUBAGENT (LLM)
  ────────────────────────────────────────────────────────────────
  RETRIEVE  ──► vault_search                  ──► RESEARCHER
                web_search                         thinking=false
                read_document                      output: sources + summary

  ANALYZE   ──► read_document                 ──► ANALYZER
                vault_search                       thinking=true
                                                   output: findings + confidence

  REASON    ──► (nessun tool; riceve contesto ──► REASONER
                da RETRIEVE/ANALYZE)               thinking=true
                                                   output: reasoning + conclusion

  GENERATE  ──► legge AGENT.md + USER.md      ──► GENERATOR
                (vault root, non tool calls)       thinking=false
                                                   output: stringa risposta

  STORE     ──► write_document                ──► STORAGE_MANAGER
                update_document                    thinking=false
                                                   output: action + target_path
```

---

## 3. Stack RAG e vault

```
    wiki/  ◄── watchdog (debounce per-file)
      │        watchdog.py
  ┌───▼──────────────────────────────┐
  │           INDEXER                │  ◄── indexer.py
  │                                  │
  │ 1. mtime check bulk              │
  │ 2. header split → sliding window │  ◄── chunking a due livelli
  │ 3. SHA-256 per skip chunk        │
  │ 4. batch embedding (bge-m3)      │
  │ 5. upsert ChromaDB               │
  │ 6. elimina zombie                │
  └─────────────────┬────────────────┘
                    │
                    ▼
                ChromaDB (locale)
                    │
  ┌─────────────────▼────────────────┐
  │         vault_search             │  ◄── vault_search.py
  │                                  │
  │  1. embedding semantico          │  ◄── ricerca vettoriale
  │  2. graph traversal [[link]]     │  ◄── espande nei due sensi
  │  3. rerank (opzionale)           │  ◄── HTTP → server su RERANKER_PORT
  │                                  │             over-fetch top_k*5 → riordina
  │  fallback graceful se            │
  │  re-ranker non raggiungibile     │
  └──────────────────────────────────┘
```

## 4. Sistema proattivo (bypassa l'Orchestrator)

```
  agenda/
  ├── recurring/task.md   (frontmatter: schedule cron)
  └── one-off/task.md     (frontmatter: execute_at ISO)
         │
  ┌──────▼──────────────┐
  │   VesperScheduler   │ ◄── scheduler.py
  │  (APScheduler via   │ ◄── legge agenda/ al boot
  │   python-telegram-  │
  │   bot JobQueue)     │
  └──────────┬──────────┘
             ├─ <trigger> (cron / one-shot)
  ┌──────────▼──────────┐
  │ ProactiveDispatcher │ ◄── dispatcher.py
  │                     │
  │  legge frontmatter  │ ◄── costruisce piano Manager direttamente
  │  bypassa Orchestr.  │     (niente LLM planning)
  │  _dispatch_with_    │
  │    retry() ×2       │
  └──────────┬──────────┘
             │
             ▼
          MANAGER  (stesso path della pipeline normale da qui in poi)
             │
             ▼
  output_target  →  telegram://<chat_id>  |  web://<session>
```

## 5. Sequenza di avvio (main.py)

```
  main.py
     │
     ├─ 1. auto-restart nel venv  (Python ≥ 3.10)
     │
     ├─ 2. preflight: vault OK? LLM server raggiungibile?
     │
     ├─ 3. ChromaDB + embedding ◄── occupa VRAM *prima* dell'LLM
     │                              così free_mib riflette il consumo reale
     ├─ 4. index_vault()
     │
     ├─ 5.  ┌──────────────────────────────────┐  asyncio task paralleli
     │      │  avvio LLM server (llama-server) │  solo se LLM_MANAGED=true
     │      │  avvio re-ranker server          │
     │      └──────────────────────────────────┘
     │
     ├─ 6. Watchdog  +  TelegramInterface  +  VesperScheduler  +  FastAPI/uvicorn
     │
     ├─ 7. attende completamento LLM e re-ranker
     │
     ├─ 8. fetch_n_ctx() ◄── legge n_ctx effettivo da GET /props (solo LLM_MANAGED=true)
     │
     └─ 9. avvia polling Telegram e web server
```

## 6. Infrastruttura LLM

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     LLMClient  (singleton)                   │
  │                          client.py                           │
  │                                                              │
  │  complete(system_prompt, user_message, …)                    │
  │    │                                                         │
  │    ├─ 1. prepende /think o /no_think  (Qwen3 tokenizer)      │
  │    ├─ 2. POST  {base_url}/chat/completions                   │
  │    ├─ 3. strip blocchi <think>…</think>  + orfani </think>   │
  │    ├─ 4. validate_json()  → ValueError se fallisce           │
  │    └─ 5. log  prompt=N compl=M t→Xs → Y tok/s                │
  │                                                              │
  │  fetch_n_ctx()  →  GET {base_url}/props                      │
  │    legge default_generation_settings.n_ctx dal server        │
  │    se LLM_N_CTX è impostato nel .env: applica come cap       │
  │    fallback a 8192 se /props non risponde                    │
  │    risultato in LLMClient.n_ctx  (usato dal Manager)         │
  │                                                              │
  │  load_prompt(name)  →  src/prompts/{name}.md  (cache RAM)    │
  └──────────────────────────────────────────────────────────────┘
              │
  ┌───────────▼──────────────────────────────────────────────────┐
  │               llama-server  (OpenAI-compat API)              │
  │                                                              │
  │  modello principale  (GGUF, Qwen3)                           │
  │                                                              │
  │  GPU layout  (src/core/llm/gpu_manager.py)                   │
  │    GPU PCIe veloce  →  layer LLM (tensor split proporzionale)│
  │    GPU PCIe lento   →  embedding + re-ranker                 │
  │    n_gpu_layers=-1  →  auto da VRAM residua post-embedding   │
  └──────────────────────────────────────────────────────────────┘
               │                                   │
  ┌────────────▼─────────────┐        ┌────────────▼─────────────┐
  │  bge-m3 (embedding)      │        │  Qwen3-Reranker (GGUF)   │
  │  ChromaDB local          │        │  server su RERANKER_PORT │
  └──────────────────────────┘        └──────────────────────────┘
```

## 7. Contratto context Manager → Subagent

```
  context passato a ogni Subagent.run():

  {
    "request":          str      ← richiesta originale dell'utente
    "tool_results": {
      "vault_results":  [...],   ← chunk interi (bounded da CHUNK_SIZE ≈ 4000 char)
      "web_results":    [...],   ← contenuto HTML pulito
      "document":       str,
      …
    }
  }

  GENERATOR riceve campi aggiuntivi (oltre ai due base):
  {
    "conversation_history": [...],   ← cronologia da SQLite (solo Telegram)
    "context_memory":       [...],   ← capsule di sessione rilevanti da session_memory.query()
                                       presenti solo se Orchestrator ha omesso RETRIEVE
                                       (le capsule fungono da contesto di sfondo per Generator)
    "user_profile":         str,     ← contenuto di vault/USER.md
    "agent_profile":        str,     ← contenuto di vault/AGENT.md
    "instructions":         str,     ← intent originale passato dall'Orchestrator
                                       (usato nei follow-up multi-turno)
    "format":               str,     ← tipo di output richiesto (es. "risposta", "briefing")
  }

  safety n_ctx — token_optimizer (BaseSubagent.run(), passo 3.5):
    dopo il filtro context_fields e prima della serializzazione JSON,
    optimize_context_dict() riduce vault_results, web_results, document e
    context_memory con regole content-type consapevoli (head/tail per tipo).
    budget = max(4000, (n_ctx − 2000) * 3) char — lascia ~2000 token per output.
    Il Manager non applica nessuna troncatura inline.

  output atteso dal subagent:
  {
    "success": bool,
    "output":  any        ← campi tipizzati per ruolo dentro "output"
  }

  max_tokens:  nessun cap (EOS naturale del modello)
  limite reale: n_ctx − prompt_tokens
    n_ctx rilevato automaticamente da GET /props (metadato GGUF del modello)
    LLM_N_CTX nel .env: se impostato, usato come cap sul valore rilevato
```
