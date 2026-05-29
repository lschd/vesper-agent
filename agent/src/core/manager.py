"""
Manager — orchestrazione algoritmica pura (una sola LLM call: policy USER.md).

Responsabilità:
- Legge il piano dall'Orchestrator e risolve i gruppi di esecuzione
- Esegue ogni gruppo in parallelo con asyncio.gather
- Chiama i tool prima di ogni subagent, passa i risultati come contesto
- Registra tool calls, risultati ed errori nel RequestContext
- Esegue la policy pre-risposta USER.md
"""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_KNOWN_ACTIONS = frozenset({"RETRIEVE", "ANALYZE", "REASON", "GENERATE", "STORE"})


# ---------------------------------------------------------------------------
# Risoluzione dipendenze
# ---------------------------------------------------------------------------

def _resolve_groups(plan: dict) -> list[list[str]]:
    """
    Assegna ogni azione a un gruppo di esecuzione seguendo le regole statiche.

    Regole:
      RETRIEVE, ANALYZE, REASON → gruppo 0 (indipendenti tra loro)
      GENERATE → gruppo 1 se RETRIEVE, ANALYZE o REASON sono nel piano, altrimenti 0
      STORE → dipende da GENERATE (se nel piano), altrimenti da RETRIEVE;
              se nessuno dei due è nel piano, gruppo 0

    Returns:
        Lista ordinata di gruppi; ogni gruppo è una lista di azioni da eseguire in parallelo.
    """
    actions = set(plan.keys()) & _KNOWN_ACTIONS
    assigned: dict[str, int] = {}

    for a in ("RETRIEVE", "ANALYZE", "REASON"):
        if a in actions:
            assigned[a] = 0

    if "GENERATE" in actions:
        has_upstream = any(a in actions for a in ("RETRIEVE", "ANALYZE", "REASON"))
        assigned["GENERATE"] = 1 if has_upstream else 0

    if "STORE" in actions:
        if "GENERATE" in actions:
            assigned["STORE"] = assigned["GENERATE"] + 1
        elif "RETRIEVE" in actions:
            assigned["STORE"] = assigned.get("RETRIEVE", 0) + 1
        else:
            assigned["STORE"] = 0

    if not assigned:
        return []

    max_group = max(assigned.values())
    return [
        [a for a, g in assigned.items() if g == grp]
        for grp in range(max_group + 1)
        if any(g == grp for g in assigned.values())
    ]


def _prerequisites_failed(plan: dict, context) -> list[str]:
    """
    Controlla se i prerequisiti di STORE hanno fallito.

    STORE dipende da GENERATE (se nel piano), altrimenti da RETRIEVE.
    Restituisce i nomi delle azioni prerequisite fallite.
    """
    if "GENERATE" in plan:
        res = context.results.get("GENERATE", {"success": False})
        if isinstance(res, dict) and not res.get("success", False):
            return ["GENERATE"]
    elif "RETRIEVE" in plan:
        res = context.results.get("RETRIEVE", {"success": False})
        if isinstance(res, dict) and not res.get("success", False):
            return ["RETRIEVE"]
    return []


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class Manager:
    """Esecuzione algoritmica del piano prodotto dall'Orchestrator."""

    async def execute(
        self,
        plan: dict,
        context,
        status_callback,
        autonomous_mode: bool = False,
        context_memory: list[dict] | None = None,
        conversation_history: list[dict] | None = None,
    ):
        """
        Esegue il piano risolvendo dipendenze e coordinando subagents e tool.

        Args:
            plan:            Output dell'Orchestrator: {"AZIONE": {"input": ...}}.
            context:         RequestContext da popolare durante l'esecuzione.
            status_callback: Callable async per aggiornamenti di stato verso l'utente.
            autonomous_mode: Riservato per uso futuro (proattività senza interazione).
            context_memory:  Capsule di memoria di sessione da passare al Generator.

        Returns:
            RequestContext aggiornato con risultati, tool calls ed errori.
        """
        for a in set(plan.keys()) - _KNOWN_ACTIONS:
            logger.warning("Manager: azione sconosciuta '%s' ignorata", a)

        groups = _resolve_groups(plan)
        logger.info(
            "Manager.execute: %d azioni, %d gruppi — %s",
            len(plan), len(groups), groups,
        )

        # Memorizza context_memory e conversation_history sull'istanza per renderli disponibili a _execute_generate
        self._context_memory = context_memory
        self._conversation_history = conversation_history

        for group_idx, group in enumerate(groups):
            logger.info("Manager: gruppo %d — %s", group_idx, group)

            coroutines = [
                self._execute_action(action, plan[action], plan, context, status_callback)
                for action in group
            ]
            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for action, res in zip(group, results):
                if isinstance(res, Exception):
                    logger.error(
                        "Manager: eccezione non catturata in '%s' — %s: %s",
                        action, type(res).__name__, res,
                    )
                    context.add_error(action, f"eccezione inattesa: {res}")

        await self._save_synthesis(context)
        await self._run_user_md_policy(plan, context)
        return context

    # ------------------------------------------------------------------
    # Dispatcher azioni
    # ------------------------------------------------------------------

    async def _execute_action(
        self,
        action: str,
        action_input: dict,
        plan: dict,
        context,
        status_callback,
    ) -> None:
        dispatch = {
            "RETRIEVE": self._execute_retrieve,
            "ANALYZE":  self._execute_analyze,
            "REASON":   self._execute_reason,
            "GENERATE": self._execute_generate,
            "STORE":    self._execute_store,
        }
        handler = dispatch.get(action)
        if handler is None:
            return

        try:
            await handler(action_input, plan, context, status_callback)
        except Exception as exc:
            logger.error(
                "Manager: errore in '%s' — %s: %s", action, type(exc).__name__, exc
            )
            context.add_error(action, f"{type(exc).__name__}: {exc}")
            try:
                await status_callback(f"Errore durante {action}: {exc}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    async def _execute_retrieve(
        self, action_input: dict, plan: dict, context, status_callback
    ) -> None:
        try:
            from src.tools.vault_search import vault_search
            from src.tools.web_search import async_web_search
            from src.tools.read_document import read_document
            from src.subagents.researcher import Researcher
            from src.core.llm.client import get_llm_client
            from config import Config
        except ImportError:
            from tools.vault_search import vault_search          # type: ignore[no-redef]
            from tools.web_search import async_web_search        # type: ignore[no-redef]
            from tools.read_document import read_document        # type: ignore[no-redef]
            from subagents.researcher import Researcher          # type: ignore[no-redef]
            from core.llm.client import get_llm_client           # type: ignore[no-redef]
            from config import Config                            # type: ignore[no-redef]

        query = action_input.get("query", "")
        source = action_input.get("source", "vault")
        path = action_input.get("path")
        tool_context: dict = {}
        tools_used: list[str] = []

        if source in ("vault", "both") and query:
            await status_callback("Sto cercando informazioni nel vault...")
            t0 = time.perf_counter()
            try:
                results = await vault_search(query, rerank=Config.reranker_enabled)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                context.add_tool_call("vault_search", {"query": query}, results, duration_ms)
                tool_context["vault_results"] = [{**r} for r in results]
                tools_used.append("vault_search")
            except Exception as exc:
                logger.warning("Manager RETRIEVE: vault_search fallita — %s", exc)
                context.add_error("RETRIEVE", f"vault_search: {exc}")

        if source in ("web", "both") and query:
            await status_callback("Sto cercando informazioni sul web...")
            t0 = time.perf_counter()
            try:
                results = await async_web_search(query)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                context.add_tool_call("web_search", {"query": query}, results, duration_ms)
                tool_context["web_results"] = [{**r} for r in results]
                tools_used.append("web_search")
            except Exception as exc:
                logger.warning("Manager RETRIEVE: web_search fallita — %s", exc)
                context.add_error("RETRIEVE", f"web_search: {exc}")

        if path:
            if path.startswith(("http://", "https://")):
                logger.warning(
                    "Manager RETRIEVE: path '%s' è un URL — usa source='web' con query. Ignorato.",
                    path,
                )
            else:
                await status_callback(f"Sto leggendo il documento: {path}...")
                t0 = time.perf_counter()
                try:
                    doc = await read_document(path)
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    context.add_tool_call("read_document", {"path": path}, doc, duration_ms)
                    tool_context["document"] = doc
                    tools_used.append("read_document")
                except Exception as exc:
                    logger.warning("Manager RETRIEVE: read_document fallita — %s", exc)
                    context.add_error("RETRIEVE", f"read_document: {exc}")

        task = (
            f"Recupera informazioni per la query: {query!r}"
            if query else f"Recupera il documento: {path!r}"
        )

        run_context: dict = {**action_input}
        if tool_context:
            run_context["tool_results"] = tool_context
        if user_request := context.results.get("request"):
            run_context["request"] = user_request

        logger.debug("Manager RETRIEVE: context keys → %s", list(run_context.keys()))
        researcher = Researcher()
        result = await researcher.run(
            task=task,
            context=run_context,
            tools=tools_used,
        )

        if not result.get("success"):
            logger.warning("Manager RETRIEVE: Researcher fallito — %s", result.get("output"))
            context.add_error("RETRIEVE", result.get("output", "errore sconosciuto"))

        context.add_result("RETRIEVE", result)

    # ------------------------------------------------------------------
    # ANALYZE
    # ------------------------------------------------------------------

    async def _execute_analyze(
        self, action_input: dict, plan: dict, context, status_callback
    ) -> None:
        try:
            from src.tools.read_document import read_document
            from src.subagents.analyzer import Analyzer
        except ImportError:
            from tools.read_document import read_document   # type: ignore[no-redef]
            from subagents.analyzer import Analyzer         # type: ignore[no-redef]

        path = action_input.get("path")
        tool_context: dict = {}
        tools_used: list[str] = []

        if path:
            await status_callback(f"Sto leggendo il documento da analizzare: {path}...")
            t0 = time.perf_counter()
            try:
                doc = await read_document(path)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                context.add_tool_call("read_document", {"path": path}, doc, duration_ms)
                tool_context["document"] = doc
                tools_used.append("read_document")
            except Exception as exc:
                logger.warning("Manager ANALYZE: read_document fallita — %s", exc)
                context.add_error("ANALYZE", f"read_document: {exc}")

        await status_callback("Sto analizzando il contenuto...")

        run_context: dict = {**action_input}
        if tool_context:
            run_context["tool_results"] = tool_context
        if user_request := context.results.get("request"):
            run_context["request"] = user_request

        logger.debug("Manager ANALYZE: context keys → %s", list(run_context.keys()))
        analyzer = Analyzer()
        result = await analyzer.run(
            task=f"Analizza: {action_input}",
            context=run_context,
            tools=tools_used,
        )

        if not result.get("success"):
            logger.warning("Manager ANALYZE: Analyzer fallito — %s", result.get("output"))
            context.add_error("ANALYZE", result.get("output", "errore sconosciuto"))

        context.add_result("ANALYZE", result)

    # ------------------------------------------------------------------
    # REASON
    # ------------------------------------------------------------------

    async def _execute_reason(
        self, action_input: dict, plan: dict, context, status_callback
    ) -> None:
        try:
            from src.subagents.reasoner import Reasoner
        except ImportError:
            from subagents.reasoner import Reasoner  # type: ignore[no-redef]

        await status_callback("Sto ragionando sulla richiesta...")

        run_context: dict = dict(action_input)
        if user_request := context.results.get("request"):
            run_context["request"] = user_request
        tool_results: dict = {}
        for key in ("RETRIEVE", "ANALYZE"):
            if key in context.results:
                r = context.results[key]
                tool_results[key.lower()] = r.get("output") if isinstance(r, dict) else r
        if tool_results:
            run_context["tool_results"] = tool_results

        logger.debug("Manager REASON: context keys → %s", list(run_context.keys()))
        reasoner = Reasoner()
        result = await reasoner.run(
            task=f"Ragiona su: {action_input}",
            context=run_context,
            tools=[],
        )

        if not result.get("success"):
            logger.warning("Manager REASON: Reasoner fallito — %s", result.get("output"))
            context.add_error("REASON", result.get("output", "errore sconosciuto"))

        context.add_result("REASON", result)

    # ------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------

    async def _execute_generate(
        self, action_input: dict, plan: dict, context, status_callback
    ) -> None:
        try:
            from src.subagents.generator import Generator
        except ImportError:
            from subagents.generator import Generator  # type: ignore[no-redef]

        await status_callback("Sto generando la risposta...")

        gen_context: dict = dict(action_input)

        if user_request := context.results.get("request"):
            gen_context["request"] = user_request

        tool_results: dict = {}
        for key in ("RETRIEVE", "ANALYZE", "REASON"):
            if key in context.results:
                r = context.results[key]
                tool_results[key.lower()] = r.get("output") if isinstance(r, dict) else r

        # Safety net: se ci sono documenti caricati e nessun tool ha già prodotto risultati
        # (Orchestrator ha pianificato solo GENERATE), leggi i documenti prima di generare.
        if (uploaded_docs := context.results.get("uploaded_documents")) and not tool_results:
            try:
                from src.tools.read_document import read_document as _rd
            except ImportError:
                from tools.read_document import read_document as _rd  # type: ignore[no-redef]
            for doc_path in uploaded_docs:
                try:
                    t0 = time.perf_counter()
                    doc_content = await _rd(doc_path)
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    context.add_tool_call("read_document", {"path": doc_path}, doc_content, duration_ms)
                    tool_results["document"] = doc_content
                    break
                except Exception as exc:
                    logger.warning("Manager GENERATE: auto-read '%s' fallita — %s", doc_path, exc)

        if tool_results:
            gen_context["tool_results"] = tool_results
        if getattr(self, "_context_memory", None):
            gen_context["context_memory"] = self._context_memory
        if getattr(self, "_conversation_history", None):
            gen_context["conversation_history"] = self._conversation_history

        logger.debug("Manager GENERATE: context keys → %s", list(gen_context.keys()))
        generator = Generator()
        result = await generator.run(
            task=f"Genera la risposta finale: {action_input}",
            context=gen_context,
            tools=[],
        )

        if not result.get("success"):
            logger.warning("Manager GENERATE: Generator fallito — %s", result.get("output"))
            context.add_error("GENERATE", result.get("output", "errore sconosciuto"))

        context.add_result("GENERATE", result)

    # ------------------------------------------------------------------
    # STORE
    # ------------------------------------------------------------------

    async def _execute_store(
        self, action_input: dict, plan: dict, context, status_callback
    ) -> None:
        try:
            from src.subagents.storage_manager import StorageManager
        except ImportError:
            from subagents.storage_manager import StorageManager  # type: ignore[no-redef]

        failed_prereqs = _prerequisites_failed(plan, context)
        if failed_prereqs:
            msg = f"prerequisiti falliti: {', '.join(failed_prereqs)}"
            logger.warning("Manager STORE: %s — salto", msg)
            context.add_error("STORE", msg)
            await status_callback("Salto il salvataggio: prerequisiti non soddisfatti.")
            return

        path = action_input.get("path", "")
        section = action_input.get("section")
        is_update = bool(action_input.get("update")) or section is not None

        content: str = action_input.get("content", "")
        if not content:
            generate_res = context.results.get("GENERATE", {})
            if isinstance(generate_res, dict) and generate_res.get("success"):
                content = str(generate_res.get("output", ""))
        if not content and "RETRIEVE" in context.results:
            retrieve_res = context.results["RETRIEVE"]
            output = retrieve_res.get("output", retrieve_res) if isinstance(retrieve_res, dict) else retrieve_res
            content = str(output)

        if not path:
            logger.warning("Manager STORE: path non specificato — salto")
            context.add_error("STORE", "path non specificato")
            return

        if not content:
            logger.warning("Manager STORE: contenuto vuoto — salto")
            context.add_error("STORE", "nessun contenuto da salvare")
            return

        storage = StorageManager()

        if is_update:
            await status_callback(f"Sto aggiornando il documento: {path}...")
            t0 = time.perf_counter()
            result = await storage.update(path, content, section)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            tool = "update_document"
            if not result.get("success") and "non trovato" in str(result.get("output", "")):
                logger.info("Manager STORE: '%s' inesistente, fallback a write_document", path)
                await status_callback(f"File non trovato, creo il documento: {path}...")
                t0 = time.perf_counter()
                result = await storage.store(path, content)
                duration_ms = int((time.perf_counter() - t0) * 1000)
                tool = "write_document"
        else:
            await status_callback(f"Sto salvando il documento: {path}...")
            t0 = time.perf_counter()
            result = await storage.store(path, content)
            duration_ms = int((time.perf_counter() - t0) * 1000)
            tool = "write_document"

        context.add_tool_call(
            tool,
            {"path": path, "content": content[:500]},
            result,
            duration_ms,
        )

        if result.get("success"):
            context.documents_modified.append(path)
        else:
            logger.warning("Manager STORE: %s fallita — %s", tool, result.get("output"))
            context.add_error("STORE", result.get("output", "errore sconosciuto"))

        context.add_result("STORE", result)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def _save_synthesis(self, context) -> None:
        """
        Salva una nota di sintesi in vault/synthesis/ dopo RETRIEVE da vault + GENERATE riusciti.
        Trigger: GENERATE success + almeno un vault_search con risultati non vuoti.
        Nessun LLM aggiuntivo — operazione puramente algoritmica.
        """
        import re
        import datetime
        from pathlib import Path as _Path

        # 1. GENERATE deve essere riuscito con output non vuoto
        generate_res = context.results.get("GENERATE", {})
        if not (isinstance(generate_res, dict) and generate_res.get("success")):
            return
        response_text = str(generate_res.get("output", "")).strip()
        if not response_text:
            return

        # 2. Deve esserci almeno una vault_search con risultati non vuoti
        vault_calls = [
            tc for tc in context.tools_called
            if tc.get("tool") == "vault_search" and tc.get("output")
        ]
        if not vault_calls:
            return

        # 3. Raccoglie paths unici dei documenti consultati e la query
        seen: set[str] = set()
        all_paths: list[str] = []
        query = ""
        for tc in vault_calls:
            if not query:
                query = tc.get("input", {}).get("query", "")
            for item in (tc.get("output") or []):
                path = item.get("path", "")
                if path and path not in seen:
                    seen.add(path)
                    all_paths.append(path)
        if not all_paths:
            return

        # 4. Estrae tag YAML dai file sorgente del vault
        try:
            from config import Config
            vault_path = _Path(Config.vault_path)
        except Exception:
            return

        all_tags: list[str] = ["sintesi"]
        _FM = re.compile(r'^---\n(.*?)\n---', re.DOTALL)
        _TAG_INLINE = re.compile(r'^tags:\s*\[([^\]]*)\]', re.MULTILINE)
        _TAG_BLOCK_HEAD = re.compile(r'^tags:\s*$', re.MULTILINE)
        _TAG_BLOCK_ITEM = re.compile(r'^\s+-\s+(.+)')

        for doc_path in all_paths:
            try:
                text = (vault_path / doc_path).read_text(encoding="utf-8")
                fm_match = _FM.match(text)
                if not fm_match:
                    continue
                fm_body = fm_match.group(1)
                m_inline = _TAG_INLINE.search(fm_body)
                if m_inline:
                    for raw in m_inline.group(1).split(','):
                        t = raw.strip().strip("\"'")
                        if t and t not in all_tags:
                            all_tags.append(t)
                else:
                    m_block = _TAG_BLOCK_HEAD.search(fm_body)
                    if m_block:
                        for line in fm_body[m_block.end():].split('\n'):
                            m_item = _TAG_BLOCK_ITEM.match(line)
                            if m_item:
                                t = m_item.group(1).strip().strip("\"'")
                                if t and t not in all_tags:
                                    all_tags.append(t)
                            elif line and not line.startswith(' '):
                                break
            except Exception as exc:
                logger.debug("_save_synthesis: impossibile leggere tag da '%s': %s", doc_path, exc)

        # 5. Costruisce filename univoco in vault/synthesis/
        today = datetime.date.today().isoformat()
        slug = re.sub(r'[^\w\s-]', '', query.lower())[:40].strip()
        slug = re.sub(r'[\s_]+', '-', slug).strip('-')
        base = f"{today}-{slug}" if slug else today

        synthesis_dir = vault_path / "synthesis"
        fname = f"{base}.md"
        counter = 1
        while (synthesis_dir / fname).exists():
            fname = f"{base}-{counter}.md"
            counter += 1

        # 6. Costruisce contenuto (frontmatter Obsidian + corpo)
        tags_yaml = "\n".join(f"  - {t}" for t in all_tags)
        # Link stile Obsidian: solo nome file senza estensione
        links_md = "\n".join(f"- [[{_Path(p).stem}]]" for p in all_paths)
        title = (context.results.get("request", "") or query).strip()

        content = (
            f"---\ntags:\n{tags_yaml}\ndate: {today}\n---\n\n"
            f"# {title}\n\n"
            f"## Fonti\n\n{links_md}\n\n"
            f"## Risposta\n\n{response_text}\n"
        )

        # 7. Salva — errori non bloccanti per la pipeline principale
        try:
            from src.tools.write_document import write_document
        except ImportError:
            from tools.write_document import write_document  # type: ignore[no-redef]

        rel_path = f"synthesis/{fname}"
        try:
            await write_document(rel_path, content)
            logger.info("_save_synthesis: salvata '%s'", rel_path)
            context.documents_modified.append(rel_path)
        except FileExistsError:
            logger.debug("_save_synthesis: '%s' esiste già — saltato", rel_path)
        except Exception as exc:
            logger.warning("_save_synthesis: errore salvataggio — %s", exc)

    async def save_web_synthesis(self, context) -> str | None:
        """
        Salva una nota di sintesi in vault/synthesis/ per ricerche web.
        Chiamato manualmente (es. dal pulsante Telegram) dopo RETRIEVE da web + GENERATE riusciti.
        Restituisce il path relativo salvato, o None se non applicabile/fallito.
        """
        import re
        import datetime
        from pathlib import Path as _Path

        generate_res = context.results.get("GENERATE", {})
        if not (isinstance(generate_res, dict) and generate_res.get("success")):
            return None
        response_text = str(generate_res.get("output", "")).strip()
        if not response_text:
            return None

        web_calls = [
            tc for tc in context.tools_called
            if tc.get("tool") == "web_search" and tc.get("output")
        ]
        if not web_calls:
            return None

        seen: set[str] = set()
        all_urls: list[str] = []
        query = ""
        for tc in web_calls:
            if not query:
                query = tc.get("input", {}).get("query", "")
            for item in (tc.get("output") or []):
                url = item.get("url", "")
                if url and url not in seen:
                    seen.add(url)
                    all_urls.append(url)

        try:
            from config import Config
            vault_path = _Path(Config.vault_path)
        except Exception:
            return None

        today = datetime.date.today().isoformat()
        slug = re.sub(r'[^\w\s-]', '', query.lower())[:40].strip()
        slug = re.sub(r'[\s_]+', '-', slug).strip('-')
        base = f"{today}-{slug}" if slug else today

        synthesis_dir = vault_path / "synthesis"
        fname = f"{base}.md"
        counter = 1
        while (synthesis_dir / fname).exists():
            fname = f"{base}-{counter}.md"
            counter += 1

        tags_yaml = "  - sintesi\n  - web"
        links_md = "\n".join(f"- {url}" for url in all_urls)
        title = (context.results.get("request", "") or query).strip()

        content = (
            f"---\ntags:\n{tags_yaml}\ndate: {today}\n---\n\n"
            f"# {title}\n\n"
            f"## Fonti\n\n{links_md}\n\n"
            f"## Risposta\n\n{response_text}\n"
        )

        try:
            from src.tools.write_document import write_document
        except ImportError:
            from tools.write_document import write_document  # type: ignore[no-redef]

        rel_path = f"synthesis/{fname}"
        try:
            await write_document(rel_path, content)
            logger.info("save_web_synthesis: salvata '%s'", rel_path)
            return rel_path
        except FileExistsError:
            logger.debug("save_web_synthesis: '%s' esiste già — saltato", rel_path)
            return None
        except Exception as exc:
            logger.warning("save_web_synthesis: errore salvataggio — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Policy USER.md
    # ------------------------------------------------------------------

    async def _run_user_md_policy(self, plan: dict, context) -> None:
        """
        Call LLM leggera per rilevare nuove preferenze utente da salvare in USER.md.
        Eseguita al termine dell'elaborazione, prima di restituire il contesto.

        Analizza SOLO il messaggio dell'utente, non la risposta generata.
        La risposta riflette le regole di AGENT.md (tono, stile, lingua) — includerla
        inquinerebbe USER.md con comportamenti dell'agente, non preferenze dell'utente.
        """
        # Esegue solo se la richiesta è disponibile e GENERATE è andato a buon fine
        original_request = context.results.get("request", "").strip()
        if not original_request:
            return

        generate_res = context.results.get("GENERATE", {})
        if not (isinstance(generate_res, dict) and generate_res.get("success")):
            return

        system_prompt = (
            "Analizza il messaggio dell'utente. Rispondi SOLO con un oggetto JSON, senza spiegazioni. "
            "Se il messaggio dichiara esplicitamente una preferenza, caratteristica o informazione "
            "personale dell'utente da ricordare (es. 'sono uno sviluppatore', 'preferisco risposte brevi', "
            "'usa il lei con me'): "
            '{"answer": true, "update_with": "..."} '
            "Altrimenti, se non c'è nessuna dichiarazione esplicita dell'utente: "
            '{"answer": false, "update_with": ""}'
        )
        user_message = f"## Messaggio utente\n{original_request}"

        logger.debug(
            "Manager: policy USER.md — analisi richiesta '%.80s'",
            original_request,
        )

        try:
            from src.core.llm.client import get_llm_client
        except ImportError:
            from core.llm.client import get_llm_client  # type: ignore[no-redef]

        llm = get_llm_client()
        try:
            policy_result = await llm.complete(
                system_prompt, user_message, temperature=0.0, max_tokens=1024
            )
        except Exception as exc:
            logger.warning("Manager: policy USER.md — errore LLM: %s", exc)
            return

        if policy_result.get("answer") is not True:
            return

        update_text = str(policy_result.get("update_with", "")).strip()
        if not update_text:
            return

        try:
            from src.subagents.storage_manager import StorageManager
        except ImportError:
            from subagents.storage_manager import StorageManager  # type: ignore[no-redef]

        storage = StorageManager()
        update_result = await storage.update("USER.md", update_text)
        if update_result["success"]:
            logger.info("Manager: USER.md aggiornato — '%s'", update_text[:100])
        else:
            logger.warning(
                "Manager: aggiornamento USER.md fallito — %s", update_result.get("output")
            )


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")
    os.environ.setdefault("VAULT_PATH", str(Path(__file__).resolve().parent.parent.parent / "vault"))

    passed = 0
    failed = 0

    def check(label: str, condition: bool) -> None:
        global passed, failed
        if condition:
            print(f"[OK] {label}")
            passed += 1
        else:
            print(f"[FAIL] {label}")
            failed += 1

    # ------------------------------------------------------------------
    # Test _resolve_groups
    # ------------------------------------------------------------------

    print("\n=== _resolve_groups ===\n")

    check(
        "piano vuoto -> []",
        _resolve_groups({}) == [],
    )
    check(
        "solo RETRIEVE -> [[RETRIEVE]]",
        _resolve_groups({"RETRIEVE": {}}) == [["RETRIEVE"]],
    )
    check(
        "RETRIEVE + ANALYZE -> stesso gruppo",
        _resolve_groups({"RETRIEVE": {}, "ANALYZE": {}}) == [["RETRIEVE", "ANALYZE"]],
    )
    check(
        "RETRIEVE + GENERATE -> 2 gruppi",
        _resolve_groups({"RETRIEVE": {}, "GENERATE": {}}) == [["RETRIEVE"], ["GENERATE"]],
    )
    check(
        "GENERATE senza RETRIEVE/ANALYZE -> gruppo 0",
        _resolve_groups({"GENERATE": {}}) == [["GENERATE"]],
    )
    g = _resolve_groups({"RETRIEVE": {}, "GENERATE": {}, "STORE": {}})
    check(
        "RETRIEVE + GENERATE + STORE -> 3 gruppi distinti",
        len(g) == 3 and "RETRIEVE" in g[0] and "GENERATE" in g[1] and "STORE" in g[2],
    )
    g2 = _resolve_groups({"RETRIEVE": {}, "STORE": {}})
    check(
        "RETRIEVE + STORE (senza GENERATE) -> 2 gruppi",
        len(g2) == 2 and "RETRIEVE" in g2[0] and "STORE" in g2[1],
    )
    check(
        "solo STORE -> gruppo 0 (nessun prerequisito)",
        _resolve_groups({"STORE": {}}) == [["STORE"]],
    )
    g3 = _resolve_groups({"RETRIEVE": {}, "ANALYZE": {}, "REASON": {}, "GENERATE": {}})
    check(
        "RETRIEVE+ANALYZE+REASON in gruppo 0, GENERATE in gruppo 1",
        len(g3) == 2
        and set(g3[0]) == {"RETRIEVE", "ANALYZE", "REASON"}
        and g3[1] == ["GENERATE"],
    )

    # ------------------------------------------------------------------
    # Test Manager.execute con subagents mockati
    # ------------------------------------------------------------------

    print("\n=== Manager.execute ===\n")

    async def _run_manager_tests() -> None:
        # Inietta stub per dipendenze di sistema non installate in ambiente di test
        _stub_pkgs = []
        for pkg in ("chromadb", "sentence_transformers", "bs4", "httpx"):
            if pkg not in sys.modules:
                stub = MagicMock()
                # beautifulsoup usa bs4.BeautifulSoup come attributo
                if pkg == "bs4":
                    stub.BeautifulSoup = MagicMock()
                sys.modules[pkg] = stub
                _stub_pkgs.append(pkg)

        # Pre-importa i moduli cosicché patch() li trovi per nome
        import src.tools.vault_search
        import src.tools.web_search
        import src.tools.read_document
        import src.tools.write_document
        import src.subagents.researcher
        import src.subagents.analyzer
        import src.subagents.generator
        import src.subagents.storage_manager
        from src.core.context import new_context

        manager = Manager()
        statuses: list[str] = []

        async def callback(msg: str) -> None:
            statuses.append(msg)

        # Mock Researcher
        mock_researcher = MagicMock()
        mock_researcher.run = AsyncMock(return_value={"success": True, "output": "dati trovati"})

        # Mock Generator
        mock_generator = MagicMock()
        mock_generator.run = AsyncMock(return_value={"success": True, "output": "risposta generata"})

        # Mock vault_search
        mock_vault_results = [{"content": "contenuto", "path": "wiki/test.md", "score": 0.9}]

        captured_researcher_ctx: list[dict] = []
        captured_generator_ctx: list[dict] = []

        async def _mock_researcher_run(task, context, tools):
            captured_researcher_ctx.append(context)
            return {"success": True, "output": "dati trovati"}

        async def _mock_generator_run(task, context, tools):
            captured_generator_ctx.append(context)
            return {"success": True, "output": "risposta generata"}

        mock_researcher.run = _mock_researcher_run
        mock_generator.run = _mock_generator_run

        with (
            patch("src.subagents.researcher.Researcher", return_value=mock_researcher),
            patch("src.subagents.generator.Generator", return_value=mock_generator),
            patch("src.tools.vault_search.vault_search", new=AsyncMock(return_value=mock_vault_results)),
        ):
            ctx = new_context()
            ctx.results["request"] = "test query"
            result_ctx = await manager.execute(
                plan={
                    "RETRIEVE": {"query": "test query", "source": "vault"},
                    "GENERATE": {"format": "summary"},
                },
                context=ctx,
                status_callback=callback,
            )

        check("execute() restituisce il context", result_ctx is ctx)
        check("RETRIEVE in actions_executed", "RETRIEVE" in ctx.actions_executed)
        check("GENERATE in actions_executed", "GENERATE" in ctx.actions_executed)
        check("RETRIEVE result registrato", ctx.results.get("RETRIEVE", {}).get("success") is True)
        check("GENERATE result registrato", ctx.results.get("GENERATE", {}).get("output") == "risposta generata")
        check("vault_search registrata in tools_called", any(t["tool"] == "vault_search" for t in ctx.tools_called))
        check("vault_search duration_ms >= 0", all(t["duration_ms"] >= 0 for t in ctx.tools_called))
        check("status_callback chiamato almeno una volta", len(statuses) > 0)
        check("nessun errore per run riuscito", len(ctx.errors) == 0)
        # Verifica che le chiavi corrette arrivino ai subagent
        check("Researcher riceve 'request'", "request" in captured_researcher_ctx[0])
        check("Researcher riceve 'tool_results'", "tool_results" in captured_researcher_ctx[0])
        check("Researcher NON riceve 'user_request'", "user_request" not in captured_researcher_ctx[0])
        check("Generator riceve 'request'", "request" in captured_generator_ctx[0])
        check("Generator riceve 'tool_results' con retrieve", "retrieve" in captured_generator_ctx[0].get("tool_results", {}))

        # Test: subagent fallisce → errore non bloccante
        mock_failing = MagicMock()
        mock_failing.run = AsyncMock(return_value={"success": False, "output": "LLM offline"})

        with (
            patch("src.subagents.analyzer.Analyzer", return_value=mock_failing),
            patch("src.tools.read_document.read_document", new=AsyncMock(return_value="# Doc\nContenuto.")),
        ):
            ctx2 = new_context()
            await manager.execute(
                plan={"ANALYZE": {"path": "wiki/test.md"}},
                context=ctx2,
                status_callback=callback,
            )

        check("errore subagent registrato in context.errors", len(ctx2.errors) > 0)
        check("ANALYZE comunque in actions_executed (fail non bloccante)", "ANALYZE" in ctx2.actions_executed)

        # Test: synthesis — vault_search riuscita + GENERATE riuscito → nota salvata
        saved_synthesis: list[tuple[str, str]] = []

        async def _mock_write_doc(path: str, content: str) -> bool:
            saved_synthesis.append((path, content))
            return True

        mock_researcher2 = MagicMock()
        mock_researcher2.run = AsyncMock(return_value={"success": True, "output": "ricerca completata"})
        mock_generator2 = MagicMock()
        mock_generator2.run = AsyncMock(return_value={"success": True, "output": "risposta generata"})

        with (
            patch("src.subagents.researcher.Researcher", return_value=mock_researcher2),
            patch("src.subagents.generator.Generator", return_value=mock_generator2),
            patch("src.tools.vault_search.vault_search", new=AsyncMock(return_value=mock_vault_results)),
            patch("src.tools.write_document.write_document", new=_mock_write_doc),
        ):
            ctx_synth = new_context()
            ctx_synth.results["request"] = "cosa sai sui transformer?"
            await manager.execute(
                plan={
                    "RETRIEVE": {"query": "transformer", "source": "vault"},
                    "GENERATE": {"format": "risposta"},
                },
                context=ctx_synth,
                status_callback=callback,
            )

        check("synthesis: write_document chiamato", len(saved_synthesis) == 1)
        check("synthesis: path in synthesis/",
              bool(saved_synthesis) and saved_synthesis[0][0].startswith("synthesis/"))
        check("synthesis: contenuto ha sezione Fonti",
              bool(saved_synthesis) and "## Fonti" in saved_synthesis[0][1])
        check("synthesis: contenuto ha sezione Risposta",
              bool(saved_synthesis) and "## Risposta" in saved_synthesis[0][1])
        check("synthesis: tag sintesi presente",
              bool(saved_synthesis) and "  - sintesi" in saved_synthesis[0][1])

        # Test: synthesis NON scatta senza vault_search (solo web)
        saved_synthesis_web: list[tuple[str, str]] = []

        async def _mock_write_doc_web(path: str, content: str) -> bool:
            saved_synthesis_web.append((path, content))
            return True

        with (
            patch("src.subagents.researcher.Researcher", return_value=mock_researcher2),
            patch("src.subagents.generator.Generator", return_value=mock_generator2),
            patch("src.tools.web_search.async_web_search", new=AsyncMock(return_value=[{"title": "t", "url": "u", "snippet": "s"}])),
            patch("src.tools.write_document.write_document", new=_mock_write_doc_web),
        ):
            ctx_web = new_context()
            ctx_web.results["request"] = "ultime notizie AI"
            await manager.execute(
                plan={
                    "RETRIEVE": {"query": "ultime notizie AI", "source": "web"},
                    "GENERATE": {"format": "risposta"},
                },
                context=ctx_web,
                status_callback=callback,
            )

        check("synthesis: NON scatta per retrieve solo web", len(saved_synthesis_web) == 0)

        # Test: synthesis documents_modified popolato dopo auto-save vault
        with (
            patch("src.subagents.researcher.Researcher", return_value=mock_researcher2),
            patch("src.subagents.generator.Generator", return_value=mock_generator2),
            patch("src.tools.vault_search.vault_search", new=AsyncMock(return_value=mock_vault_results)),
            patch("src.tools.write_document.write_document", new=_mock_write_doc),
        ):
            saved_synthesis.clear()
            ctx_mod = new_context()
            ctx_mod.results["request"] = "test documents_modified"
            await manager.execute(
                plan={
                    "RETRIEVE": {"query": "transformer", "source": "vault"},
                    "GENERATE": {"format": "risposta"},
                },
                context=ctx_mod,
                status_callback=callback,
            )
        check("synthesis: path aggiunto a documents_modified",
              any(p.startswith("synthesis/") for p in ctx_mod.documents_modified))

        # Test: save_web_synthesis salva nota con URLs
        saved_web_synth: list[tuple[str, str]] = []

        async def _mock_write_web(path: str, content: str) -> bool:
            saved_web_synth.append((path, content))
            return True

        from src.core.context import new_context as _nc
        ctx_ws = _nc()
        ctx_ws.results["request"] = "ultime notizie AI"
        ctx_ws.results["GENERATE"] = {"success": True, "output": "risposta generata da web"}
        ctx_ws.tools_called = [{
            "tool": "web_search",
            "input": {"query": "ultime notizie AI"},
            "output": [
                {"url": "https://example.com/1", "content": "..."},
                {"url": "https://example.com/2", "content": "..."},
            ],
            "duration_ms": 100,
        }]

        with patch("src.tools.write_document.write_document", new=_mock_write_web):
            mgr_ws = Manager()
            result_path = await mgr_ws.save_web_synthesis(ctx_ws)

        check("save_web_synthesis: restituisce path relativo", result_path is not None and result_path.startswith("synthesis/"))
        check("save_web_synthesis: write_document chiamato", len(saved_web_synth) == 1)
        check("save_web_synthesis: contenuto ha Fonti",
              bool(saved_web_synth) and "## Fonti" in saved_web_synth[0][1])
        check("save_web_synthesis: URL incluso nelle fonti",
              bool(saved_web_synth) and "https://example.com/1" in saved_web_synth[0][1])
        check("save_web_synthesis: tag web presente",
              bool(saved_web_synth) and "- web" in saved_web_synth[0][1])

        # Test: save_web_synthesis → None senza web_search
        ctx_novault = _nc()
        ctx_novault.results["request"] = "test"
        ctx_novault.results["GENERATE"] = {"success": True, "output": "risposta"}
        ctx_novault.tools_called = []
        mgr_nw = Manager()
        result_none = await mgr_nw.save_web_synthesis(ctx_novault)
        check("save_web_synthesis: None senza web_search", result_none is None)

        # Test: save_web_synthesis → None senza GENERATE success
        ctx_nogen = _nc()
        ctx_nogen.results["request"] = "test"
        ctx_nogen.results["GENERATE"] = {"success": False, "output": "errore"}
        ctx_nogen.tools_called = [{"tool": "web_search", "input": {}, "output": [{"url": "x"}]}]
        result_nogen = await mgr_nw.save_web_synthesis(ctx_nogen)
        check("save_web_synthesis: None senza GENERATE success", result_nogen is None)

        # Test: STORE salta se GENERATE fallisce
        mock_gen_fail = MagicMock()
        mock_gen_fail.run = AsyncMock(return_value={"success": False, "output": "errore"})

        with patch("src.subagents.generator.Generator", return_value=mock_gen_fail):
            ctx3 = new_context()
            await manager.execute(
                plan={
                    "GENERATE": {"format": "summary"},
                    "STORE": {"path": "raw/out.md"},
                },
                context=ctx3,
                status_callback=callback,
            )

        check("STORE saltato se GENERATE fallisce", "STORE" not in ctx3.actions_executed)
        check("errore STORE registrato", any(e["step"] == "STORE" for e in ctx3.errors))

    asyncio.run(_run_manager_tests())

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
