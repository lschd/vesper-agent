"""
AbstractInterface — contratto comune per tutte le interfacce di accesso a Vesper.

Ogni interfaccia (Telegram, Web, …) implementa questa classe e decide:
- Quali azioni può pianificare l'Orchestrator (get_permissions)
- Come recapitare messaggi e errori al destinatario (send, send_error)
- Se e come modificare il piano post-Orchestrator (_post_plan_hook)

La pipeline core (session_memory → Orchestrator → Manager → session_memory.store)
è fornita dal metodo concreto process_request(), che le sottoclassi invocano
passando history, session_id e status_callback specifici della propria piattaforma.

Utility pubblica:
    resolve_target(output_target) -> (scheme, target_id)
"""
import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractInterface(ABC):

    @abstractmethod
    async def send(self, target: str, message: str) -> None:
        """
        Invia un messaggio al destinatario.

        Args:
            target:  Identificatore del destinatario (es. chat_id Telegram come stringa).
            message: Testo del messaggio da inviare.
        """

    @abstractmethod
    async def send_error(self, target: str, task_name: str, reason: str) -> None:
        """
        Invia un messaggio di errore strutturato.

        Args:
            target:    Identificatore del destinatario.
            task_name: Nome del task che ha fallito (per log e debug utente).
            reason:    Motivo del fallimento in linguaggio naturale.
        """

    @abstractmethod
    def get_permissions(self) -> list[str]:
        """
        Restituisce la lista di azioni disponibili per questa interfaccia.

        Usata dall'Orchestrator per sapere cosa può pianificare in base
        all'interfaccia di origine della richiesta.

        Returns:
            Lista di nomi azione (sottoinsieme di RETRIEVE, ANALYZE, REASON, GENERATE, STORE).
        """

    # ------------------------------------------------------------------
    # Utility interfacce
    # ------------------------------------------------------------------

    @staticmethod
    def _web_synthesis_available(req_ctx) -> bool:
        """
        True se la risposta corrente è basata su una ricerca web e non ha ancora una sintesi.

        Usato dalle interfacce per decidere se mostrare il pulsante "Salva sintesi".
        Condizioni: GENERATE success + almeno un web_search con output + nessun path
        synthesis/ in documents_modified (la sintesi vault auto-save non era disponibile).
        """
        generate_res = req_ctx.results.get("GENERATE", {})
        if not (isinstance(generate_res, dict) and generate_res.get("success")):
            return False

        has_web = any(
            tc.get("tool") == "web_search" and tc.get("output")
            for tc in req_ctx.tools_called
        )
        if not has_web:
            return False

        already_synthesized = any(
            isinstance(p, str) and p.startswith("synthesis/")
            for p in getattr(req_ctx, "documents_modified", [])
        )
        return not already_synthesized

    # ------------------------------------------------------------------
    # Pipeline condivisa
    # ------------------------------------------------------------------

    def _post_plan_hook(self, plan: dict) -> dict:
        """
        Hook per modifiche al piano dopo l'Orchestrator, prima del Manager.

        Default: identità (nessuna modifica). Le sottoclassi possono fare
        override per applicare restrizioni o trasformazioni platform-specific
        (es. WebInterface limita vault_search alle sezioni permesse).

        Args:
            plan: Piano prodotto dall'Orchestrator {"AZIONE": {...}}.

        Returns:
            Piano (eventualmente modificato) da passare al Manager.
        """
        return plan

    async def process_request(
        self,
        user_text: str,
        history: list[dict],
        session_id: str,
        status_callback,
        *,
        effective_user_text: str | None = None,
        extra_context: dict | None = None,
    ) -> tuple[str, object, dict]:
        """
        Pipeline completa: session_memory.query → Orchestrator → Manager → session_memory.store.

        Gestisce interamente il ciclo di vita di una richiesta. Le sottoclassi
        si occupano solo di estrarre input e formattare output specifici della
        propria piattaforma.

        Args:
            user_text:          Testo originale dell'utente (salvato in req_ctx e history).
            history:            Cronologia messaggi precedenti [{"role": str, "content": str}].
            session_id:         Identificatore univoco di sessione/utente.
            status_callback:    Callable async(str) per aggiornamenti durante l'elaborazione.
            effective_user_text: Testo da passare all'Orchestrator se diverso da user_text
                                (es. con snippet di documento allegato inline).
                                Se None, usa user_text.
            extra_context:      Campi aggiuntivi da iniettare in req_ctx.results prima
                                dell'elaborazione (es. {"uploaded_documents": [...]} per
                                Telegram, {"document_content": "..."} per Web).

        Returns:
            Tuple (response_text, req_ctx, stats):
            - response_text: output grezzo del Generator (stringa vuota se GENERATE fallisce)
            - req_ctx:       RequestContext completato (tools_called, results, errors)
            - stats:         Dict statistiche LLM {completion_tokens, elapsed_sec,
                             tokens_per_sec} o dict vuoto se non disponibili
        """
        try:
            from src.core.orchestrator import Orchestrator
            from src.core.manager import Manager
            from src.core.context import new_context
            from src.core.llm.client import get_llm_client
        except ImportError:
            from core.orchestrator import Orchestrator      # type: ignore[no-redef]
            from core.manager import Manager                # type: ignore[no-redef]
            from core.context import new_context            # type: ignore[no-redef]
            from core.llm.client import get_llm_client      # type: ignore[no-redef]

        req_ctx = new_context()
        req_ctx.results["request"] = user_text
        if extra_context:
            req_ctx.results.update(extra_context)

        # Query session memory per contesto da turni precedenti
        context_memory: list[dict] = []
        try:
            from src.core.session_memory import query as _mem_query
        except ImportError:
            try:
                from core.session_memory import query as _mem_query  # type: ignore[no-redef]
            except ImportError:
                _mem_query = None  # type: ignore[assignment]

        if _mem_query is not None:
            try:
                context_memory = await _mem_query(session_id, user_text, top_k=3)
            except Exception as exc:
                logger.warning(
                    "AbstractInterface.process_request: session_memory query fallita — %s", exc
                )

        llm = get_llm_client()

        # Pre-search vault — segnale per l'Orchestrator, nessun LLM aggiuntivo
        vault_hint: str | None = None
        try:
            from src.tools.vault_search import vault_search as _vs
        except ImportError:
            try:
                from tools.vault_search import vault_search as _vs  # type: ignore[no-redef]
            except ImportError:
                _vs = None  # type: ignore[assignment]

        if _vs is not None:
            try:
                _pre = await _vs(
                    effective_user_text or user_text,
                    top_k=3,
                    rerank=False,
                    follow_links=False,
                )
                _relevant = [r for r in _pre if r.get("score", 0) >= 0.6]
                if _relevant:
                    top_score = max(r["score"] for r in _relevant)
                    vault_hint = (
                        f"Trovati {len(_relevant)} document"
                        f"{'o' if len(_relevant) == 1 else 'i'} rilevant"
                        f"{'e' if len(_relevant) == 1 else 'i'} nel vault "
                        f"(score massimo: {top_score:.2f})."
                    )
            except Exception as exc:
                logger.debug("AbstractInterface: vault pre-search fallita — %s", exc)

        # Helper locale per aggiornare la TUI senza import circolare a livello modulo.
        # tui.py importa AbstractInterface da questo file, quindi non possiamo importare
        # tui a livello modulo — usiamo un import deferred dentro la funzione.
        def _tui_set(state: str) -> None:
            try:
                from src.interfaces.tui import set_tui_state as _f
                _f(state)
            except (ImportError, AttributeError):
                try:
                    from interfaces.tui import set_tui_state as _f  # type: ignore[no-redef]
                    _f(state)
                except (ImportError, AttributeError):
                    pass

        _tui_set("processing")
        try:
            orch = Orchestrator()
            plan = await orch.plan(
                user_message=effective_user_text or user_text,
                available_actions=self.get_permissions(),
                context=req_ctx,
                conversation_history=history,
                context_memory=context_memory or None,
                vault_hint=vault_hint,
            )
            llm.consume_stats()  # scarica le stats dell'Orchestrator (non mostrate all'utente)

            plan = self._post_plan_hook(plan)

            mgr = Manager()
            req_ctx = await mgr.execute(
                plan=plan,
                context=req_ctx,
                status_callback=status_callback,
                context_memory=context_memory or None,
                conversation_history=history or None,
            )

            generate_result = req_ctx.results.get("GENERATE", {})
            if isinstance(generate_result, dict) and generate_result.get("success"):
                response_text = str(generate_result.get("output", ""))
            else:
                response_text = ""

            stats = llm.consume_stats()

            # Salva capsule di sessione in background — fire-and-forget, nessuna latenza aggiunta
            try:
                from src.core.session_memory import store as _mem_store
            except ImportError:
                try:
                    from core.session_memory import store as _mem_store  # type: ignore[no-redef]
                except ImportError:
                    _mem_store = None  # type: ignore[assignment]

            if _mem_store is not None:
                asyncio.create_task(_mem_store(session_id, req_ctx))

            return response_text, req_ctx, stats
        finally:
            _tui_set("idle")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def resolve_target(output_target: str) -> tuple[str, str]:
    """
    Parsa una stringa output_target e restituisce (scheme, target_id).

    Formato atteso: "scheme://target_id"
    Esempi:
        "telegram://123456789"  →  ("telegram", "123456789")
        "email://user@example.com"  →  ("email", "user@example.com")

    Args:
        output_target: Stringa tipizzata nel formato "scheme://target_id".

    Returns:
        Tuple (scheme, target_id) con scheme in lowercase.

    Raises:
        ValueError: se il formato non è riconosciuto, lo scheme è vuoto
                    o il target_id è vuoto.
    """
    if "://" not in output_target:
        raise ValueError(
            f"Formato output_target non riconosciuto: {output_target!r}. "
            "Atteso: 'scheme://target_id' (es. 'telegram://123456789')."
        )

    scheme, _, target_id = output_target.partition("://")
    scheme = scheme.strip().lower()
    target_id = target_id.strip()

    if not scheme:
        raise ValueError(
            f"Scheme vuoto in output_target: {output_target!r}."
        )

    if not target_id:
        raise ValueError(
            f"target_id vuoto in output_target: {output_target!r}."
        )

    return scheme, target_id


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

    def expect_raises(label: str, exc_type: type, fn, *args) -> None:
        global passed, failed
        try:
            fn(*args)
            print(f"[FAIL] {label} - nessuna eccezione sollevata")
            failed += 1
        except exc_type:
            print(f"[OK] {label}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {label} - eccezione sbagliata: {type(e).__name__}: {e}")
            failed += 1

    print("\n=== resolve_target ===\n")

    scheme, target_id = resolve_target("telegram://123456789")
    check("telegram: scheme corretto", scheme == "telegram")
    check("telegram: target_id corretto", target_id == "123456789")

    scheme2, target2 = resolve_target("email://user@example.com")
    check("email: scheme corretto", scheme2 == "email")
    check("email: target_id corretto", target2 == "user@example.com")

    scheme3, _ = resolve_target("TELEGRAM://999")
    check("scheme convertito in lowercase", scheme3 == "telegram")

    expect_raises("ValueError senza ://", ValueError, resolve_target, "telegram-123456")
    expect_raises("ValueError con scheme vuoto", ValueError, resolve_target, "://123456")
    expect_raises("ValueError con target_id vuoto", ValueError, resolve_target, "telegram://")
    expect_raises("ValueError con stringa vuota", ValueError, resolve_target, "")

    print("\n=== _web_synthesis_available ===\n")

    class _FakeCtx:
        def __init__(self, generate_success, tools_called, documents_modified=None):
            self.results = {"GENERATE": {"success": generate_success, "output": "x"}}
            self.tools_called = tools_called
            self.documents_modified = documents_modified or []

    class _CheckIface(AbstractInterface):
        async def send(self, t, m): pass
        async def send_error(self, t, n, r): pass
        def get_permissions(self): return []

    _iface_check = _CheckIface()

    _web_tc = [{"tool": "web_search", "input": {"query": "x"}, "output": [{"url": "u"}]}]
    _vault_tc = [{"tool": "vault_search", "input": {}, "output": [{"path": "wiki/a.md"}]}]

    check("web+GENERATE OK: True",
          _iface_check._web_synthesis_available(_FakeCtx(True, _web_tc)))
    check("GENERATE fail: False",
          not _iface_check._web_synthesis_available(_FakeCtx(False, _web_tc)))
    check("nessun web_search: False",
          not _iface_check._web_synthesis_available(_FakeCtx(True, _vault_tc)))
    check("web_search senza output: False",
          not _iface_check._web_synthesis_available(
              _FakeCtx(True, [{"tool": "web_search", "input": {}, "output": []}])))
    check("gia sintetizzato (synthesis/ in documents_modified): False",
          not _iface_check._web_synthesis_available(
              _FakeCtx(True, _web_tc, ["synthesis/2026-01-01-test.md"])))
    check("synthesis/ non in documents_modified: True",
          _iface_check._web_synthesis_available(
              _FakeCtx(True, _web_tc, ["raw/doc.md"])))

    print("\n=== process_request ===\n")

    # Implementazione concreta minima per i test
    class _TestInterface(AbstractInterface):
        async def send(self, target, message): pass
        async def send_error(self, target, task_name, reason): pass
        def get_permissions(self): return ["RETRIEVE", "GENERATE"]

    async def _run_process_tests() -> None:
        import src.core.orchestrator
        import src.core.manager
        import src.core.context
        import src.core.llm.client
        import src.core.session_memory

        iface = _TestInterface()

        mock_llm = MagicMock()
        mock_llm.consume_stats.return_value = {
            "completion_tokens": 100, "elapsed_sec": 1.5, "tokens_per_sec": 66.7
        }

        mock_ctx = MagicMock()
        mock_ctx.results = {
            "request": "test",
            "GENERATE": {"success": True, "output": "risposta generata"},
        }
        mock_ctx.tools_called = []
        mock_ctx.actions_executed = ["GENERATE"]
        mock_ctx.request_id = "req-test"

        mock_orch = MagicMock()
        mock_orch.plan = AsyncMock(return_value={"GENERATE": {"format": "summary"}})

        mock_mgr = MagicMock()
        mock_mgr.execute = AsyncMock(return_value=mock_ctx)

        mock_mem_store = AsyncMock()
        mock_mem_query = AsyncMock(return_value=[])

        statuses: list[str] = []
        async def status_cb(text: str) -> None:
            statuses.append(text)

        with (
            patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm),
            patch.object(src.core.orchestrator, "Orchestrator", return_value=mock_orch),
            patch.object(src.core.manager, "Manager", return_value=mock_mgr),
            patch.object(src.core.session_memory, "query", mock_mem_query),
            patch.object(src.core.session_memory, "store", mock_mem_store),
        ):
            resp, ctx, stats = await iface.process_request(
                user_text="test query",
                history=[{"role": "user", "content": "ciao"}],
                session_id="sess-1",
                status_callback=status_cb,
            )
            # lascia girare il task fire-and-forget
            await asyncio.sleep(0)

        check("process_request: response_text corretto", resp == "risposta generata")
        check("process_request: req_ctx restituito", ctx is mock_ctx)
        check("process_request: stats restituiti", stats["completion_tokens"] == 100)
        check("process_request: Orchestrator.plan chiamato", mock_orch.plan.called)
        check("process_request: Manager.execute chiamato", mock_mgr.execute.called)
        check("process_request: session_memory.query chiamato", mock_mem_query.called)
        check("process_request: session_memory.store schedulato", mock_mem_store.called)
        check("process_request: consume_stats chiamato 2 volte",
              mock_llm.consume_stats.call_count == 2)

        # Test: extra_context propagato in req_ctx
        captured_contexts: list = []
        async def _capture_execute(plan, context, status_callback, **kw):
            captured_contexts.append(dict(context.results))
            return mock_ctx

        mock_mgr.execute = _capture_execute

        with (
            patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm),
            patch.object(src.core.orchestrator, "Orchestrator", return_value=mock_orch),
            patch.object(src.core.manager, "Manager", return_value=mock_mgr),
            patch.object(src.core.session_memory, "query", mock_mem_query),
            patch.object(src.core.session_memory, "store", mock_mem_store),
        ):
            mock_llm.consume_stats.reset_mock()
            await iface.process_request(
                user_text="upload test",
                history=[],
                session_id="sess-2",
                status_callback=status_cb,
                extra_context={"uploaded_documents": ["/vault/raw/doc.pdf"]},
            )
            await asyncio.sleep(0)

        check("extra_context: uploaded_documents in req_ctx.results",
              captured_contexts and "/vault/raw/doc.pdf" in
              captured_contexts[-1].get("uploaded_documents", []))

        # Test: effective_user_text passato all'Orchestrator
        captured_orch_calls: list[str] = []
        async def _capture_plan(user_message, **kw):
            captured_orch_calls.append(user_message)
            return {"GENERATE": {}}

        mock_orch.plan = _capture_plan
        mock_mgr.execute = AsyncMock(return_value=mock_ctx)

        with (
            patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm),
            patch.object(src.core.orchestrator, "Orchestrator", return_value=mock_orch),
            patch.object(src.core.manager, "Manager", return_value=mock_mgr),
            patch.object(src.core.session_memory, "query", mock_mem_query),
            patch.object(src.core.session_memory, "store", mock_mem_store),
        ):
            mock_llm.consume_stats.reset_mock()
            await iface.process_request(
                user_text="testo originale",
                history=[],
                session_id="sess-3",
                status_callback=status_cb,
                effective_user_text="testo con documento allegato",
            )
            await asyncio.sleep(0)

        check("effective_user_text: Orchestrator riceve testo modificato",
              captured_orch_calls and captured_orch_calls[-1] == "testo con documento allegato")

        # Test: _post_plan_hook viene chiamato sul piano
        hook_calls: list[dict] = []

        class _HookInterface(_TestInterface):
            def _post_plan_hook(self, plan: dict) -> dict:
                hook_calls.append(plan)
                return {**plan, "INJECTED": {}}

        hook_iface = _HookInterface()
        mock_orch.plan = AsyncMock(return_value={"GENERATE": {"format": "test"}})
        mock_mgr.execute = AsyncMock(return_value=mock_ctx)

        with (
            patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm),
            patch.object(src.core.orchestrator, "Orchestrator", return_value=mock_orch),
            patch.object(src.core.manager, "Manager", return_value=mock_mgr),
            patch.object(src.core.session_memory, "query", mock_mem_query),
            patch.object(src.core.session_memory, "store", mock_mem_store),
        ):
            mock_llm.consume_stats.reset_mock()
            await hook_iface.process_request(
                user_text="hook test",
                history=[],
                session_id="sess-4",
                status_callback=status_cb,
            )
            await asyncio.sleep(0)

        check("_post_plan_hook: chiamato con il piano originale", len(hook_calls) == 1)
        # Manager deve ricevere il piano modificato dall'hook
        last_execute_call = mock_mgr.execute.call_args
        plan_passed = last_execute_call[1].get("plan") or (last_execute_call[0][0] if last_execute_call[0] else None)
        check("_post_plan_hook: Manager riceve piano modificato",
              plan_passed is not None and "INJECTED" in plan_passed)

        # Test: GENERATE fallisce → response_text vuoto, stats ancora restituiti
        mock_ctx_fail = MagicMock()
        mock_ctx_fail.results = {
            "request": "test fail",
            "GENERATE": {"success": False, "output": "errore"},
        }
        mock_ctx_fail.tools_called = []
        mock_ctx_fail.actions_executed = []
        mock_ctx_fail.request_id = "req-fail"

        mock_mgr.execute = AsyncMock(return_value=mock_ctx_fail)
        mock_orch.plan = AsyncMock(return_value={"GENERATE": {}})

        with (
            patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm),
            patch.object(src.core.orchestrator, "Orchestrator", return_value=mock_orch),
            patch.object(src.core.manager, "Manager", return_value=mock_mgr),
            patch.object(src.core.session_memory, "query", mock_mem_query),
            patch.object(src.core.session_memory, "store", mock_mem_store),
        ):
            mock_llm.consume_stats.reset_mock()
            resp_fail, ctx_fail, stats_fail = await iface.process_request(
                user_text="test fail",
                history=[],
                session_id="sess-5",
                status_callback=status_cb,
            )
            await asyncio.sleep(0)

        check("GENERATE fallisce: response_text vuoto", resp_fail == "")
        check("GENERATE fallisce: stats ancora presenti", isinstance(stats_fail, dict))

    asyncio.run(_run_process_tests())

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
