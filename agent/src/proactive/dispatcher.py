"""
ProactiveDispatcher — esegue task pianificati bypassando l'Orchestrator a runtime.

Flusso:
    APScheduler -> ProactiveDispatcher
        ├── legge il task file dal vault (frontmatter YAML con le azioni)
        ├── costruisce il RequestContext (autonomous_mode=True)
        └── passa le azioni direttamente al Manager
              └── Manager -> Subagents -> output_target.send()

output_target è una stringa tipizzata (es. "telegram://{{admin_chat_id}}").
Il Dispatcher risolve l'interfaccia corretta e chiama sempre send() — non conosce
i dettagli della piattaforma.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers privati
# ---------------------------------------------------------------------------

def _parse_frontmatter(task_path: str) -> dict:
    """
    Legge un task file e restituisce il frontmatter YAML come dict.

    Il frontmatter deve essere delimitato da '---' all'inizio del file.

    Args:
        task_path: Path assoluto o relativo al file task nel vault.

    Returns:
        Dict con i campi del frontmatter (schedule/execute_at, output_target, actions…).

    Raises:
        ValueError: se il frontmatter è assente o non chiuso.
    """
    import yaml

    text = Path(task_path).read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(f"Frontmatter YAML mancante in {task_path!r}")

    close_pos = text.find("\n---", 3)
    if close_pos == -1:
        raise ValueError(f"Frontmatter non chiuso in {task_path!r}")

    yaml_text = text[3:close_pos].strip()
    return yaml.safe_load(yaml_text) or {}


def _get_interface(interface_type: str):
    """
    Istanzia l'interfaccia corretta in base allo scheme dell'output_target.

    Args:
        interface_type: Scheme dell'output_target in lowercase (es. "telegram", "web").

    Returns:
        Istanza di AbstractInterface concreta.

    Raises:
        ValueError: se il tipo non è supportato.
    """
    if interface_type == "telegram":
        try:
            from src.interfaces.telegram import TelegramInterface
        except ImportError:
            from interfaces.telegram import TelegramInterface  # type: ignore[no-redef]
        return TelegramInterface()

    if interface_type == "web":
        try:
            from src.interfaces.web import WebInterface
        except ImportError:
            from interfaces.web import WebInterface  # type: ignore[no-redef]
        return WebInterface()

    raise ValueError(f"Tipo di interfaccia non supportato: {interface_type!r}")


# ---------------------------------------------------------------------------
# ProactiveDispatcher
# ---------------------------------------------------------------------------

class ProactiveDispatcher:
    """Esegue task proattivi leggendo le azioni direttamente dal file YAML."""

    async def dispatch(self, task_path: str) -> None:
        """
        Esegue un task dal suo file path.

        Passi:
            1. Legge il task file con read_document.
            2. Parsa il frontmatter YAML.
            3. Estrae output_target, actions, autonomous_mode=True.
            4. Crea un RequestContext con new_context(autonomous_mode=True).
            5. Costruisce il piano nel formato del Manager: {"AZIONE": input_dict}.
            6. Risolve output_target -> (interface_type, target_id).
            7. Istanzia l'interfaccia appropriata.
            8. Crea status_callback che chiama interface.send(target_id, msg).
            9. Chiama Manager.execute(plan, context, status_callback, autonomous_mode=True).
            10. Invia la risposta finale con interface.send(target_id, risposta).

        Args:
            task_path: Path al file task nel vault (es. "vault/agenda/recurring/briefing.md").

        Raises:
            KeyError:   se il frontmatter manca di campi obbligatori.
            ValueError: se output_target non è nel formato "scheme://target_id".
            Exception:  se Manager o interface falliscono.
        """
        try:
            from src.tools.read_document import read_document
            from src.core.context import new_context
            from src.core.manager import Manager
            from src.interfaces.base import resolve_target
        except ImportError:
            from tools.read_document import read_document      # type: ignore[no-redef]
            from core.context import new_context               # type: ignore[no-redef]
            from core.manager import Manager                   # type: ignore[no-redef]
            from interfaces.base import resolve_target         # type: ignore[no-redef]

        # 1. Legge il task file
        await read_document(task_path)

        # 2-3. Parsa frontmatter ed estrae i campi
        frontmatter = _parse_frontmatter(task_path)
        output_target: str = frontmatter["output_target"]
        actions_list: list = frontmatter.get("actions", [])

        # 4. Crea RequestContext in modalità autonoma
        context = new_context(autonomous_mode=True)

        # 5. Costruisce il piano nel formato {"AZIONE": input_dict}
        plan = {item["action"]: item.get("input", {}) for item in actions_list}

        # 6. Risolve output_target
        interface_type, target_id = resolve_target(output_target)

        # 7. Istanzia l'interfaccia
        interface = _get_interface(interface_type)

        # 8. Status callback
        async def status_callback(message: str) -> None:
            try:
                await interface.send(target_id, message)
            except Exception as exc:
                logger.warning("ProactiveDispatcher: status_callback fallito — %s", exc)

        # 9. Esegue il piano tramite Manager
        manager = Manager()
        context = await manager.execute(
            plan, context, status_callback, autonomous_mode=True
        )

        # 10. Invia la risposta finale
        generate_result = context.results.get("GENERATE", {})
        if isinstance(generate_result, dict) and generate_result.get("success"):
            response = str(generate_result.get("output", ""))
        else:
            response = "Task completato."

        await interface.send(target_id, response)

    async def _dispatch_with_retry(self, task_path: str) -> None:
        """
        Esegue il task con retry silenzioso al primo fallimento.

        Se anche il secondo tentativo fallisce, invia un messaggio di errore
        strutturato al destinatario tramite interface.send_error().

        Args:
            task_path: Path al file task nel vault.
        """
        try:
            await self.dispatch(task_path)
        except Exception as e:
            logger.warning(
                "Task %s fallito al primo tentativo: %s. Retry...", task_path, e
            )
            try:
                await self.dispatch(task_path)
            except Exception as e2:
                logger.error("Task %s fallito anche al retry: %s", task_path, e2)

                try:
                    from src.interfaces.base import resolve_target
                except ImportError:
                    from interfaces.base import resolve_target  # type: ignore[no-redef]

                try:
                    frontmatter = _parse_frontmatter(task_path)
                    interface_type, target_id = resolve_target(frontmatter["output_target"])
                    interface = _get_interface(interface_type)
                    task_name = Path(task_path).stem
                    await interface.send_error(target_id, task_name, str(e2))
                except Exception as notify_exc:
                    logger.error(
                        "ProactiveDispatcher: impossibile notificare errore per %s — %s",
                        task_path, notify_exc,
                    )


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    import os
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # Stub dipendenze pesanti non installate in ambiente di test
    for _pkg in (
        "chromadb", "sentence_transformers",
        "telegram", "telegram.ext",
        "fastapi", "pydantic",
        "sse_starlette", "sse_starlette.sse",
        "uvicorn",
    ):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = MagicMock()

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

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

    # ------------------------------------------------------------------
    # Test _parse_frontmatter
    # ------------------------------------------------------------------

    print("\n=== _parse_frontmatter ===\n")

    _RECURRING_CONTENT = """\
---
schedule: "0 8 * * *"
output_target: "telegram://123456789"
actions:
  - action: RETRIEVE
    input:
      query: "ultime notizie tech"
      source: "web"
  - action: GENERATE
    input:
      format: "briefing mattutino"
---
Descrizione human-readable del task ricorrente.
"""

    _ONE_OFF_CONTENT = """\
---
execute_at: "2099-12-31T08:00"
output_target: "telegram://987654321"
actions:
  - action: GENERATE
    input:
      format: "promemoria"
---
Task one-off.
"""

    with tempfile.TemporaryDirectory() as _tmp:
        _rec = Path(_tmp) / "briefing.md"
        _rec.write_text(_RECURRING_CONTENT)

        _one = Path(_tmp) / "promemoria.md"
        _one.write_text(_ONE_OFF_CONTENT)

        _no_front = Path(_tmp) / "no-front.md"
        _no_front.write_text("Nessun frontmatter qui.")

        _unclosed = Path(_tmp) / "unclosed.md"
        _unclosed.write_text("---\nschedule: daily\nManca la chiusura")

        fm = _parse_frontmatter(str(_rec))
        check("recurring: schedule corretto", fm.get("schedule") == "0 8 * * *")
        check("recurring: output_target corretto", fm.get("output_target") == "telegram://123456789")
        check("recurring: 2 azioni", len(fm.get("actions", [])) == 2)
        check("recurring: prima azione e' RETRIEVE", fm["actions"][0]["action"] == "RETRIEVE")
        check("recurring: input RETRIEVE corretto", fm["actions"][0]["input"]["query"] == "ultime notizie tech")
        check("recurring: source RETRIEVE corretto", fm["actions"][0]["input"]["source"] == "web")
        check("recurring: seconda azione e' GENERATE", fm["actions"][1]["action"] == "GENERATE")

        fm2 = _parse_frontmatter(str(_one))
        check("one-off: execute_at corretto", fm2.get("execute_at") == "2099-12-31T08:00")
        check("one-off: output_target corretto", fm2.get("output_target") == "telegram://987654321")
        check("one-off: 1 azione", len(fm2.get("actions", [])) == 1)

        expect_raises("senza frontmatter -> ValueError", ValueError, _parse_frontmatter, str(_no_front))
        expect_raises("frontmatter non chiuso -> ValueError", ValueError, _parse_frontmatter, str(_unclosed))

    # ------------------------------------------------------------------
    # Test ProactiveDispatcher.dispatch()
    # ------------------------------------------------------------------

    print("\n=== ProactiveDispatcher.dispatch() ===\n")

    async def _test_dispatch() -> None:
        import src.tools.read_document
        import src.core.manager
        import src.core.context
        import src.interfaces.base

        dispatcher = ProactiveDispatcher()

        mock_interface = MagicMock()
        mock_interface.send = AsyncMock()
        mock_interface.send_error = AsyncMock()

        mock_context = MagicMock()
        mock_context.results = {"GENERATE": {"success": True, "output": "risposta task"}}

        mock_manager = MagicMock()
        mock_manager.execute = AsyncMock(return_value=mock_context)

        with tempfile.TemporaryDirectory() as _tmp2:
            task_file = Path(_tmp2) / "test-task.md"
            task_file.write_text(_RECURRING_CONTENT)

            with (
                patch("src.tools.read_document.read_document", new=AsyncMock(return_value="contenuto")),
                patch("src.core.manager.Manager", return_value=mock_manager),
                patch("__main__._get_interface", return_value=mock_interface),
            ):
                await dispatcher.dispatch(str(task_file))

        check("dispatch: Manager.execute chiamato", mock_manager.execute.called)
        call_kwargs = mock_manager.execute.call_args
        plan_arg = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("plan")
        check("dispatch: piano ha RETRIEVE", "RETRIEVE" in (plan_arg or {}))
        check("dispatch: piano ha GENERATE", "GENERATE" in (plan_arg or {}))
        check("dispatch: interface.send chiamato almeno una volta", mock_interface.send.called)
        # L'ultimo send deve contenere la risposta finale
        last_call_args = mock_interface.send.call_args_list[-1][0]
        check("dispatch: risposta finale inviata", last_call_args[1] == "risposta task")

    asyncio.run(_test_dispatch())

    # ------------------------------------------------------------------
    # Test _dispatch_with_retry
    # ------------------------------------------------------------------

    print("\n=== _dispatch_with_retry ===\n")

    async def _test_retry() -> None:
        dispatcher = ProactiveDispatcher()

        # Test: primo fallimento, secondo successo
        call_count = 0

        async def _dispatch_fail_once(path: str) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("errore simulato primo tentativo")

        dispatcher.dispatch = _dispatch_fail_once
        await dispatcher._dispatch_with_retry("fake/path.md")
        check("retry: dispatch chiamato 2 volte al primo fail", call_count == 2)

        # Test: entrambi i tentativi falliscono -> send_error
        mock_iface2 = MagicMock()
        mock_iface2.send_error = AsyncMock()

        async def _dispatch_always_fail(path: str) -> None:
            raise RuntimeError("errore simulato sempre")

        dispatcher.dispatch = _dispatch_always_fail

        with tempfile.TemporaryDirectory() as _tmp3:
            task_file2 = Path(_tmp3) / "failing-task.md"
            task_file2.write_text(_ONE_OFF_CONTENT)

            with patch("__main__._get_interface", return_value=mock_iface2):
                await dispatcher._dispatch_with_retry(str(task_file2))

        check("retry: send_error chiamato dopo doppio fail", mock_iface2.send_error.called)
        err_args = mock_iface2.send_error.call_args[0]
        check("retry: target_id corretto", err_args[0] == "987654321")
        check("retry: task_name e' il nome del file", err_args[1] == "failing-task")

    asyncio.run(_test_retry())

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
