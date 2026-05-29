"""
Scheduler — integrazione APScheduler per l'esecuzione periodica dei task.

Legge i task file da vault/agenda/recurring/ e li registra in APScheduler
usando il campo `schedule` del frontmatter YAML (formato cron standard).
I task one-off in vault/agenda/one-off/ vengono registrati con run_once()
solo se execute_at è ancora nel futuro.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class VesperScheduler:
    """
    Integra i task proattivi con il JobQueue di python-telegram-bot (APScheduler).

    Uso tipico:
        scheduler = VesperScheduler(application.job_queue)
        scheduler.load_tasks()
    """

    def __init__(self, job_queue) -> None:
        """
        Args:
            job_queue: JobQueue di python-telegram-bot che wrappa APScheduler.
        """
        self.job_queue = job_queue
        try:
            from src.proactive.dispatcher import ProactiveDispatcher
        except ImportError:
            from proactive.dispatcher import ProactiveDispatcher  # type: ignore[no-redef]
        self.dispatcher = ProactiveDispatcher()

    def load_tasks(self) -> None:
        """
        Scansiona vault/agenda/recurring/ e vault/agenda/one-off/ e registra i job.

        Per i recurring: usa job_queue.run_custom() con CronTrigger.
        Per i one-off: usa job_queue.run_once() solo se execute_at e' futuro.
        Logga ogni task registrato con nome file e schedule/execute_at.
        """
        try:
            from src.proactive.dispatcher import _parse_frontmatter
        except ImportError:
            from proactive.dispatcher import _parse_frontmatter  # type: ignore[no-redef]

        from apscheduler.triggers.cron import CronTrigger

        try:
            from config import Config
        except ImportError:
            from config import Config  # type: ignore[no-redef]

        vault_path = Path(Config.vault_path)
        agenda_recurring = vault_path / "agenda" / "recurring"
        agenda_one_off   = vault_path / "agenda" / "one-off"
        registered = 0

        # --- Ricorrenti ---
        if agenda_recurring.is_dir():
            for task_file in sorted(agenda_recurring.glob("*.md")):
                try:
                    frontmatter = _parse_frontmatter(str(task_file))
                    schedule = frontmatter.get("schedule")
                    if not schedule:
                        logger.warning(
                            "Scheduler: '%s' senza campo 'schedule', saltato",
                            task_file.name,
                        )
                        continue

                    task_path = str(task_file)

                    async def _recurring_cb(context, _p=task_path):
                        await self.dispatcher._dispatch_with_retry(_p)

                    self.job_queue.run_custom(
                        _recurring_cb,
                        job_kwargs={"trigger": CronTrigger.from_crontab(schedule)},
                    )
                    logger.info(
                        "Scheduler: registrato recurring '%s' (schedule=%s)",
                        task_file.name, schedule,
                    )
                    registered += 1

                except Exception as exc:
                    logger.error(
                        "Scheduler: errore caricamento '%s' — %s", task_file.name, exc
                    )

        # --- One-off ---
        if agenda_one_off.is_dir():
            for task_file in sorted(agenda_one_off.glob("*.md")):
                try:
                    frontmatter = _parse_frontmatter(str(task_file))
                    execute_at_raw = frontmatter.get("execute_at")
                    if not execute_at_raw:
                        logger.warning(
                            "Scheduler: '%s' senza campo 'execute_at', saltato",
                            task_file.name,
                        )
                        continue

                    execute_at = datetime.fromisoformat(str(execute_at_raw))
                    if execute_at.tzinfo is None:
                        execute_at = execute_at.replace(tzinfo=timezone.utc)

                    if execute_at <= datetime.now(timezone.utc):
                        logger.info(
                            "Scheduler: one-off '%s' gia' scaduto (execute_at=%s), saltato",
                            task_file.name, execute_at.isoformat(),
                        )
                        continue

                    task_path = str(task_file)

                    async def _one_off_cb(context, _p=task_path):
                        await self.dispatcher._dispatch_with_retry(_p)

                    self.job_queue.run_once(_one_off_cb, when=execute_at)
                    logger.info(
                        "Scheduler: registrato one-off '%s' (execute_at=%s)",
                        task_file.name, execute_at.isoformat(),
                    )
                    registered += 1

                except Exception as exc:
                    logger.error(
                        "Scheduler: errore caricamento '%s' — %s", task_file.name, exc
                    )

        logger.info("Scheduler: %d task registrati", registered)

    def reload_tasks(self) -> None:
        """
        Rimuove tutti i job esistenti e ricarica i task dal vault da zero.

        Utile quando l'utente modifica i task file senza riavviare il bot.
        """
        existing = self.job_queue.jobs()
        for job in existing:
            job.schedule_removal()
        logger.info("Scheduler: %d job rimossi", len(existing))
        self.load_tasks()


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # Stub dipendenze pesanti
    for _pkg in (
        "chromadb", "sentence_transformers",
        "telegram", "telegram.ext",
        "fastapi", "pydantic",
        "sse_starlette", "sse_starlette.sse",
        "uvicorn",
    ):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = MagicMock()

    # Stub apscheduler se non installato
    if "apscheduler" not in sys.modules:
        _aps = MagicMock()
        _cron_trigger = MagicMock()
        _cron_trigger.from_crontab = MagicMock(return_value="cron-trigger-mock")
        _aps.triggers.cron.CronTrigger = _cron_trigger
        sys.modules["apscheduler"] = _aps
        sys.modules["apscheduler.triggers"] = _aps.triggers
        sys.modules["apscheduler.triggers.cron"] = _aps.triggers.cron

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

    # ------------------------------------------------------------------
    # Setup: task file di esempio
    # ------------------------------------------------------------------

    _RECURRING_TASK = """\
---
schedule: "0 8 * * *"
output_target: "telegram://123456789"
actions:
  - action: RETRIEVE
    input:
      query: "notizie"
      source: "web"
  - action: GENERATE
    input:
      format: "briefing"
---
Briefing mattutino.
"""

    _FUTURE_ONE_OFF = """\
---
execute_at: "2099-06-01T09:00"
output_target: "telegram://123456789"
actions:
  - action: GENERATE
    input:
      format: "promemoria"
---
Promemoria futuro.
"""

    _PAST_ONE_OFF = """\
---
execute_at: "2000-01-01T00:00"
output_target: "telegram://123456789"
actions:
  - action: GENERATE
    input:
      format: "vecchio"
---
Task scaduto.
"""

    _NO_SCHEDULE = """\
---
output_target: "telegram://123456789"
actions:
  - action: GENERATE
    input:
      format: "test"
---
Task senza schedule.
"""

    print("\n=== VesperScheduler.load_tasks() ===\n")

    with tempfile.TemporaryDirectory() as _tmp:
        vault = Path(_tmp) / "vault"
        rec_dir = vault / "agenda" / "recurring"
        one_dir = vault / "agenda" / "one-off"
        rec_dir.mkdir(parents=True)
        one_dir.mkdir(parents=True)

        (rec_dir / "briefing.md").write_text(_RECURRING_TASK)
        (rec_dir / "no-schedule.md").write_text(_NO_SCHEDULE)
        (one_dir / "futuro.md").write_text(_FUTURE_ONE_OFF)
        (one_dir / "scaduto.md").write_text(_PAST_ONE_OFF)

        mock_jq = MagicMock()
        mock_jq.run_custom = MagicMock()
        mock_jq.run_once = MagicMock()
        mock_jq.jobs = MagicMock(return_value=[])

        import config as _config_mod
        original_vault = _config_mod.Config.vault_path
        _config_mod.Config.vault_path = vault
        try:
            scheduler = VesperScheduler(mock_jq)
            scheduler.load_tasks()
        finally:
            _config_mod.Config.vault_path = original_vault

        check(
            "load_tasks: run_custom chiamato 1 volta (solo task con schedule)",
            mock_jq.run_custom.call_count == 1,
        )
        check(
            "load_tasks: run_once chiamato 1 volta (solo task futuro)",
            mock_jq.run_once.call_count == 1,
        )

        # Verifica argomenti di run_custom
        rc_kwargs = mock_jq.run_custom.call_args[1]
        check(
            "load_tasks: run_custom ha job_kwargs con trigger",
            "trigger" in rc_kwargs.get("job_kwargs", {}),
        )

        # Verifica argomenti di run_once (when deve essere un datetime)
        ro_kwargs = mock_jq.run_once.call_args[1]
        when_arg = ro_kwargs.get("when")
        check(
            "load_tasks: run_once ha when come datetime",
            isinstance(when_arg, datetime),
        )
        check(
            "load_tasks: run_once ha when nel futuro",
            when_arg > datetime.now(timezone.utc) if when_arg else False,
        )

    print("\n=== VesperScheduler.reload_tasks() ===\n")

    with tempfile.TemporaryDirectory() as _tmp2:
        vault2 = Path(_tmp2) / "vault"
        (vault2 / "agenda" / "recurring").mkdir(parents=True)
        (vault2 / "agenda" / "one-off").mkdir(parents=True)
        (vault2 / "agenda" / "recurring" / "task.md").write_text(_RECURRING_TASK)

        mock_job1 = MagicMock()
        mock_job2 = MagicMock()
        mock_jq2 = MagicMock()
        mock_jq2.jobs = MagicMock(return_value=[mock_job1, mock_job2])
        mock_jq2.run_custom = MagicMock()
        mock_jq2.run_once = MagicMock()

        import config as _config_mod2
        original_vault2 = _config_mod2.Config.vault_path
        _config_mod2.Config.vault_path = vault2
        try:
            scheduler2 = VesperScheduler(mock_jq2)
            scheduler2.reload_tasks()
        finally:
            _config_mod2.Config.vault_path = original_vault2

        check(
            "reload_tasks: schedule_removal chiamato per ogni job esistente",
            mock_job1.schedule_removal.called and mock_job2.schedule_removal.called,
        )
        check(
            "reload_tasks: load_tasks eseguita dopo rimozione (run_custom chiamato)",
            mock_jq2.run_custom.called,
        )

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
