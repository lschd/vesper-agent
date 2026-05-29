"""
VesperTUI — interfaccia terminale locale con stellina animata in fondo allo schermo.

Usa rich.live.Live per il "fixed bottom" e un thread daemon per l'animazione.
Per il funzionamento corretto del fixed bottom, tutto l'output (logging incluso)
deve passare per lo stesso Console: in main.py il RichHandler riceve `tui.console`.

Stati della stellina:
  IDLE       — stellina statica (✦): Vesper pronto, nessuna attività
  LOADING    — blink ✦ ↔ ✧: operazioni di avvio (ChromaDB, indicizzazione, LLM)
  PROCESSING — animazione glyph (✦ ✧ ★ ✱ ✸ ✹): elaborazione LLM in corso

Espansione futura: chat interattiva, pannello log, statistiche in tempo reale.

Test standalone:
    python src/interfaces/tui.py
"""

import logging
import sys
import threading
from typing import ClassVar

try:
    from rich.console import Console
    from rich.live import Live
    from rich.text import Text
    from rich.padding import Padding
    from rich.rule import Rule
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    Console = None   # type: ignore[assignment, misc]
    Live = None      # type: ignore[assignment, misc]
    Text = None      # type: ignore[assignment, misc]
    Padding = None   # type: ignore[assignment, misc]
    Rule = None      # type: ignore[assignment, misc]

try:
    from src.interfaces.base import AbstractInterface
except ImportError:
    from base import AbstractInterface  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ── Singleton globale — esposto per base.py senza import circolare ────────────
# base.py importa tui.py (deferred, dentro una funzione), non a livello modulo:
# sicuro perché tui.py importa AbstractInterface da base.py solo a livello modulo.

_active_tui: "VesperTUI | None" = None


def _register_active_tui(tui: "VesperTUI") -> None:
    """Chiamato da main.py dopo aver istanziato la TUI."""
    global _active_tui
    _active_tui = tui


def set_tui_state(state: str) -> None:
    """
    Aggiorna lo stato dell'animazione se una TUI è attiva.

    No-op se nessuna TUI è registrata o se lo stato non è valido.
    Thread-safe (delega al lock interno di VesperTUI.set_state).
    Usato da AbstractInterface.process_request() per segnalare
    inizio/fine elaborazione senza dipendere direttamente dalla classe.
    """
    if _active_tui is not None:
        _active_tui.set_state(state)


# ── Animazioni per stato ──────────────────────────────────────────────────────
#
# Ogni lista è una sequenza di glyph (il thread daemon cicla ad intervallo fisso).
# Frame singolo → stellina statica (nessuna iterazione effettiva).
#
#   IDLE       ─── ✦                  (statico)
#   LOADING    ─── ✦ ↔ ✧             (blink, alternanza solido/vuoto)
#   PROCESSING ─── animazione glyph   (✦ ✧ ★ ✱ ✸ ✹ …)

_FRAMES: dict[str, list[str]] = {
    "idle":       ["✦  send it"],
    "loading":    ["✧", "✧", "✧ ·", "✧ ··", "✧ ···", "✧ ··", "✧ ·"],
    "processing": ["✦", "✧", "✦", "✱", "✸", "✹", "✸", "✱"],
}

# Intervallo tra frame per stato (secondi).
# IDLE non ha bisogno di un'entrata: frame singolo, il loop non avanza mai.
# Il fallback 0.20 si applica a qualsiasi stato non elencato.
_FRAME_INTERVALS: dict[str, float] = {
    "loading":    0.25,
    "processing": 0.13,
}


class VesperTUI(AbstractInterface):
    """
    Interfaccia TUI locale con status bar animata in fondo al terminale.

    Il colore della stellina viene passato da main.py per coerenza con il banner.
    Usa threading (non asyncio) così è avviabile prima di asyncio.run().
    """

    # Costanti di stato — stringhe dirette per uso in main.py senza import VesperTUI
    IDLE:       ClassVar[str] = "idle"
    LOADING:    ClassVar[str] = "loading"
    PROCESSING: ClassVar[str] = "processing"

    def __init__(self, color: str = "rgb(74,222,128)") -> None:
        """
        Args:
            color: Colore della stellina in formato rich (es. "rgb(74,222,128)").
                   Definito in main.py per coerenza col banner.
        """
        self._color = color
        self._state: str = self.LOADING
        self._frame: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if _RICH_AVAILABLE:
            self._console: "Console" = Console(highlight=False)
            self._live: "Live | None" = None
        else:
            self._console = None  # type: ignore[assignment]
            self._live = None
            logger.warning("VesperTUI: rich non disponibile. TUI disabilitata")

    # ── AbstractInterface ─────────────────────────────────────────────────────

    async def send(self, target: str, message: str) -> None:
        """
        Mostra la risposta di Vesper sopra la status bar.

        Visivamente separata dai log: riga orizzontale verde sopra e sotto,
        testo con padding laterale.
        """
        if self._console is None:
            return
        self._console.print()
        self._console.print(Rule(style=f"dim {self._color}"))
        self._console.print(Padding(Text(message), (0, 1)))
        self._console.print(Rule(style=f"dim {self._color}"))
        self._console.print()

    async def send_error(self, target: str, task_name: str, reason: str) -> None:
        """Stampa un errore strutturato via Console rich."""
        if self._console is not None:
            self._console.print(f"[red]✗ {task_name}: {reason}[/red]")

    def get_permissions(self) -> list[str]:
        """Permessi completi — la TUI è interfaccia admin come Telegram."""
        return ["RETRIEVE", "ANALYZE", "REASON", "GENERATE", "STORE"]

    # ── Proprietà pubblica ────────────────────────────────────────────────────

    @property
    def console(self) -> "Console | None":
        """
        Console condiviso — passarlo a RichHandler in main.py.

        Tutti i log che vanno per questo Console vengono automaticamente
        visualizzati sopra il Live display, mantenendo la stellina fissa in basso.
        """
        return self._console

    # ── API di stato ──────────────────────────────────────────────────────────

    def set_state(self, state: str) -> None:
        """
        Aggiorna lo stato dell'animazione.

        Può essere chiamato da qualsiasi thread; thread-safe via Lock.

        Args:
            state: Una delle costanti IDLE / LOADING / PROCESSING.
                   Equivalente alle stringhe "idle" / "loading" / "processing".
        """
        if state not in _FRAMES:
            logger.warning("VesperTUI.set_state: stato sconosciuto '%s' — ignorato", state)
            return
        with self._lock:
            if state != self._state:
                self._state = state
                self._frame = 0  # ricomincia dall'inizio dell'animazione

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self) -> "Padding":
        """Costruisce il renderable corrente: stellina animata con riga di separazione sopra."""
        with self._lock:
            state = self._state
            frame_idx = self._frame

        frames = _FRAMES.get(state, _FRAMES[self.IDLE])
        glyph = frames[frame_idx % len(frames)]

        star_line = Text()
        star_line.append(f"{glyph}\n", style=f"bold {self._color}")

        return Padding(star_line, (1, 0, 0, 0))

    # ── Thread di animazione ──────────────────────────────────────────────────

    def _animate_thread(self) -> None:
        """
        Loop di animazione eseguito in un thread daemon.

        Avanza il frame corrente e aggiorna il Live display, poi attende
        l'intervallo specifico dello stato corrente (da _FRAME_INTERVALS).
        Si ferma quando _stop_event viene settato.
        """
        while not self._stop_event.is_set():
            with self._lock:
                state = self._state
                frames = _FRAMES.get(state, _FRAMES[self.IDLE])
                if len(frames) > 1:
                    self._frame = (self._frame + 1) % len(frames)

            if self._live is not None:
                try:
                    self._live.update(self._render())
                except Exception:
                    pass

            self._stop_event.wait(_FRAME_INTERVALS.get(state, 0.20))

    # ── Ciclo di vita ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Avvia il Live display e il thread di animazione.

        Non richiede un event loop asyncio attivo — chiamabile prima di asyncio.run().
        Idempotente: successive chiamate vengono ignorate se già avviato.
        """
        if not _RICH_AVAILABLE or self._live is not None:
            return

        self._stop_event.clear()

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
            vertical_overflow="visible",
        )
        self._live.start(refresh=True)

        self._thread = threading.Thread(
            target=self._animate_thread,
            name="vesper-tui-anim",
            daemon=True,
        )
        self._thread.start()

        logger.debug("TUI avviata")

    def stop(self) -> None:
        """
        Ferma il thread di animazione e il Live display.

        Chiamare nel blocco finally di main._main() per cleanup corretto.
        Idempotente: successive chiamate vengono ignorate se già fermato.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._live is not None:
            self._live.stop()
            self._live = None

        logger.debug("VesperTUI fermata")


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=== VesperTUI — test standalone ===")
    print("Ogni stato dura 3 secondi.\n")

    tui = VesperTUI(color="rgb(74,222,128)")
    tui.start()

    if tui.console:
        tui.console.print("[dim]TUI avviata — stato iniziale: IDLE[/dim]")

    time.sleep(2)

    tui.set_state(VesperTUI.LOADING)
    if tui.console:
        tui.console.print("[dim]→ LOADING  (simulazione startup)[/dim]")
    time.sleep(3)

    tui.set_state(VesperTUI.PROCESSING)
    if tui.console:
        tui.console.print("[dim]→ PROCESSING  (simulazione elaborazione LLM)[/dim]")
    time.sleep(3)

    tui.set_state(VesperTUI.IDLE)
    if tui.console:
        tui.console.print("[dim]→ IDLE  (Vesper pronto)[/dim]")
    time.sleep(2)

    tui.stop()
    print("\n=== Fine test ===")
    sys.exit(0)
