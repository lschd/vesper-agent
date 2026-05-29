"""RequestContext — schema condiviso per il ciclo di vita di una richiesta."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RequestContext:
    """
    Accumulatore di stato per una singola richiesta utente o task proattivo.

    Viene creato all'inizio della richiesta, arricchito durante l'elaborazione da
    Manager e subagents, e salvato nella memoria persistente al termine.

    Fields:
        request_id:         UUID univoco della richiesta.
        timestamp:          Momento di inizio (UTC).
        autonomous_mode:    True per task eseguiti dal ProactiveDispatcher.
        actions_executed:   Nomi delle azioni eseguite (es. ["RETRIEVE", "GENERATE"]).
        results:            Output per azione: {"RETRIEVE": [...], "GENERATE": "..."}.
        documents_modified: Path dei documenti scritti o aggiornati nel vault.
        tools_called:       Log di ogni chiamata tool con input, output e durata.
        errors:             Errori strutturati per step con timestamp ISO.
    """

    request_id: str
    timestamp: datetime
    autonomous_mode: bool = False
    actions_executed: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    documents_modified: list[str] = field(default_factory=list)
    tools_called: list[dict] = field(default_factory=list)
    # ogni entry: {"tool": str, "input": dict, "output": any, "duration_ms": int}
    errors: list[dict] = field(default_factory=list)
    # ogni entry: {"step": str, "reason": str, "timestamp": str ISO}

    # ------------------------------------------------------------------
    # Helper per accumulo progressivo durante l'elaborazione
    # ------------------------------------------------------------------

    def add_tool_call(
        self,
        tool: str,
        input: dict,  # noqa: A002
        output: object,
        duration_ms: int,
    ) -> None:
        """Registra l'invocazione di un tool con il suo risultato e la durata."""
        self.tools_called.append(
            {"tool": tool, "input": input, "output": output, "duration_ms": duration_ms}
        )

    def add_error(self, step: str, reason: str) -> None:
        """Registra un errore strutturato per lo step indicato."""
        self.errors.append(
            {
                "step": step,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def add_result(self, action: str, result: object) -> None:
        """Salva il risultato di un'azione in results[action]."""
        self.results[action] = result
        if action not in self.actions_executed:
            self.actions_executed.append(action)

    def to_dict(self) -> dict:
        """
        Serializzazione completa per il salvataggio su SQLite.

        Converte i tipi non-JSON-nativi (datetime → ISO string) in modo che
        il dict possa essere passato direttamente a json.dumps().
        """
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "autonomous_mode": self.autonomous_mode,
            "actions_executed": self.actions_executed,
            "results": self.results,
            "documents_modified": self.documents_modified,
            "tools_called": self.tools_called,
            "errors": self.errors,
        }


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def new_context(autonomous_mode: bool = False) -> RequestContext:
    """
    Crea un RequestContext con request_id UUID4 e timestamp UTC generati automaticamente.

    Args:
        autonomous_mode: True per task proattivi eseguiti dal ProactiveDispatcher.

    Returns:
        RequestContext pronto all'uso.
    """
    return RequestContext(
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        autonomous_mode=autonomous_mode,
    )
