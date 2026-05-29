"""
SubagentConfig — parametri di sampling, contesto e schema di output per ogni ruolo.

Tutti i valori sono costanti di design: non vengono letti da .env né da file esterni.

Il Manager opera su {"success": bool, "output": any} per uniformità. I campi
tipizzati per ogni ruolo vivono dentro "output": il Manager non sa come sono
strutturati internamente.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentConfig:
    """Configurazione immutabile per un ruolo di subagent.

    Attributes:
        thinking:        Se premettere /think (True) o /no_think (False) al messaggio utente.
        temperature:     Temperatura di sampling.
        top_p:           Nucleus sampling threshold.
        repeat_penalty:  Penalità per la ripetizione di token.
        context_fields:  Chiavi del context dict da includere nel prompt.
        output_schema:   Schema atteso in output, usato per documentazione nel prompt.
                         Deve includere "success" e "output". Il valore di "output"
                         è una stringa (per output testuale) o un dict (per output strutturato).
    """

    thinking: bool
    temperature: float
    top_p: float
    repeat_penalty: float
    context_fields: tuple[str, ...]
    output_schema: dict


DEFAULT_SUBAGENT_CONFIG = SubagentConfig(
    thinking=False,
    temperature=0.7,
    top_p=1.0,
    repeat_penalty=1.0,
    context_fields=(),
    output_schema={"success": "bool", "output": "any"},
)

SUBAGENT_CONFIGS: dict[str, SubagentConfig] = {
    "RESEARCHER": SubagentConfig(
        thinking=False,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        context_fields=("request", "tool_results"),
        output_schema={
            "success": "bool",
            "output": {
                "sources": "list[{title: str, content: str, url_or_path: str}]",
                "summary": "str — sintesi delle informazioni recuperate",
            },
        },
    ),
    "ANALYZER": SubagentConfig(
        thinking=True,
        temperature=0.4,
        top_p=0.9,
        repeat_penalty=1.1,
        context_fields=("request", "tool_results"),
        output_schema={
            "success": "bool",
            "output": {
                "findings": "list[str] — osservazioni specifiche sull'analisi",
                "confidence": '"high" | "medium" | "low"',
                "summary": "str — sintesi dell'analisi",
            },
        },
    ),
    "REASONER": SubagentConfig(
        thinking=True,
        temperature=0.3,
        top_p=0.85,
        repeat_penalty=1.1,
        context_fields=("request", "tool_results", "errors"),
        output_schema={
            "success": "bool",
            "output": {
                "reasoning": "str — ragionamento con passaggi espliciti",
                "conclusion": "str — conclusione difendibile",
                "caveats": "list[str] — limitazioni o assunzioni rilevanti",
            },
        },
    ),
    "GENERATOR": SubagentConfig(
        thinking=False,
        temperature=0.72,
        top_p=0.95,
        repeat_penalty=1.05,
        context_fields=("request", "conversation_history", "context_memory", "tool_results", "user_profile", "agent_profile", "instructions", "format"),
        output_schema={
            "success": "bool",
            "output": "str — risposta finale destinata all'utente",
        },
    ),
    "STORAGE_MANAGER": SubagentConfig(
        thinking=False,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.15,
        context_fields=("request", "tool_results"),
        output_schema={
            "success": "bool",
            "output": {
                "action": '"write" | "update" | "skip"',
                "target_path": "str — path del documento nel vault",
                "reason": "str — motivazione della decisione",
            },
        },
    ),
}


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    for role, cfg in SUBAGENT_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {role}")
        print(f"{'='*60}")
        print(f"  thinking:       {cfg.thinking}")
        print(f"  temperature:    {cfg.temperature}")
        print(f"  top_p:          {cfg.top_p}")
        print(f"  repeat_penalty: {cfg.repeat_penalty}")
        print(f"  context_fields: {cfg.context_fields}")
        print(f"  output_schema:")
        for k, v in cfg.output_schema.items():
            if isinstance(v, dict):
                print(f"    {k!r}:")
                for nk, nv in v.items():
                    print(f"      {nk!r}: {nv!r}")
            else:
                print(f"    {k!r}: {v!r}")

    print(f"\n{'='*60}")
    print("  DEFAULT")
    print(f"{'='*60}")
    print(f"  thinking={DEFAULT_SUBAGENT_CONFIG.thinking} "
          f"temperature={DEFAULT_SUBAGENT_CONFIG.temperature} "
          f"top_p={DEFAULT_SUBAGENT_CONFIG.top_p} "
          f"repeat_penalty={DEFAULT_SUBAGENT_CONFIG.repeat_penalty}")
