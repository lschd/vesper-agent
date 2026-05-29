"""
Orchestrator — riceve la richiesta dell'utente e pianifica le azioni necessarie.

Risponde esclusivamente in JSON. Non conosce le piattaforme di origine: ragiona
solo sulle azioni disponibili che riceve (già filtrate per permessi dall'interfaccia).
In autonomous_mode non chiede mai chiarimenti: esegue o fallisce con errore strutturato.

Output atteso:
    {
        "RETRIEVE": {"query": "...", "source": "vault"},
        "GENERATE": {"format": "briefing"}
    }
"""
import logging

logger = logging.getLogger(__name__)

ACTION_DESCRIPTIONS: dict[str, str] = {
    "RETRIEVE": "recupera informazioni dal vault o dal web",
    "ANALYZE":  "analizza in profondità un documento o contenuto",
    "REASON":   "ragionamento logico, valutazione opzioni, analisi rischi",
    "GENERATE": "crea contenuto testuale per l'utente",
    "STORE":    "salva o aggiorna documenti nel vault",
}


class Orchestrator:
    """Pianifica le azioni necessarie per rispondere alla richiesta dell'utente."""

    async def plan(
        self,
        user_message: str,
        available_actions: list[str],
        context,
        conversation_history: list[dict] | None = None,
        context_memory: list[dict] | None = None,
        vault_hint: str | None = None,
    ) -> dict:
        """
        Chiama il LLM con il system prompt dell'Orchestrator e restituisce il piano.

        Args:
            user_message:         Testo della richiesta dell'utente.
            available_actions:    Azioni disponibili in questa sessione (già filtrate per permessi).
            context:              RequestContext corrente (usato per autonomous_mode).
            conversation_history: Messaggi precedenti nel formato [{"role": "user"|"assistant", "content": str}].
            context_memory:       Capsule di memoria di sessione rilevanti (vault/web recuperati in turni precedenti).

        Returns:
            Piano validato: {"AZIONE": {"chiave": "valore"}, ...}.

        Raises:
            ValueError:      se il piano non contiene azioni valide.
            FileNotFoundError: se src/prompts/orchestrator.md non esiste.
            httpx.HTTPError: per errori di rete verso il server LLM.
        """
        try:
            from src.core.llm.client import get_llm_client
        except ImportError:
            from core.llm.client import get_llm_client  # type: ignore[no-redef]

        llm = get_llm_client()

        # 1. Carica il system prompt dal vault
        system_prompt = await llm.load_prompt("orchestrator")

        # 2. Costruisce il messaggio per il LLM
        llm_user_message = _build_user_message(
            user_message=user_message,
            available_actions=available_actions,
            autonomous_mode=getattr(context, "autonomous_mode", False),
            conversation_history=conversation_history,
            context_memory=context_memory,
            vault_hint=vault_hint,
        )

        # 3. Chiama il LLM — validate_json è già applicato internamente da llm.complete()
        raw_plan = await llm.complete(system_prompt, llm_user_message, temperature=0.3, max_tokens=1024)

        # Safety net: {"error": ...} in modalità normale → fallback a GENERATE
        if "error" in raw_plan and not getattr(context, "autonomous_mode", False):
            logger.warning(
                "Orchestrator: piano errore in modalità normale — fallback a GENERATE. Piano: %r",
                raw_plan,
            )
            raw_plan = {"GENERATE": {"format": "risposta"}}

        # 4. Filtra le azioni non disponibili in questa sessione
        available_set = set(available_actions)
        filtered_plan: dict = {}

        for action, action_input in raw_plan.items():
            if action not in available_set:
                logger.warning(
                    "Orchestrator: azione '%s' non disponibile in questa sessione — rimossa",
                    action,
                )
                continue

            if isinstance(action_input, dict):
                filtered_plan[action] = action_input
            elif action_input is None:
                filtered_plan[action] = {}
            else:
                # L'LLM ha restituito un input non-dict: lo wrappa preservando l'informazione
                filtered_plan[action] = {"value": action_input}
                logger.warning(
                    "Orchestrator: input di '%s' non e' un dict (%s) — wrappato",
                    action, type(action_input).__name__,
                )

        # 5. Il piano deve contenere almeno un'azione valida
        if not filtered_plan:
            raise ValueError(
                f"Il piano LLM non contiene azioni valide. "
                f"Piano ricevuto: {raw_plan!r}. "
                f"Azioni disponibili: {available_actions!r}"
            )

        logger.debug("Orchestrator: piano finale — %r", filtered_plan)
        return filtered_plan


# ---------------------------------------------------------------------------
# Costruzione del messaggio utente per il LLM
# ---------------------------------------------------------------------------

def _build_user_message(
    user_message: str,
    available_actions: list[str],
    autonomous_mode: bool,
    conversation_history: list[dict] | None,
    context_memory: list[dict] | None = None,
    vault_hint: str | None = None,
) -> str:
    """
    Assembla il testo da passare al LLM come 'user message'.

    Sezioni (nell'ordine):
      1. Cronologia conversazione (se presente)
      2. Memoria di sessione (se presente)
      3. Richiesta utente
      4. Azioni disponibili con descrizioni
      5. Istruzione autonoma (solo se autonomous_mode=True)
    """
    parts: list[str] = []

    if conversation_history:
        lines = ["## Cronologia conversazione"]
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"[Utente]: {content}")
            elif role == "assistant":
                lines.append(f"[Vesper]: {content}")
            else:
                lines.append(f"[{role}]: {content}")
        parts.append("\n".join(lines))

    if context_memory:
        _MAX_CAPSULE_CHARS = 300
        _MAX_CAPSULES = 3
        caps: list[str] = []
        for cap in context_memory[:_MAX_CAPSULES]:
            content = cap.get("content", "")
            if len(content) > _MAX_CAPSULE_CHARS:
                content = content[:_MAX_CAPSULE_CHARS] + "..."
            if content:
                caps.append(content)
        if caps:
            parts.append("## Memoria di sessione\n\n" + "\n\n---\n\n".join(caps))

    if vault_hint:
        parts.append(f"## Contenuto vault rilevante\n{vault_hint}")

    parts.append(f"## Richiesta utente\n{user_message}")

    actions_lines = ["## Azioni disponibili"]
    for action in available_actions:
        desc = ACTION_DESCRIPTIONS.get(action, "(nessuna descrizione)")
        actions_lines.append(f"- {action}: {desc}")
    parts.append("\n".join(actions_lines))

    if autonomous_mode:
        parts.append(
            "## Modalita' autonoma\n"
            "Non chiedere mai chiarimenti all'utente. "
            "Esegui con le informazioni disponibili o restituisci un errore strutturato in JSON."
        )

    return "\n\n".join(parts)


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
    # Test _build_user_message
    # ------------------------------------------------------------------

    print("\n=== _build_user_message ===\n")

    msg = _build_user_message(
        user_message="Cerca informazioni su Python asyncio.",
        available_actions=["RETRIEVE", "GENERATE"],
        autonomous_mode=False,
        conversation_history=None,
    )
    check("contiene la richiesta utente", "Cerca informazioni su Python asyncio." in msg)
    check("contiene RETRIEVE", "RETRIEVE" in msg)
    check("contiene GENERATE", "GENERATE" in msg)
    check("contiene descrizione RETRIEVE", ACTION_DESCRIPTIONS["RETRIEVE"] in msg)
    check("nessuna sezione autonoma senza autonomous_mode", "Modalita' autonoma" not in msg)

    msg_hint = _build_user_message(
        user_message="Cosa sai sui transformer?",
        available_actions=["RETRIEVE", "GENERATE"],
        autonomous_mode=False,
        conversation_history=None,
        vault_hint="Trovati 2 documenti rilevanti nel vault (score massimo: 0.87).",
    )
    check("vault_hint: sezione presente nel messaggio", "## Contenuto vault rilevante" in msg_hint)
    check("vault_hint: testo hint incluso", "0.87" in msg_hint)
    check("vault_hint: sezione precede richiesta utente",
          msg_hint.index("Contenuto vault rilevante") < msg_hint.index("Richiesta utente"))

    msg_no_hint = _build_user_message(
        user_message="Cosa sai sui transformer?",
        available_actions=["RETRIEVE", "GENERATE"],
        autonomous_mode=False,
        conversation_history=None,
        vault_hint=None,
    )
    check("vault_hint=None: sezione assente", "## Contenuto vault rilevante" not in msg_no_hint)

    msg_auto = _build_user_message(
        user_message="Briefing mattutino.",
        available_actions=["RETRIEVE"],
        autonomous_mode=True,
        conversation_history=None,
    )
    check("autonomous_mode aggiunge istruzione", "autonoma" in msg_auto.lower())

    msg_hist = _build_user_message(
        user_message="Domanda nuova.",
        available_actions=["GENERATE"],
        autonomous_mode=False,
        conversation_history=[
            {"role": "user", "content": "primo messaggio"},
            {"role": "assistant", "content": "prima risposta"},
        ],
    )
    check("cronologia include messaggio utente", "primo messaggio" in msg_hist)
    check("cronologia include risposta assistant", "prima risposta" in msg_hist)
    check("cronologia precede richiesta corrente",
          msg_hist.index("primo messaggio") < msg_hist.index("Domanda nuova."))

    # ------------------------------------------------------------------
    # Test Orchestrator.plan
    # ------------------------------------------------------------------

    print("\n=== Orchestrator.plan ===\n")

    async def _run_orchestrator_tests() -> None:
        import src.core.llm.client
        from src.core.context import new_context

        orchestrator = Orchestrator()
        mock_llm = MagicMock()
        mock_llm.load_prompt = AsyncMock(return_value="Sei l'Orchestrator. Rispondi in JSON.")

        # Test 1: piano valido con azioni disponibili
        mock_llm.complete = AsyncMock(
            return_value={"RETRIEVE": {"query": "notizie tech", "source": "web"}, "GENERATE": {"format": "summary"}}
        )
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx = new_context()
            plan = await orchestrator.plan(
                user_message="Dammi le ultime notizie tech.",
                available_actions=["RETRIEVE", "ANALYZE", "GENERATE"],
                context=ctx,
            )
        check("piano restituisce azioni valide", set(plan.keys()) == {"RETRIEVE", "GENERATE"})
        check("input RETRIEVE e' un dict", isinstance(plan.get("RETRIEVE"), dict))
        check("query RETRIEVE presente", plan["RETRIEVE"].get("query") == "notizie tech")

        # Test 2: azione non disponibile rimossa con warning
        mock_llm.complete = AsyncMock(
            return_value={"RETRIEVE": {"query": "test"}, "STORE": {"path": "raw/out.md"}}
        )
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx2 = new_context()
            plan2 = await orchestrator.plan(
                user_message="Cerca e salva.",
                available_actions=["RETRIEVE"],  # STORE non disponibile
                context=ctx2,
            )
        check("azione non disponibile rimossa", "STORE" not in plan2)
        check("azione disponibile conservata", "RETRIEVE" in plan2)

        # Test 3: tutte le azioni filtrate → ValueError
        mock_llm.complete = AsyncMock(
            return_value={"STORE": {"path": "raw/doc.md"}}
        )
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx3 = new_context()
            raised = False
            try:
                await orchestrator.plan(
                    user_message="Qualcosa.",
                    available_actions=["RETRIEVE", "GENERATE"],
                    context=ctx3,
                )
            except ValueError:
                raised = True
        check("ValueError se nessuna azione valida nel piano", raised)

        # Test 4: input non-dict normalizzato a dict
        mock_llm.complete = AsyncMock(
            return_value={"RETRIEVE": "cerca qualcosa"}
        )
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx4 = new_context()
            plan4 = await orchestrator.plan(
                user_message="Cerca.",
                available_actions=["RETRIEVE"],
                context=ctx4,
            )
        check("input non-dict wrappato in dict", isinstance(plan4.get("RETRIEVE"), dict))
        check("valore originale preservato nel wrapper", plan4["RETRIEVE"].get("value") == "cerca qualcosa")

        # Test 5: autonomous_mode — la istruzione appare nel user_message inviato al LLM
        captured_messages: list[str] = []

        async def _capture_complete(system_prompt: str, user_message: str, **kw) -> dict:
            captured_messages.append(user_message)
            return {"RETRIEVE": {"query": "briefing"}}

        mock_llm.complete = _capture_complete
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx5 = new_context(autonomous_mode=True)
            await orchestrator.plan(
                user_message="Briefing mattutino.",
                available_actions=["RETRIEVE", "GENERATE"],
                context=ctx5,
            )
        check("autonomous_mode aggiunge istruzione al prompt LLM",
              any("autonoma" in m.lower() for m in captured_messages))

        # Test 6: errore LLM propagato
        mock_llm.complete = AsyncMock(side_effect=ValueError("connessione rifiutata"))
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx6 = new_context()
            exc_raised = False
            try:
                await orchestrator.plan("Prova.", ["RETRIEVE"], ctx6)
            except ValueError:
                exc_raised = True
        check("errore LLM viene propagato", exc_raised)

        # Test 7: cronologia conversazione inclusa nel prompt
        captured_hist: list[str] = []

        async def _capture_hist(system_prompt: str, user_message: str, **kw) -> dict:
            captured_hist.append(user_message)
            return {"GENERATE": {"format": "risposta"}}

        mock_llm.complete = _capture_hist
        with patch.object(src.core.llm.client, "get_llm_client", return_value=mock_llm):
            ctx7 = new_context()
            await orchestrator.plan(
                user_message="Nuova domanda.",
                available_actions=["GENERATE"],
                context=ctx7,
                conversation_history=[
                    {"role": "user", "content": "domanda precedente"},
                    {"role": "assistant", "content": "risposta precedente"},
                ],
            )
        check("cronologia presente nel prompt inviato al LLM",
              any("domanda precedente" in m for m in captured_hist))

    asyncio.run(_run_orchestrator_tests())

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
