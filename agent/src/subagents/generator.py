"""
GENERATOR — genera la risposta finale per l'utente.

Legge AGENT.md e USER.md prima di produrre l'output, per rispettare le regole
dell'agente e le preferenze apprese dell'utente.
"""
import logging

from src.subagents.base import BaseSubagent

logger = logging.getLogger(__name__)


class Generator(BaseSubagent):
    """
    Genera la risposta finale destinata all'utente.

    Prima di invocare il LLM, arricchisce il contesto con i contenuti di
    AGENT.md (regole e comportamento dell'agente) e USER.md (profilo utente),
    in modo che il modello rispetti personalità e preferenze apprese.
    """

    tool_whitelist: list[str] = []

    def __init__(self) -> None:
        super().__init__("generator")

    async def run(self, task: str, context: dict, tools: list[str]) -> dict:
        """
        Override che inietta AGENT.md e USER.md nel contesto prima della chiamata LLM.

        I file vengono letti con read_document. Se non disponibili, viene loggato
        un warning ma l'esecuzione prosegue senza interrompersi.
        """
        try:
            from src.tools.read_document import read_document
        except ImportError:
            from read_document import read_document  # type: ignore[no-redef]

        enriched = dict(context)

        for filename, context_key in [("AGENT.md", "agent_profile"), ("USER.md", "user_profile")]:
            try:
                enriched[context_key] = await read_document(filename)
            except (FileNotFoundError, OSError) as exc:
                logger.warning("Generator: impossibile caricare '%s': %s", filename, exc)

        return await super().run(task, enriched, tools)
