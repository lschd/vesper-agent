"""
ANALYZER — analisi e comprensione profonda di documenti, email, codice, thread.

Tool disponibili: read_document.
"""
from src.subagents.base import BaseSubagent


class Analyzer(BaseSubagent):
    """Analizza in profondità contenuti forniti nel contesto."""

    tool_whitelist = ["read_document"]

    def __init__(self) -> None:
        super().__init__("analyzer")
