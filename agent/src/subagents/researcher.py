"""
RESEARCHER — recupera informazioni da vault o web senza interpretarle.

Tool disponibili: vault_search, web_search, read_document.
Conosce la struttura del vault e la usa per orientare le ricerche.
"""
from src.subagents.base import BaseSubagent


class Researcher(BaseSubagent):
    """Recupera informazioni grezze da vault o web senza interpretarle."""

    tool_whitelist = ["vault_search", "web_search", "read_document"]

    def __init__(self) -> None:
        super().__init__("researcher")
