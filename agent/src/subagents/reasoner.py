"""
REASONER — ragionamento logico e strategico.

Valuta opzioni, rischi e scenari. Tool futuri da definire.
"""
from src.subagents.base import BaseSubagent


class Reasoner(BaseSubagent):
    """Ragionamento critico: valuta opzioni, rischi e scenari."""

    tool_whitelist: list[str] = []  # tool futuri

    def __init__(self) -> None:
        super().__init__("reasoner")
