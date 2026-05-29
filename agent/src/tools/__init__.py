"""Tool Python invocati dai subagents."""

TOOL_DESCRIPTIONS: dict[str, str] = {
    "vault_search": "cerca nella wiki tramite RAG + graph traversal sui [[link]] Obsidian",
    "web_search": "ricerca web: recupera link DuckDuckGo, esegue GET parallelo, pulisce HTML",
    "read_document": "legge un documento .md dal vault (path assoluto o relativo a VAULT_PATH)",
    "write_document": "scrive un nuovo documento .md nel vault; non sovrascrive file esistenti",
    "update_document": "aggiorna una sezione o fa append su un documento .md; crea il file se non esiste",
}
