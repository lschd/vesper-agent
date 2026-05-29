"""
web_search — Ricerca web tramite DuckDuckGo HTML con fetch parallelo delle pagine.

Pipeline: DDG HTML → estrazione link → GET parallelo → pulizia HTML.

Funzioni esposte:
    async_web_search(query, max_results) -> list[dict]   — da usare con await nell'app
    web_search(query, max_results) -> list[dict]          — wrapper sincrono per script/test

Risultato: [{"url": str, "content": str}]
"""
import asyncio
import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# UA realistico richiesto da DuckDuckGo: "VesperBot/1.0" riceve un bot-challenge 202.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_DDG_POST_URL = "https://html.duckduckgo.com/html/"
_TIMEOUT = httpx.Timeout(10.0)
_TAGS_TO_REMOVE = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


# ---------------------------------------------------------------------------
# Parsing DDG e pulizia HTML
# ---------------------------------------------------------------------------

def _extract_links(html: str, limit: int) -> list[str]:
    """
    Estrae gli URL dei risultati dalla pagina HTML di DuckDuckGo.

    Gestisce i redirect DDG (/l/?uddg=URL_encoded) e i link diretti.
    Filtra i link che puntano ancora a duckduckgo.com.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()

    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        if not href:
            continue

        # Redirect DDG: "//duckduckgo.com/l/?uddg=URL_encoded&rut=..."
        if "/l/" in href and "uddg" in href:
            full = href if href.startswith("http") else f"https:{href}"
            try:
                uddg = parse_qs(urlparse(full).query).get("uddg", [None])[0]
            except Exception:
                continue
            if uddg:
                href = uddg

        if not href.startswith("http"):
            continue
        if "duckduckgo.com" in href:
            continue
        if href in seen:
            continue

        seen.add(href)
        links.append(href)

        if len(links) >= limit:
            break

    return links


def _clean_html(html: str) -> str:
    """
    Rimuove tag di struttura/navigazione, estrae il testo, normalizza spazi e newline.

    Returns:
        Testo pulito.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(_TAGS_TO_REMOVE):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Fetch singola pagina
# ---------------------------------------------------------------------------

async def _fetch_page(client: httpx.AsyncClient, url: str) -> dict | None:
    """
    Recupera e pulisce una singola pagina web.

    Restituisce None (senza sollevare eccezioni) se la pagina non è recuperabile:
    errori di rete, timeout, status HTTP non-2xx, Content-Type non HTML, body vuoto.
    """
    try:
        resp = await client.get(url, follow_redirects=True)
    except Exception as exc:
        logger.debug("web_search: skip '%s' — %s: %s", url, type(exc).__name__, exc)
        return None

    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type:
        logger.debug(
            "web_search: skip '%s' — Content-Type non HTML: '%s'", url, content_type
        )
        return None

    if not resp.is_success:
        logger.debug("web_search: skip '%s' — HTTP %d", url, resp.status_code)
        return None

    content = _clean_html(resp.text)
    if not content:
        logger.debug("web_search: skip '%s' — contenuto vuoto dopo pulizia HTML", url)
        return None

    return {"url": url, "content": content}


# ---------------------------------------------------------------------------
# Pipeline principale
# ---------------------------------------------------------------------------

async def async_web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Ricerca web asincrona: DuckDuckGo HTML → link → fetch parallelo → pulizia.

    Versione async da usare con await nelle chiamate dall'app (FastAPI, Telegram bot).

    Args:
        query: Testo da cercare.
        max_results: Numero massimo di risultati restituiti.

    Returns:
        Lista di dict [{"url": str, "content": str}].
        Lista vuota se nessun risultato è recuperabile (nessuna eccezione sollevata).
    """
    base_headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(
        headers=base_headers, timeout=_TIMEOUT, follow_redirects=True
    ) as client:
        # 1. Interroga DuckDuckGo HTML tramite POST (restituisce URL diretti senza redirect)
        try:
            ddg_resp = await client.post(
                _DDG_POST_URL,
                data={"q": query, "b": "", "kl": "wt-wt"},
                headers={
                    **base_headers,
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://duckduckgo.com",
                    "Referer": "https://duckduckgo.com/",
                },
            )
            ddg_resp.raise_for_status()
        except Exception as exc:
            logger.error(
                "web_search: errore DDG per query %r — %s: %s",
                query, type(exc).__name__, exc,
            )
            return []

        links = _extract_links(ddg_resp.text, limit=max_results * 2)
        logger.info(
            "web_search: query=%r — DDG HTTP %d, trovati %d link",
            query, ddg_resp.status_code, len(links),
        )

        if not links:
            return []

        # 2. Fetch parallelo con asyncio.gather
        tasks = [_fetch_page(client, url) for url in links]
        raw = await asyncio.gather(*tasks)

    results = [r for r in raw if r is not None][:max_results]
    logger.info(
        "web_search: %d pagine recuperate con successo su %d link totali",
        len(results), len(links),
    )
    return results


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Wrapper sincrono di async_web_search. Usa asyncio.run() internamente.

    Compatibile solo con contesti non-async (script, REPL, test).
    Nei contesti async (FastAPI, Telegram bot) usare:
        results = await async_web_search(query, max_results)

    Args:
        query: Testo da cercare.
        max_results: Numero massimo di risultati restituiti.

    Returns:
        Lista di dict [{"url": str, "content": str}].
    """
    return asyncio.run(async_web_search(query, max_results))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    query = sys.argv[1] if len(sys.argv) > 1 else "Python asyncio tutorial"
    max_r = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Ricerca: {query!r}  (max_results={max_r})\n")

    results = web_search(query, max_results=max_r)

    if not results:
        print("Nessun risultato recuperabile.")
        sys.exit(1)

    for i, r in enumerate(results, 1):
        preview = r["content"][:200].replace("\n", " ")
        print(f"[{i}] {r['url']}")
        print(f"     {preview!r}")
        print()

    print(f"Totale: {len(results)} risultati su {max_r} richiesti.")
