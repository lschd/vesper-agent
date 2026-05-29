"""Re-ranker HTTP client. Interroga il re-ranker server (llama-server su RERANKER_PORT)."""
import asyncio
import logging
import math

import httpx

logger = logging.getLogger(__name__)

_SYSTEM_MSG = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)

# Empty <think> block suppresses reasoning tokens for speed
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n\n"
    "<Query>: {query}\n\n"
    "<Document>: {document}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

_MAX_DOC_CHARS = 6000  # ~1500 tokens, fits in 4096 ctx with instruction+query overhead


def _score_from_logprobs(top_lp: dict) -> float:
    yes_lp = next((lp for tok, lp in top_lp.items() if tok.strip().lower() == "yes"), None)
    no_lp  = next((lp for tok, lp in top_lp.items() if tok.strip().lower() == "no"),  None)

    if yes_lp is None and no_lp is None:
        return 0.5
    if yes_lp is None:
        return 0.0
    if no_lp is None:
        return 1.0

    yes_p = math.exp(yes_lp)
    no_p  = math.exp(no_lp)
    return yes_p / (yes_p + no_p)


async def _score_pair(
    client: httpx.AsyncClient,
    base_url: str,
    query: str,
    document: str,
    instruction: str,
) -> float:
    prompt = _PROMPT_TEMPLATE.format(
        system=_SYSTEM_MSG,
        instruction=instruction,
        query=query,
        document=document[:_MAX_DOC_CHARS],
    )
    resp = await client.post(
        f"{base_url}/v1/completions",
        json={"prompt": prompt, "max_tokens": 1, "logprobs": 20},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()
    top_lp: dict = data["choices"][0].get("logprobs", {}).get("top_logprobs", [{}])[0]
    return _score_from_logprobs(top_lp)


async def rerank(query: str, documents: list[str], instruction: str) -> list[dict]:
    """
    Scores each document against the query using the cross-encoder reranker.

    Returns list of {"index": int, "relevance_score": float}.
    Compatible with the /rerank API format expected by vault_search.
    """
    from config import Config

    base_url = f"http://127.0.0.1:{Config.reranker_port}"

    async with httpx.AsyncClient() as client:
        scores = await asyncio.gather(
            *[_score_pair(client, base_url, query, doc, instruction) for doc in documents],
            return_exceptions=True,
        )

    if all(isinstance(s, Exception) for s in scores):
        raise scores[0]  # server completamente irraggiungibile — vault_search usa il fallback vettoriale

    results = []
    for i, score in enumerate(scores):
        if isinstance(score, Exception):
            logger.warning("Errore scoring documento %d: %s", i, score)
            results.append({"index": i, "relevance_score": 0.5})
        else:
            results.append({"index": i, "relevance_score": score})

    return results
