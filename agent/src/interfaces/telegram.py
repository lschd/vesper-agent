"""
TelegramInterface — interfaccia admin con permessi completi.

Comandi: /start, /status, /reindex, /clear.
Upload documenti: max 20 MB, salvati in vault/raw/.
Memoria conversazione persistente via SQLite (data/conversations.db).
Pulsante "Riprova" come inline button al termine di ogni risposta.
"""
import sys as _sys
from pathlib import Path as _Path

if __name__ == "__main__":
    # Evita il conflitto di nome con la libreria python-telegram-bot
    # quando il file viene eseguito direttamente come script.
    _here = str(_Path(__file__).resolve().parent)
    _sys.path = [p for p in _sys.path if str(_Path(p).resolve()) != _here]
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import asyncio
import html as _html
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from src.interfaces.base import AbstractInterface
except ImportError:
    from base import AbstractInterface  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_TG_MAX_CHARS = 4096
_HISTORY_LIMIT = 20      # max messaggi restituiti dalla cronologia
_COMPACT_THRESHOLD = 35  # messaggi totali oltre cui scatta l'auto-compattazione
_COMPACT_MARKER = "[🗜️ MEMORIA COMPATTATA"  # prefisso riconoscibile nei messaggi compressi

# Lock per-chat: impedisce che un messaggio venga elaborato mentre la compattazione è in corso
_compact_locks: dict[int, asyncio.Lock] = {}


def _get_compact_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _compact_locks:
        _compact_locks[chat_id] = asyncio.Lock()
    return _compact_locks[chat_id]

# Regex per la conversione Markdown → HTML Telegram
_FENCED_CODE_RE  = re.compile(r'```[\w]*\n?(.*?)```', re.DOTALL)
_INLINE_CODE_RE  = re.compile(r'`([^`\n]+)`')
_BOLD_RE         = re.compile(r'\*\*(.+?)\*\*', re.DOTALL)
_ITALIC_STAR_RE  = re.compile(r'(?<!\*)\*([^*\n]+)\*(?!\*)')
_ITALIC_UNDER_RE = re.compile(r'(?<!\w)_([^_\n]+)_(?!\w)')
_HEADING_RE      = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
_STRIKE_RE       = re.compile(r'~~(.+?)~~', re.DOTALL)
_STRIP_TAGS_RE   = re.compile(r'<[^>]+'  r'>')


# ---------------------------------------------------------------------------
# SQLite — helpers di modulo (non dipendono dall'istanza del bot)
# ---------------------------------------------------------------------------

def _init_db(db_path: str) -> None:
    """Crea la tabella conversations se non esiste."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id   INTEGER NOT NULL,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                timestamp TEXT    NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_chat_id ON conversations(chat_id)"
        )
        conn.commit()


def _load_history(db_path: str, chat_id: int) -> list[dict]:
    """
    Restituisce gli ultimi _HISTORY_LIMIT messaggi per chat_id,
    nel formato [{"role": str, "content": str}] ordinati dal meno recente.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM (
                SELECT id, role, content
                FROM conversations
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (chat_id, _HISTORY_LIMIT),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def _save_message(db_path: str, chat_id: int, role: str, content: str) -> None:
    """Salva un messaggio nella cronologia di chat_id."""
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (chat_id, role, content, timestamp) VALUES (?,?,?,?)",
            (chat_id, role, content, ts),
        )
        conn.commit()


def _clear_history(db_path: str, chat_id: int) -> None:
    """Cancella tutta la cronologia di chat_id."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
        conn.commit()


def _delete_last_assistant_messages(db_path: str, chat_id: int) -> None:
    """Rimuove i messaggi assistant non-marker successivi all'ultimo user message.

    Chiamato prima di salvare la risposta di un retry: evita che messaggi assistant
    consecutivi si accumulino nella cronologia passata all'Orchestrator.
    I marker di compattazione (prefisso _COMPACT_MARKER) non vengono toccati.
    """
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM conversations WHERE chat_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
            (chat_id,),
        ).fetchone()
        if row:
            conn.execute(
                "DELETE FROM conversations WHERE chat_id = ? AND id > ? AND role = 'assistant' AND content NOT LIKE ?",
                (chat_id, row[0], f"{_COMPACT_MARKER}%"),
            )
        conn.commit()


async def _compact_history(db_path: str, chat_id: int) -> tuple[int, int]:
    """
    Compatta i messaggi storici in un summary LLM.

    Prende tutti i messaggi non ancora compressi tranne gli ultimi 5 (finestra recente),
    li riduce con il token_optimizer, chiede al LLM un riassunto e lo salva come marker.
    I messaggi originali compressi vengono eliminati.

    Returns:
        (n_compressi, token_stimati_risparmiati) — (0, 0) se non c'era nulla da comprimere.
    """
    # Carica tutti i messaggi senza LIMIT
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, role, content FROM conversations WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        ).fetchall()

    # Escludi i marker già compressi per non ricomprimere rumore
    non_compact = [(row[0], row[1], row[2]) for row in rows
                   if not row[2].startswith(_COMPACT_MARKER)]

    # Tieni gli ultimi 5 intatti
    if len(non_compact) <= 5:
        return 0, 0

    to_compress = non_compact[:-5]
    if len(to_compress) < 3:
        return 0, 0

    ids_to_delete = [row[0] for row in to_compress]
    messages = [{"role": row[1], "content": row[2]} for row in to_compress]

    # Riduzione input con token_optimizer prima di inviare all'LLM
    try:
        from src.core.token_optimizer import reduce_content
        from src.core.llm.client import get_llm_client
    except ImportError:
        from core.token_optimizer import reduce_content   # type: ignore[no-redef]
        from core.llm.client import get_llm_client        # type: ignore[no-redef]

    llm = get_llm_client()
    budget = max(4000, (llm.n_ctx - 2000) * 3) // 2
    combined = "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in messages)
    reduced_result = reduce_content(combined, "generic/text", budget)
    history_for_llm = reduced_result.text

    chars_before = sum(len(m["content"]) for m in messages)

    system_prompt = (
        "Sei un assistente che compatta la cronologia di una conversazione. "
        "Produci un riassunto conciso dei temi trattati, delle informazioni scambiate e delle "
        "conclusioni raggiunte. "
        'Rispondi SOLO con JSON: {"summary": "testo riassunto"} '
        "Il riassunto deve essere in italiano, massimo 150 parole."
    )

    try:
        result = await llm.complete(system_prompt, history_for_llm, temperature=0.2, max_tokens=256)
        summary_text = str(result.get("summary", "")).strip()
    except Exception as exc:
        logger.warning("compact_history: LLM summary fallito — %s", exc)
        return 0, 0

    if not summary_text:
        return 0, 0

    date_str = datetime.now(timezone.utc).strftime("%-d %b %Y")
    marker_content = f"{_COMPACT_MARKER} — {date_str}]\n{summary_text}"

    chars_after = len(marker_content)
    token_saved = max(0, (chars_before - chars_after) // 4)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO conversations (chat_id, role, content, timestamp) VALUES (?,?,?,?)",
            (chat_id, "assistant", marker_content, datetime.now(timezone.utc).isoformat()),
        )
        placeholders = ",".join("?" * len(ids_to_delete))
        conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", ids_to_delete)
        conn.commit()

    logger.info(
        "compact_history: chat_id=%s — compressi %d messaggi, risparmiati ~%d token",
        chat_id, len(ids_to_delete), token_saved,
    )
    return len(ids_to_delete), token_saved


async def _notify_and_compact(db_path: str, chat_id: int, bot) -> None:
    """Invia notifica 🔴 → 🟢 ed esegue la compattazione."""
    try:
        notice = await bot.send_message(chat_id=chat_id, text="🔴 Comprimo la memoria...")
    except Exception:
        notice = None

    n_compressed, token_saved = await _compact_history(db_path, chat_id)

    if notice is not None:
        try:
            if n_compressed > 0:
                saved_str = _fmt_tokens(token_saved)
                text = f"🟢 Memoria compattata — risparmiati ~{saved_str} token"
            else:
                text = "🟢 Memoria già compatta, nessuna operazione necessaria"
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=notice.message_id,
                text=text,
            )
        except Exception:
            pass


def _md_to_html(text: str) -> str:
    """
    Converte Markdown standard (output LLM) in HTML compatibile con Telegram.

    Ordine operazioni:
      1. Estrae e protegge blocchi codice (``` ... ```)
      2. Estrae e protegge codice inline (` ... `)
      3. Escapa le entità HTML nel testo rimanente (&, <, >, ")
      4. Converte intestazioni (# … ######) → <b>
      5. Converte **bold** → <b>
      6. Converte ~~strike~~ → <s>
      7. Converte *italic* → <i>
      8. Converte _italic_ → <i>
      9. Ripristina i placeholder con i tag HTML

    I caratteri speciali Markdown (*, _, #, ~) non sono entità HTML, quindi
    sopravvivono all'escaping del punto 3 e possono essere convertiti nei
    punti successivi senza interferenze.
    """
    placeholders: list[tuple[str, str]] = []

    def _protect(tag: str) -> str:
        key = f'\x00{len(placeholders)}\x00'
        placeholders.append((key, tag))
        return key

    def _fenced(m: re.Match) -> str:
        return _protect(f'<pre>{_html.escape(m.group(1).strip())}</pre>')

    def _inline(m: re.Match) -> str:
        return _protect(f'<code>{_html.escape(m.group(1))}</code>')

    text = _FENCED_CODE_RE.sub(_fenced, text)
    text = _INLINE_CODE_RE.sub(_inline, text)
    text = _html.escape(text)
    text = _HEADING_RE.sub(lambda m: f'<b>{m.group(1)}</b>', text)
    text = _BOLD_RE.sub(lambda m: f'<b>{m.group(1)}</b>', text)
    text = _STRIKE_RE.sub(lambda m: f'<s>{m.group(1)}</s>', text)
    text = _ITALIC_STAR_RE.sub(lambda m: f'<i>{m.group(1)}</i>', text)
    text = _ITALIC_UNDER_RE.sub(lambda m: f'<i>{m.group(1)}</i>', text)

    for key, tag in placeholders:
        text = text.replace(key, tag)

    return text


# ---------------------------------------------------------------------------
# TelegramInterface
# ---------------------------------------------------------------------------

class TelegramInterface(AbstractInterface):
    """
    Interfaccia Telegram con accesso admin completo.

    Gestisce comandi, messaggi di testo, upload documenti e pulsante Riprova.
    La cronologia conversazione è persistita in SQLite per ogni chat_id.
    """

    def __init__(self) -> None:
        from config import Config

        self._config = Config
        self._db_path = Config.conversations_db_path
        self._start_time = datetime.now(timezone.utc)

        # Stato per chat: {"last_user_message": str, "last_bot_message_id": int | None}
        self._chat_state: dict[int, dict] = {}
        # Documenti in attesa di essere inclusi nella prossima richiesta
        self._pending_docs: dict[int, list[str]] = {}

        _init_db(self._db_path)

        self._app = (
            Application.builder()
            .token(Config.telegram_bot_token)
            .build()
        )
        self._register_handlers()
        logger.info("TelegramInterface: inizializzata")

    # ------------------------------------------------------------------
    # AbstractInterface
    # ------------------------------------------------------------------

    def get_permissions(self) -> list[str]:
        return ["RETRIEVE", "ANALYZE", "REASON", "GENERATE", "STORE"]

    async def send(self, target: str, message: str) -> None:
        """Converte Markdown → HTML e invia (con chunking se > 4096 caratteri)."""
        chat_id = int(target)
        html_text = _md_to_html(message)
        for chunk in _split_text(html_text):
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id, text=chunk, parse_mode="HTML"
                )
            except Exception:
                await self._app.bot.send_message(
                    chat_id=chat_id, text=_STRIP_TAGS_RE.sub("", chunk)
                )

    async def send_error(self, target: str, task_name: str, reason: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        text = f"⚠️ **Task fallito**: {task_name}\n📅 {ts}\n❌ {reason}"
        await self.send(target, text)

    # ------------------------------------------------------------------
    # Avvio
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Avvia il bot in modalità polling (bloccante)."""
        logger.info("TelegramInterface: avvio polling")
        self._app.run_polling(drop_pending_updates=True)

    # ------------------------------------------------------------------
    # Registrazione handler
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        app = self._app

        admin_id_str = self._config.telegram_admin_chat_id
        if admin_id_str:
            try:
                _admin_filter = filters.Chat(int(admin_id_str))
            except ValueError:
                logger.warning(
                    "TELEGRAM_ADMIN_CHAT_ID '%s' non è un intero valido — accesso non filtrato",
                    admin_id_str,
                )
                _admin_filter = filters.ALL
        else:
            logger.warning("TELEGRAM_ADMIN_CHAT_ID non impostato — il bot accetta messaggi da chiunque")
            _admin_filter = filters.ALL

        app.add_handler(CommandHandler("start",   self._cmd_start,   filters=_admin_filter))
        app.add_handler(CommandHandler("status",  self._cmd_status,  filters=_admin_filter))
        app.add_handler(CommandHandler("reindex", self._cmd_reindex, filters=_admin_filter))
        app.add_handler(CommandHandler("clear",   self._cmd_clear,   filters=_admin_filter))
        app.add_handler(CommandHandler("compact", self._cmd_compact, filters=_admin_filter))
        app.add_handler(CommandHandler("help",    self._cmd_help,    filters=_admin_filter))
        app.add_handler(
            MessageHandler(_admin_filter & filters.TEXT & ~filters.COMMAND, self._handle_message, block=False)
        )
        app.add_handler(
            MessageHandler(_admin_filter & filters.Document.ALL, self._handle_document)
        )
        app.add_handler(CallbackQueryHandler(self._handle_retry, pattern="^retry$", block=False))
        app.add_handler(CallbackQueryHandler(self._handle_save_synthesis, pattern="^save_synthesis$", block=False))
        app.add_handler(CallbackQueryHandler(self._handle_synthesis_noop, pattern="^synthesis_saved$", block=False))

    # ------------------------------------------------------------------
    # Comandi
    # ------------------------------------------------------------------

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = (
            "🤖 <b>Vesper</b> è attivo.\n\n"
            "Sono il tuo agente AI locale. Posso:\n"
            "• Cercare informazioni nel vault e sul web\n"
            "• Analizzare documenti e testi\n"
            "• Ragionare su problemi e scenari\n"
            "• Generare contenuti su misura\n"
            "• Salvare note e aggiornare il vault\n\n"
            "Invia un messaggio per iniziare, o carica un documento."
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id

        # LLM server
        llm_url = (
            f"http://localhost:{self._config.llm_port}/v1"
            if self._config.llm_managed
            else self._config.llm_base_url
        )
        llm_ok = await _check_llm_server(llm_url)
        llm_status = "✅ online" if llm_ok else "❌ non raggiungibile"

        # Vault
        file_count = _get_vault_file_count(self._config.vault_path)
        chunk_count = _get_chroma_doc_count()

        # Conversazione
        msg_count = _get_chat_message_count(self._db_path, chat_id)
        capsule_count = await _get_session_capsule_count(str(chat_id))

        # DB SQLite
        try:
            db_size_kb = os.path.getsize(self._db_path) // 1024
        except OSError:
            db_size_kb = 0

        # Uptime
        uptime = datetime.now(timezone.utc) - self._start_time
        total_sec = int(uptime.total_seconds())
        hours, rem = divmod(total_sec, 3600)
        minutes, seconds = divmod(rem, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        text = (
            "📊 <b>Stato Vesper</b>\n\n"
            f"🤖 <b>LLM server</b> — {llm_status}\n"
            f"📂 <b>Vault</b> — {file_count} file, {chunk_count} chunk\n"
            f"💬 <b>Conversazione</b> — {msg_count} messaggi, {capsule_count} capsule ({db_size_kb} KB)\n"
            f"⏱ <b>Uptime</b> — {uptime_str}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_reindex(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text("⏳ Indicizzazione in corso...")
        try:
            from src.core.rag.indexer import index_vault
        except ImportError:
            from core.rag.indexer import index_vault  # type: ignore[no-redef]

        try:
            report = await index_vault(force=True)
            text = (
                f"✅ Indicizzazione completata: "
                f"{report['indexed']} documenti, {report['skipped']} saltati"
            )
            if report.get("errors"):
                text += f", {len(report['errors'])} errori"
        except Exception as exc:
            text = f"❌ Indicizzazione fallita: {exc}"
            logger.error("TelegramInterface /reindex: %s", exc)

        await update.message.reply_text(text)

    async def _cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        _clear_history(self._db_path, chat_id)
        self._chat_state.pop(chat_id, None)
        self._pending_docs.pop(chat_id, None)
        try:
            from src.core.session_memory import forget as _mem_forget
        except ImportError:
            from core.session_memory import forget as _mem_forget  # type: ignore[no-redef]
        try:
            await _mem_forget(str(chat_id))
        except Exception as exc:
            logger.warning("_cmd_clear: session_memory forget fallito — %s", exc)
        await update.message.reply_text("\U0001f5d1️ Memoria conversazione cancellata.")

    async def _cmd_compact(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        async with _get_compact_lock(chat_id):
            await _notify_and_compact(self._db_path, chat_id, ctx.bot)

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        text = (
            "/start — Avvia il bot\n"
            "/status — Stato del sistema\n"
            "/reindex — Reindicizza il vault\n"
            "/compact — Compatta la memoria della conversazione\n"
            "/clear — Cancella la cronologia\n"
            "/help — Mostra questo messaggio"
        )
        await update.message.reply_text(text)

    # ------------------------------------------------------------------
    # Handler messaggi
    # ------------------------------------------------------------------

    async def _handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        user_text = update.message.text or ""

        async with _get_compact_lock(chat_id):
            # Rimuovi il pulsante Riprova dall'ultima risposta
            await self._remove_retry_button(chat_id, ctx.bot)

            # Salva messaggio utente
            _save_message(self._db_path, chat_id, "user", user_text)

            # Aggiorna stato: salva l'ultimo messaggio per il retry
            state = self._chat_state.setdefault(chat_id, {})
            state["last_user_message"] = user_text

            # Messaggio di stato "⏳"
            status_msg = await update.message.reply_text("⏳ Elaborazione in corso...")
            status_msg_id = status_msg.message_id

            req_ctx = None
            history_text: str
            try:
                response_text, history_text, req_ctx = await self._process_message(
                    chat_id=chat_id,
                    user_text=user_text,
                    bot=ctx.bot,
                    status_message_id=status_msg_id,
                )
            except Exception as exc:
                logger.error("TelegramInterface: errore per chat_id=%s — %s", chat_id, exc)
                response_text = f"❌ Si \xe8 verificato un errore: {exc}"
                history_text = response_text

            # Cancella il messaggio di stato ⏳
            try:
                await ctx.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

            # Aggiorna stato con il req_ctx corrente (usato da _handle_save_synthesis)
            if req_ctx is not None:
                state["last_req_ctx"] = req_ctx

            # Costruisce tastiera: "📝 Salva sintesi" (se web) a sinistra di "🔄 Riprova"
            _kbd_row = []
            if req_ctx is not None and self._web_synthesis_available(req_ctx):
                _kbd_row.append(InlineKeyboardButton("📝 Salva sintesi", callback_data="save_synthesis"))
            _kbd_row.append(InlineKeyboardButton("🔄 Riprova", callback_data="retry"))
            retry_markup = InlineKeyboardMarkup([_kbd_row])
            sent_messages = await _send_chunked(ctx.bot, chat_id, response_text, reply_markup=retry_markup)

            if sent_messages:
                state["last_bot_message_id"] = sent_messages[-1].message_id

            # Salva in SQLite solo l'output del Generator (senza footer di telemetria)
            _save_message(self._db_path, chat_id, "assistant", history_text)

            # Auto-compact se la cronologia supera la soglia (dentro il lock)
            if _get_chat_message_count(self._db_path, chat_id) > _COMPACT_THRESHOLD:
                await _notify_and_compact(self._db_path, chat_id, ctx.bot)

    async def _handle_document(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        doc = update.message.document
        caption = (update.message.caption or "").strip()

        max_bytes = 20 * 1024 * 1024
        if doc.file_size and doc.file_size > max_bytes:
            await update.message.reply_text(
                "❌ File troppo grande (max 20 MB). Invia un file piu' piccolo."
            )
            return

        filename = doc.file_name or f"upload_{doc.file_unique_id}.bin"
        dest = self._config.vault_path / "raw" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            tg_file = await doc.get_file()
            await tg_file.download_to_drive(dest)
        except Exception as exc:
            logger.error("TelegramInterface: download fallito — %s", exc)
            await update.message.reply_text(f"❌ Download fallito: {exc}")
            return

        # Accoda il path per la prossima richiesta
        self._pending_docs.setdefault(chat_id, []).append(str(dest))

        if not caption:
            # Nessun caption: notifica e attendi il prossimo messaggio
            await update.message.reply_text(
                f"📎 <b>{_html.escape(filename)}</b> salvato nel vault.\n"
                "Verrà incluso nella tua prossima richiesta.",
                parse_mode="HTML",
            )
            return

        # Caption presente: processa subito come messaggio utente (il path è già in pending_docs)
        async with _get_compact_lock(chat_id):
            await self._remove_retry_button(chat_id, ctx.bot)
            _save_message(self._db_path, chat_id, "user", caption)

            state = self._chat_state.setdefault(chat_id, {})
            state["last_user_message"] = caption

            status_msg = await update.message.reply_text("⏳ Elaborazione in corso...")

            req_ctx = None
            history_text: str
            try:
                response_text, history_text, req_ctx = await self._process_message(
                    chat_id=chat_id,
                    user_text=caption,
                    bot=ctx.bot,
                    status_message_id=status_msg.message_id,
                )
            except Exception as exc:
                logger.error("TelegramInterface: errore doc+caption per chat_id=%s — %s", chat_id, exc)
                response_text = f"❌ Si è verificato un errore: {exc}"
                history_text = response_text

            try:
                await ctx.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
            except Exception:
                pass

            if req_ctx is not None:
                state["last_req_ctx"] = req_ctx

            _kbd_row = []
            if req_ctx is not None and self._web_synthesis_available(req_ctx):
                _kbd_row.append(InlineKeyboardButton("📝 Salva sintesi", callback_data="save_synthesis"))
            _kbd_row.append(InlineKeyboardButton("🔄 Riprova", callback_data="retry"))
            retry_markup = InlineKeyboardMarkup([_kbd_row])
            sent_messages = await _send_chunked(ctx.bot, chat_id, response_text, reply_markup=retry_markup)

            if sent_messages:
                state["last_bot_message_id"] = sent_messages[-1].message_id

            _save_message(self._db_path, chat_id, "assistant", history_text)

            if _get_chat_message_count(self._db_path, chat_id) > _COMPACT_THRESHOLD:
                await _notify_and_compact(self._db_path, chat_id, ctx.bot)

    async def _handle_retry(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        # Ack immediato: Telegram richiede risposta entro pochi secondi dal click.
        await query.answer()

        chat_id = update.effective_chat.id
        state = self._chat_state.get(chat_id, {})
        last_message = state.get("last_user_message")

        if not last_message:
            await query.edit_message_reply_markup(reply_markup=None)
            await ctx.bot.send_message(chat_id=chat_id, text="Nessun messaggio da ripetere.")
            return

        # Rimuovi il pulsante prima di acquisire il lock: è solo una chiamata API Telegram,
        # non tocca lo stato locale, e risponde subito all'utente.
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        async with _get_compact_lock(chat_id):
            status_msg = await ctx.bot.send_message(chat_id=chat_id, text="⏳ Elaborazione in corso...")
            status_msg_id = status_msg.message_id

            req_ctx = None
            history_text: str
            try:
                response_text, history_text, req_ctx = await self._process_message(
                    chat_id=chat_id,
                    user_text=last_message,
                    bot=ctx.bot,
                    status_message_id=status_msg_id,
                )
            except Exception as exc:
                logger.error("TelegramInterface retry: errore per chat_id=%s — %s", chat_id, exc)
                response_text = f"❌ Si \xe8 verificato un errore: {exc}"
                history_text = response_text

            try:
                await ctx.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

            if req_ctx is not None:
                state["last_req_ctx"] = req_ctx

            _kbd_row = []
            if req_ctx is not None and self._web_synthesis_available(req_ctx):
                _kbd_row.append(InlineKeyboardButton("📝 Salva sintesi", callback_data="save_synthesis"))
            _kbd_row.append(InlineKeyboardButton("🔄 Riprova", callback_data="retry"))
            retry_markup = InlineKeyboardMarkup([_kbd_row])
            sent_messages = await _send_chunked(ctx.bot, chat_id, response_text, reply_markup=retry_markup)

            if sent_messages:
                state["last_bot_message_id"] = sent_messages[-1].message_id

            # Sostituisce la risposta precedente invece di appendere: rimuove tutti i messaggi
            # assistant non-marker successivi all'ultimo user, poi inserisce la nuova risposta.
            _delete_last_assistant_messages(self._db_path, chat_id)
            _save_message(self._db_path, chat_id, "assistant", history_text)

            if _get_chat_message_count(self._db_path, chat_id) > _COMPACT_THRESHOLD:
                await _notify_and_compact(self._db_path, chat_id, ctx.bot)

    async def _handle_save_synthesis(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Salva la ricerca web corrente come nota in vault/synthesis/."""
        query = update.callback_query
        await query.answer()

        chat_id = update.effective_chat.id
        state = self._chat_state.get(chat_id, {})
        req_ctx = state.get("last_req_ctx")

        if req_ctx is None:
            # Memoria cancellata: segnala che l'azione non è più disponibile
            try:
                await query.edit_message_reply_markup(
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("⚠️ Non disponibile", callback_data="synthesis_saved"),
                        InlineKeyboardButton("🔄 Riprova", callback_data="retry"),
                    ]])
                )
            except Exception:
                try:
                    await ctx.bot.send_message(
                        chat_id=chat_id,
                        text="⚠️ Sintesi non più disponibile: la memoria è stata cancellata.",
                    )
                except Exception:
                    pass
            return

        # Aggiorna subito il pulsante in modo che non triggeri altre azioni
        try:
            await query.edit_message_reply_markup(
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("✅ Ricerca salvata", callback_data="synthesis_saved"),
                    InlineKeyboardButton("🔄 Riprova", callback_data="retry"),
                ]])
            )
        except Exception:
            pass

        try:
            from src.core.manager import Manager
        except ImportError:
            from core.manager import Manager  # type: ignore[no-redef]

        mgr = Manager()
        saved_path = await mgr.save_web_synthesis(req_ctx)
        if saved_path:
            logger.info("TelegramInterface: sintesi web salvata — '%s'", saved_path)
        else:
            logger.warning("TelegramInterface: save_web_synthesis non ha prodotto risultati")

    async def _handle_synthesis_noop(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Risponde al tap su un pulsante sintesi già salvato (inerte)."""
        await update.callback_query.answer()

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def _process_message(
        self,
        chat_id: int,
        user_text: str,
        bot: Bot,
        status_message_id: int,
    ) -> tuple[str, str, object]:
        """
        Delega la pipeline a process_request() e aggiunge il footer Telegram.

        Returns:
            Tupla (display_text, history_text, req_ctx):
            - display_text include il footer di telemetria (fonti, stats) da mostrare all'utente
            - history_text è il solo output del Generator, senza footer, da salvare in SQLite
            - req_ctx è il RequestContext completato
        """
        async def status_callback(text: str) -> None:
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message_id,
                    text=f"⏳ {text}",
                )
            except Exception:
                pass

        history = _load_history(self._db_path, chat_id)
        pending = self._pending_docs.pop(chat_id, [])

        # Costruisce effective_user_text con i path vault-relativi dei documenti caricati,
        # così l'Orchestrator può pianificare ANALYZE/RETRIEVE con il path corretto.
        effective_user_text: str | None = None
        if pending:
            vault_path = self._config.vault_path
            rel_paths: list[str] = []
            for full_path in pending:
                try:
                    rel = Path(full_path).relative_to(vault_path)
                    rel_paths.append(str(rel).replace("\\", "/"))
                except ValueError:
                    rel_paths.append(full_path)
            doc_refs = ", ".join(f'"{p}"' for p in rel_paths)
            effective_user_text = f"{user_text}\n\n[Documento caricato: {doc_refs}]"

        response_text, req_ctx, stats = await self.process_request(
            user_text=user_text,
            history=history,
            session_id=str(chat_id),
            status_callback=status_callback,
            effective_user_text=effective_user_text,
            extra_context={"uploaded_documents": pending} if pending else None,
        )

        # history_text: output puro del Generator, senza footer — va in SQLite.
        # Il footer è telemetria per l'utente, non contesto utile all'Orchestrator.
        history_text = response_text or "Elaborazione completata."
        display_text = self._build_display_text(response_text, req_ctx, stats)

        return display_text, history_text, req_ctx

    def _build_display_text(self, response_text: str, req_ctx, stats: dict) -> str:
        """
        Compone il testo da mostrare all'utente aggiungendo il footer di telemetria.

        Il footer elenca le fonti consultate (vault, web, documenti letti/scritti)
        e le statistiche di inferenza. È specifico di Telegram e non va in SQLite.
        """
        consultati: list[str] = []
        fonti_web: list[str] = []
        aggiornati: list[str] = []
        creati: list[str] = []
        seen_consultati: set[str] = set()
        seen_web: set[str] = set()

        for tc in req_ctx.tools_called:
            tool = tc.get("tool", "")
            if tool == "vault_search":
                for r in (tc.get("output") or []):
                    p = r.get("path")
                    if p and p not in seen_consultati:
                        seen_consultati.add(p)
                        consultati.append(f"  · {p}")
            elif tool == "web_search":
                for r in (tc.get("output") or []):
                    url = r.get("url")
                    if url and url not in seen_web:
                        seen_web.add(url)
                        fonti_web.append(f"  · {url}")
            elif tool == "read_document":
                p = (tc.get("input") or {}).get("path")
                if p and p not in seen_consultati:
                    seen_consultati.add(p)
                    consultati.append(f"  · {p}")
            elif tool == "update_document":
                p = (tc.get("input") or {}).get("path")
                if p:
                    aggiornati.append(f"  · {p}")
            elif tool == "write_document":
                p = (tc.get("input") or {}).get("path")
                if p:
                    creati.append(f"  · {p}")

        footer_parts: list[str] = []
        if consultati:
            footer_parts.append("Consultati:\n" + "\n".join(consultati))
        if fonti_web:
            footer_parts.append("Fonti web:\n" + "\n".join(fonti_web))
        if aggiornati:
            footer_parts.append("Aggiornati:\n" + "\n".join(aggiornati))
        if creati:
            footer_parts.append("Creati:\n" + "\n".join(creati))
        for sp in (p for p in getattr(req_ctx, "documents_modified", [])
                   if isinstance(p, str) and p.startswith("synthesis/")):
            footer_parts.append(f"📝 Sintesi salvata: `{sp}`")
        if stats:
            footer_parts.append(
                f"`{_fmt_tokens(stats['completion_tokens'])} tokens · "
                f"{stats['elapsed_sec']}s · "
                f"{stats['tokens_per_sec']} t/s`"
            )

        base = response_text or "Elaborazione completata."
        if footer_parts:
            return base + "\n\n" + "\n".join(footer_parts)
        return base

    # ------------------------------------------------------------------
    # Helpers privati
    # ------------------------------------------------------------------

    async def _remove_retry_button(self, chat_id: int, bot: Bot) -> None:
        """Rimuove la tastiera inline dall'ultima risposta del bot, se presente."""
        state = self._chat_state.get(chat_id, {})
        last_id = state.get("last_bot_message_id")
        if last_id is None:
            return
        try:
            await bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=last_id, reply_markup=None
            )
        except Exception:
            pass
        state.pop("last_bot_message_id", None)


# ---------------------------------------------------------------------------
# Helpers di modulo
# ---------------------------------------------------------------------------

def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _split_text(text: str) -> list[str]:
    """
    Divide il testo in chunk da al massimo _TG_MAX_CHARS caratteri.

    Preferisce spezzare tra paragrafi (\\n\\n), poi tra righe (\\n),
    poi tra parole (spazio). Taglia esatto solo come ultima risorsa,
    minimizzando il rischio di spezzare tag HTML a metà.
    """
    if len(text) <= _TG_MAX_CHARS:
        return [text]

    chunks: list[str] = []
    while len(text) > _TG_MAX_CHARS:
        cut = -1
        for sep in ('\n\n', '\n', ' '):
            pos = text.rfind(sep, 0, _TG_MAX_CHARS)
            if pos > 0:
                cut = pos + len(sep)
                break
        if cut <= 0:
            cut = _TG_MAX_CHARS
        chunks.append(text[:cut].rstrip('\n'))
        text = text[cut:].lstrip('\n')

    if text:
        chunks.append(text)
    return chunks


async def _send_chunked(
    bot: Bot,
    chat_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> list[Message]:
    """
    Converte Markdown → HTML, poi invia in uno o più chunk da max _TG_MAX_CHARS.

    Il reply_markup viene aggiunto all'ultimo chunk.
    Fallback: testo plain (tag HTML rimossi) se parse_mode="HTML" fallisce.

    Returns:
        Lista dei messaggi inviati.
    """
    html_text = _md_to_html(text)
    chunks = _split_text(html_text)
    sent: list[Message] = []

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        markup = reply_markup if is_last else None

        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode="HTML",
                reply_markup=markup,
            )
        except Exception:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=_STRIP_TAGS_RE.sub("", chunk),
                reply_markup=markup,
            )
        sent.append(msg)

    return sent


async def _get_session_capsule_count(session_id: str) -> int:
    """Restituisce il numero di capsule session_memory per questa chat, o 0 in caso di errore."""
    try:
        from src.core.session_memory import count as _mem_count
    except ImportError:
        try:
            from core.session_memory import count as _mem_count  # type: ignore[no-redef]
        except ImportError:
            return 0
    try:
        return await _mem_count(session_id)
    except Exception:
        return 0


def _get_chroma_doc_count() -> int:
    """Restituisce il numero di chunk indicizzati in ChromaDB, o 0 in caso di errore."""
    try:
        from src.core.rag.indexer import get_chroma_client
    except ImportError:
        try:
            from core.rag.indexer import get_chroma_client  # type: ignore[no-redef]
        except ImportError:
            return 0

    try:
        return get_chroma_client().count()
    except Exception:
        return 0


def _get_vault_file_count(vault_path: Path) -> int:
    """Conta i file .md presenti in vault/wiki/ su disco."""
    try:
        return len(list((vault_path / "wiki").rglob("*.md")))
    except Exception:
        return 0


def _get_chat_message_count(db_path: str, chat_id: int) -> int:
    """Restituisce il totale dei messaggi salvati per chat_id."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


async def _check_llm_server(base_url: str) -> bool:
    """Pinga il server LLM, restituisce True se raggiungibile."""
    import httpx
    url = base_url.rstrip("/").removesuffix("/v1") + "/v1/models"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=2.0)
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Test minimale
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

    print("\n=== _split_text ===\n")

    check("testo corto -> lista singola", _split_text("ciao") == ["ciao"])
    check("testo vuoto -> lista singola", _split_text("") == [""])

    long_text = "x" * 9000
    chunks = _split_text(long_text)
    check("testo 9000 car -> 3 chunks", len(chunks) == 3)
    check("ogni chunk <= 4096 car", all(len(c) <= _TG_MAX_CHARS for c in chunks))
    check("contenuto preservato (no separatori)", "".join(chunks) == long_text)

    # Split preferisce il confine di paragrafo
    para_text = ("A" * 3000 + "\n\n") * 3
    para_chunks = _split_text(para_text)
    check("split a paragrafo: ogni chunk <= 4096", all(len(c) <= _TG_MAX_CHARS for c in para_chunks))
    # Il join non preserva i newline rimossi da strip, ma il testo significativo deve esserci
    joined = "".join(para_chunks)
    check("split a paragrafo: testo significativo preservato",
          joined.replace("\n", "") == para_text.replace("\n", ""))

    print("\n=== _md_to_html ===\n")

    check("bold **text**",
          _md_to_html("**grassetto**") == "<b>grassetto</b>")
    check("italic *text*",
          _md_to_html("*corsivo*") == "<i>corsivo</i>")
    check("italic _text_",
          _md_to_html("parola _corsivo_ fine") == "parola <i>corsivo</i> fine")
    check("heading ## -> bold",
          _md_to_html("## Titolo") == "<b>Titolo</b>")
    check("inline code",
          _md_to_html("`codice`") == "<code>codice</code>")
    check("code block",
          "<pre>" in _md_to_html("```\ncodice\n```"))
    check("html entities escaped",
          _md_to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d")
    check("bold non tocca < >",
          _md_to_html("**a < b**") == "<b>a &lt; b</b>")
    check("code block: contenuto escaped",
          _md_to_html("```\na < b\n```") == "<pre>a &lt; b</pre>")
    check("testo plain invariato (solo escape)",
          _md_to_html("ciao mondo") == "ciao mondo")
    check("strikethrough",
          _md_to_html("~~barrato~~") == "<s>barrato</s>")

    print("\n=== SQLite helpers ===\n")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = str(Path(tmp) / "test.db")
        _init_db(db)

        # Tabella creata
        with sqlite3.connect(db) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        check("tabella conversations creata", "conversations" in tables)

        # save + load
        _save_message(db, 42, "user", "ciao Vesper")
        _save_message(db, 42, "assistant", "ciao utente")
        history = _load_history(db, 42)
        check("storia caricata: 2 messaggi", len(history) == 2)
        check("ordine cronologico", history[0]["role"] == "user")
        check("contenuto corretto", history[0]["content"] == "ciao Vesper")

        # chat_id diverso non interferisce
        _save_message(db, 99, "user", "altro utente")
        history42 = _load_history(db, 42)
        check("chat separati: chat 42 ha ancora 2 msg", len(history42) == 2)

        # clear
        _clear_history(db, 42)
        cleared = _load_history(db, 42)
        check("clear: storia vuota dopo cancellazione", len(cleared) == 0)
        still_there = _load_history(db, 99)
        check("clear: altra chat non toccata", len(still_there) == 1)

        # HISTORY_LIMIT: non restituisce piu' di _HISTORY_LIMIT messaggi
        for i in range(_HISTORY_LIMIT + 5):
            _save_message(db, 7, "user", f"msg {i}")
        limited = _load_history(db, 7)
        check(
            f"history limit: max {_HISTORY_LIMIT} messaggi restituiti",
            len(limited) <= _HISTORY_LIMIT,
        )
        check("history limit: sono gli ultimi", limited[-1]["content"] == f"msg {_HISTORY_LIMIT + 4}")

    print("\n=== _delete_last_assistant_messages ===\n")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = str(Path(tmp) / "test_retry.db")
        _init_db(db)

        # Caso base: user → assistant → retry → deve restare solo 1 assistant
        _save_message(db, 1, "user", "domanda")
        _save_message(db, 1, "assistant", "risposta 1")
        _delete_last_assistant_messages(db, 1)
        _save_message(db, 1, "assistant", "risposta 2")
        h = _load_history(db, 1)
        check("retry singolo: 2 messaggi totali", len(h) == 2)
        check("retry singolo: l'assistant è la nuova risposta", h[-1]["content"] == "risposta 2")

        # Retry multipli consecutivi: non deve accumulare assistant
        _delete_last_assistant_messages(db, 1)
        _save_message(db, 1, "assistant", "risposta 3")
        _delete_last_assistant_messages(db, 1)
        _save_message(db, 1, "assistant", "risposta 4")
        h = _load_history(db, 1)
        check("retry multipli: ancora 2 messaggi totali", len(h) == 2)
        check("retry multipli: ultima risposta è la più recente", h[-1]["content"] == "risposta 4")

        # I marker di compattazione non vengono eliminati
        _save_message(db, 2, "user", "u1")
        _save_message(db, 2, "assistant", "r1")
        _save_message(db, 2, "user", "u2")
        _save_message(db, 2, "assistant", f"{_COMPACT_MARKER} — 1 Jan 2025]\nriassunto")
        _save_message(db, 2, "assistant", "r2")
        _delete_last_assistant_messages(db, 2)
        h2 = _load_history(db, 2)
        marker_present = any(_COMPACT_MARKER in m["content"] for m in h2)
        check("marker compattazione non eliminato", marker_present)
        non_marker_after_u2 = [m for m in h2 if m["role"] == "assistant" and _COMPACT_MARKER not in m["content"]
                               and h2.index(m) > next(i for i, m2 in enumerate(h2) if m2["content"] == "u2")]
        check("assistant non-marker dopo ultimo user rimosso", len(non_marker_after_u2) == 0)

        # Nessun user in history: nessuna eliminazione (non si tocca niente)
        _save_message(db, 3, "assistant", "senza user")
        _delete_last_assistant_messages(db, 3)
        h3 = _load_history(db, 3)
        check("nessun user: nessuna eliminazione", len(h3) == 1)

    print("\n=== resolve_target (import da base) ===\n")

    try:
        from src.interfaces.base import resolve_target
    except ImportError:
        from base import resolve_target  # type: ignore[no-redef]

    s, t = resolve_target("telegram://123")
    check("resolve_target: schema telegram", s == "telegram")
    check("resolve_target: target_id 123", t == "123")

    print(f"\nRisultato: {passed} OK, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)
