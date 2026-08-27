"""SQLite-backed session retrieval mirroring the TS ``session-query`` family.

A durable FTS5 index over derived session messages lets ``session.search``
answer across cold sessions without loading every log.  The index is an
accelerator: callers fall back to in-memory search whenever the index is
absent, empty, or the platform sqlite lacks FTS5.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_session ON documents(session_id);
"""


class SessionSearchIndex:
    """FTS5 corpus over session messages; safe on sqlite builds without FTS5."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._fts_available = self._prepare()

    def _prepare(self) -> bool:
        try:
            with self._connection:
                self._connection.executescript(_SCHEMA)
                row = self._connection.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('documents')"
                ).fetchone()
                if row is None or row[0] == 0:
                    return False
                self._connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5("
                    "session_id UNINDEXED, seq UNINDEXED, role UNINDEXED, text)"
                )
            return True
        except sqlite3.Error:
            return False

    @property
    def available(self) -> bool:
        return self._fts_available

    def index_session(self, session_id: str, documents: list[tuple[int, str, str]]) -> None:
        """Replace one session's documents with ``(seq, role, text)`` rows."""

        if not self._fts_available:
            return
        with self._connection:
            self._connection.execute(
                "DELETE FROM search WHERE session_id = ?", (session_id,)
            )
            self._connection.execute(
                "DELETE FROM documents WHERE session_id = ?", (session_id,)
            )
            self._connection.executemany(
                "INSERT INTO documents (session_id, seq, role, text) VALUES (?, ?, ?, ?)",
                [(session_id, seq, role, text) for seq, role, text in documents],
            )
            self._connection.executemany(
                "INSERT INTO search (session_id, seq, role, text) VALUES (?, ?, ?, ?)",
                [(session_id, seq, role, text) for seq, role, text in documents],
            )

    def remove_session(self, session_id: str) -> None:
        if not self._fts_available:
            return
        with self._connection:
            self._connection.execute(
                "DELETE FROM search WHERE session_id = ?", (session_id,)
            )
            self._connection.execute(
                "DELETE FROM documents WHERE session_id = ?", (session_id,)
            )

    def session_count(self) -> int:
        if not self._fts_available:
            return 0
        row = self._connection.execute(
            "SELECT COUNT(DISTINCT session_id) FROM documents"
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, str]]:
        """Match the corpus with FTS5 ``MATCH``; a malformed query yields no rows.

        Each session contributes its best-ranked document.  Whitespace terms
        are quoted individually so punctuation like ``same.txt`` stays a
        phrase instead of FTS5 syntax.
        """

        needle = query.strip()
        if not needle or not self._fts_available:
            return []
        match_query = " ".join(f'"{term}"' for term in needle.split())
        try:
            best = self._connection.execute(
                "SELECT session_id, rowid, min(rank) FROM search "
                "WHERE search MATCH ? GROUP BY session_id ORDER BY min(rank) LIMIT ?",
                (match_query, limit),
            ).fetchall()
            hits: list[dict[str, str]] = []
            for session_id, rowid, _rank in best:
                row = self._connection.execute(
                    "SELECT snippet(search, 3, '[', ']', '…', 12) FROM search WHERE rowid = ?",
                    (rowid,),
                ).fetchone()
                hits.append(
                    {"sessionId": str(session_id), "snippet": str(row[0]) if row else ""}
                )
            return hits
        except sqlite3.Error:
            return []

    def close(self) -> None:
        self._connection.close()


def documents_from_messages(
    messages: list[tuple[str, str]],
) -> list[tuple[int, str, str]]:
    """Convert ``(role, text)`` message pairs into indexed document rows."""

    return [
        (index, role, text)
        for index, (role, text) in enumerate(messages)
        if text
    ]


__all__ = ["SessionSearchIndex", "documents_from_messages"]
