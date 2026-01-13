# afcon_app/db.py
from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple

from .backend import DB_PATH

# ----------------------------
# SQL safety / parsing
# ----------------------------
FORBIDDEN = re.compile(
    r"\b(attach|detach|pragma|vacuum|reindex|analyze|"
    r"insert|update|delete|drop|create|alter|replace|truncate)\b",
    re.IGNORECASE,
)
FROM_JOIN = re.compile(r"\b(from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", re.IGNORECASE)

ALLOWED_TABLES = {"games", "events", "lineups", "column_map"}


def db_ro_uri() -> str:
    # SQLite read-only URI
    return f"file:{DB_PATH}?mode=ro"


def _strip_code_fences(sql: str) -> str:
    s = (sql or "").strip()
    s = re.sub(r"^```(\w+)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    return s


def _tables_used(sql: str) -> Set[str]:
    """Return real table names used in FROM/JOIN excluding CTE names."""
    all_found = {m.group(2).lower() for m in FROM_JOIN.finditer(sql)}
    cte_names = {
        m.group(1).lower()
        for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql, re.IGNORECASE)
    }
    return all_found - cte_names


def ensure_limit(sql: str, limit: int) -> str:
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {int(limit)}"


def sanitize_sql(sql: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Returns (clean_sql, error_dict).
    Never raises.

    error_dict shape:
      {"ok": False, "error_type": "...", "message": "...", "hint": "...", "details": {...}}
    """
    s = _strip_code_fences(sql)

    if not s:
        return None, {
            "ok": False,
            "error_type": "EMPTY_SQL",
            "message": "SQL is empty.",
            "hint": "Provide a single SELECT or WITH query.",
            "details": {},
        }

    # Block multi-statement: allow ONE trailing semicolon only
    if ";" in s.rstrip(";"):
        return None, {
            "ok": False,
            "error_type": "MULTI_STATEMENT",
            "message": "Multiple statements are not allowed. Use ONE SELECT/WITH statement only.",
            "hint": "Remove extra semicolons and keep a single query.",
            "details": {"sql_preview": s[:500]},
        }

    head = s.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        return None, {
            "ok": False,
            "error_type": "NOT_READONLY",
            "message": "Only SELECT/WITH queries are allowed.",
            "hint": "Rewrite as a SELECT or WITH query.",
            "details": {"sql_preview": s[:200]},
        }

    if FORBIDDEN.search(s):
        return None, {
            "ok": False,
            "error_type": "FORBIDDEN_KEYWORD",
            "message": "Forbidden keyword detected (write operations are blocked).",
            "hint": "Remove any DDL/DML keywords and use SELECT-only queries.",
            "details": {"sql_preview": s[:500]},
        }

    clean = s.rstrip(";").strip()
    return clean, None


def validate_tables(clean_sql: str) -> Optional[Dict[str, Any]]:
    """
    Returns error_dict or None (ok).
    Never raises.
    """
    used = _tables_used(clean_sql)
    if used and not used.issubset(ALLOWED_TABLES):
        return {
            "ok": False,
            "error_type": "DISALLOWED_TABLE",
            "message": f"Query uses disallowed tables: {sorted(used - ALLOWED_TABLES)}",
            "hint": f"Allowed tables: {sorted(ALLOWED_TABLES)}",
            "details": {"tables_used": sorted(used), "allowed_tables": sorted(ALLOWED_TABLES)},
        }
    return None


# ----------------------------
# Public API (never raises)
# ----------------------------
def run_sql_readonly(sql: str, limit: int = 500) -> Dict[str, Any]:
    """
    Execute ONE read-only SQL query (SELECT/WITH).
    Always returns a dict.

    Success:
      {"ok": True, "sql_executed": "...", "row_count": N, "rows": [...]}  # rows is list[dict]

    Error:
      {"ok": False, "error_type": "...", "message": "...", "hint": "...", "details": {...}}
    """
    clean, err = sanitize_sql(sql)
    if err:
        return err

    tbl_err = validate_tables(clean)
    if tbl_err:
        return tbl_err

    clean_limited = ensure_limit(clean, limit)

    try:
        conn = sqlite3.connect(db_ro_uri(), uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(clean_limited).fetchall()
            out_rows = [dict(r) for r in rows]
        finally:
            conn.close()

        return {
            "ok": True,
            "sql_executed": clean_limited,
            "row_count": len(out_rows),
            "rows": out_rows,
        }

    except sqlite3.Error as e:
        # Let the agent see SQLite's message to self-correct
        return {
            "ok": False,
            "error_type": "SQLITE_ERROR",
            "message": str(e),
            "hint": "Check column names via table_columns(tool) and ensure tables/aliases are correct.",
            "details": {"sql_executed": clean_limited},
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": "UNEXPECTED_ERROR",
            "message": str(e),
            "hint": "Unexpected error. Try simplifying the query.",
            "details": {"sql_executed": clean_limited},
        }


def table_info_with_sample(table_name: str) -> Dict[str, Any]:
    """
    Returns schema + 2 random sample rows.
    Never raises; returns error dict on failure.

    Success:
      {"ok": True, "table": "...", "columns": [{"name":..,"type":..},...], "sample_rows":[...]}
    """
    table = (table_name or "").strip().lower()
    if not table:
        return {
            "ok": False,
            "error_type": "EMPTY_TABLE_NAME",
            "message": "Table name is empty.",
            "hint": f"Use one of: {sorted(ALLOWED_TABLES)}",
            "details": {},
        }

    if table not in ALLOWED_TABLES:
        return {
            "ok": False,
            "error_type": "UNKNOWN_TABLE",
            "message": f"Unknown table: {table}",
            "hint": f"Allowed tables: {sorted(ALLOWED_TABLES)}",
            "details": {"allowed_tables": sorted(ALLOWED_TABLES)},
        }

    try:
        conn = sqlite3.connect(db_ro_uri(), uri=True)
        try:
            cols = conn.execute(f"PRAGMA table_info({table});").fetchall()
            columns = [{"name": c[1], "type": c[2]} for c in cols]

            sample_rows_raw = conn.execute(
                f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 2;"
            ).fetchall()

            col_names = [c["name"] for c in columns]
            sample_rows = [dict(zip(col_names, row)) for row in sample_rows_raw]

        finally:
            conn.close()

        return {
            "ok": True,
            "table": table,
            "columns": columns,
            "sample_rows": sample_rows,
        }

    except sqlite3.Error as e:
        return {
            "ok": False,
            "error_type": "SQLITE_ERROR",
            "message": str(e),
            "hint": "Ensure the database exists and is readable.",
            "details": {"table": table},
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": "UNEXPECTED_ERROR",
            "message": str(e),
            "hint": "Unexpected error. Try rebuilding the DB.",
            "details": {"table": table},
        }


def is_db_ready() -> Dict[str, Any]:
    """
    Convenience check for Streamlit before running queries.
    Never raises.
    """
    if not DB_PATH.exists():
        return {
            "ok": False,
            "error_type": "DB_MISSING",
            "message": f"Database not found at {DB_PATH}",
            "hint": "Build the database first.",
            "details": {"db_path": str(DB_PATH)},
        }
    return {"ok": True, "db_path": str(DB_PATH)}
