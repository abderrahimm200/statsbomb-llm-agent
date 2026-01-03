from __future__ import annotations

import glob
import io
import json
import traceback
import uuid
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import pandas as pd

from langchain_core.tools import tool

from .backend import PLOT_DIR
from .db import run_sql_readonly, table_info_with_sample, ensure_limit, sanitize_sql, db_ro_uri, validate_tables

# In-memory dataframe store (plot tool accesses these by name)
DF_STORE: Dict[str, pd.DataFrame] = {}


@tool
def table_columns(table_name: str) -> Dict[str, Any]:
    """Return schema + 2 random sample rows for a table (games/events/lineups/column_map)."""
    return table_info_with_sample(table_name)


@tool
def sql_query(sql: str, limit: int = 500) -> Dict[str, Any]:
    """Run ONE read-only SQL query against AFCON SQLite (games/events/lineups/column_map)."""
    return run_sql_readonly(sql, limit=limit)


@tool
def sql_to_df(sql: str, name: str = "df", limit: int = 50000) -> Dict[str, Any]:
    """
    Run read-only SQL and store result as a pandas DataFrame in memory under `name`.
    Returns preview + schema info (keeps debug useful but not massive).
    """
    import sqlite3

    clean, err = sanitize_sql(sql)
    if err:
        return err

    # ✅ validate tables (same safety as run_sql_readonly)
    tbl_err = validate_tables(clean)
    if tbl_err:
        return tbl_err

    clean_limited = ensure_limit(clean, limit)
    
    try:
        conn = sqlite3.connect(db_ro_uri(), uri=True)
        df = pd.read_sql_query(clean_limited, conn)
        conn.close()
    except Exception as e:
        return {"ok": False, "error": traceback.format_exc()}
    
    DF_STORE[name] = df
    return {
        "name": name,
        "sql_executed": clean,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "preview_rows": df.head(10).to_dict(orient="records"),
    }


@tool
def python_viz(code: str, dataframes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Execute plotting code with access to named DataFrames.
    - Saves any open matplotlib figures into data/plots/
    - Returns image paths + stdout + error (if any)

    banned imports :
        "import os", "import sys", "subprocess", "pip ", "apt ",
        "requests", "urllib", "open(", "pathlib", "shutil
    and similar imports
    """ 
    dataframes = dataframes or []

    banned = [
        "import os", "import sys", "subprocess", "pip ", "apt ",
        "requests", "urllib", "open(", "pathlib", "shutil",
    ]
    if any(b in code.lower() for b in banned):
        return {"ok": False, "error": f"Blocked: disallowed operation/import. these and similar imports are not allowed: {banned}", "images": [], "stdout": ""}

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import io
    import json
    from contextlib import redirect_stdout

    try:
        from mplsoccer import Pitch, VerticalPitch
    except Exception:
        Pitch = None
        VerticalPitch = None

    env: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "plt": plt,
        "json": json,
        "Pitch": Pitch,
        "VerticalPitch": VerticalPitch,
    }
    for n in dataframes:
        if n in DF_STORE:
            env[n] = DF_STORE[n]

    stdout_buf = io.StringIO()

    try:
        plt.close("all")

        with redirect_stdout(stdout_buf):
            exec(code, env, env)

        images: List[bytes] = []

        # Convert each open figure into PNG bytes
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            buf.seek(0)
            images.append(buf.read())
            plt.close(fig)

        if not images:
            return {"ok": False, "error": "No figures created.", "images": [], "stdout": stdout_buf.getvalue()}

        return {"ok": True, "error": None, "images": images, "stdout": stdout_buf.getvalue()}

    except Exception:
        return {"ok": False, "error": traceback.format_exc(), "images": [], "stdout": stdout_buf.getvalue()}


@tool
def search_mplsoccer_docs(query: str) -> Any:
    """
    Optional helper: searches mplsoccer docs. If networking isn't available, returns a helpful error string.
    """
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    except Exception:
        return "search_mplsoccer_docs unavailable: install langchain-community (or remove this tool)."

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    refined = f"site:mplsoccer.readthedocs.io {query}"
    try:
        return wrapper.results(refined, max_results=3)
    except Exception as e:
        return f"search failed: {e}"


def get_tools():
    # Keep tools list centralized for the agent
    return [table_columns, sql_query, sql_to_df, python_viz, search_mplsoccer_docs]
