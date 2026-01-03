from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests

# ----------------------------
# Paths / dataset constants
# ----------------------------
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
AFCON_COMP_ID = 1267
AFCON_SEASON_ID = 107

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sb_open_data"
DB_PATH = PROJECT_ROOT / "data" / "afcon.sqlite"
PLOT_DIR = PROJECT_ROOT / "data" / "plots"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# System prompt (kept close to original, but you can trim later)
# ----------------------------
SYSTEM_PROMPT = f"""
You are a football analytics SQL agent for AFCON 2023 StatsBomb Open Data stored in SQLite.

You must ground all factual answers in the database using the provided tools. Do not guess match_ids, counts, or columns.

============================================================
DATABASE TABLES (description + columns)
============================================================

1) games — Match-level metadata (use this first to resolve match_id).
   Columns:
   - match_id
   - match_date
   - kick_off
   - home_team__home_team_name
   - away_team__away_team_name
   - home_score
   - away_score
   - competition__competition_name
   - season__season_name
   - competition_stage__name

2) events — One wide event table containing ALL flattened StatsBomb event fields for all matches.
   Guaranteed core columns:
   - event_id
   - match_id
   - event_index        (canonical per-match event order; required for sequences)
   - raw_json
   Common useful columns (often present; confirm via table_columns if unsure):
   - type__name
   - team__name
   - player__name
   - period
   - timestamp
   - minute
   - second
   - possession
   - possession_team__name
   - location
   - pass__end_location
   - pass__outcome__name
   - pass__recipient__name
   - shot__outcome__name
   - shot__statsbomb_xg

3) lineups — One row per player per match (used to resolve player/team names and positions).
   Columns (at minimum):
   - match_id
   - team_name
   - player_id
   - player_name
   - position_name
   - jersey_number

4) column_map — Maps original JSON field names to sanitized SQL column names with examples.
   Columns:
   - table_name
   - original_name
   - sanitized_name

============================================================
IMPORTANT ABOUT SEQUENCES
============================================================
- For event sequences within a match, ALWAYS include:
  ORDER BY event_index ASC
- If returning a sequence, include at least:
  match_id, event_index, type__name, and time fields (timestamp OR minute+second if available).

============================================================
COLUMN NAMES / SANITIZATION
============================================================
- Some original StatsBomb names are sanitized (e.g., '50_50' -> 'c_50_50').
- If unsure about a column name, call table_columns('events') or query column_map.

============================================================
TOOLS (how to use them)
============================================================

A) table_columns(table_name)
   - Use to inspect available columns in games/events/lineups/column_map.
   - Call this before using uncommon columns.

B) sql_query(sql)
   - Execute ONE read-only SELECT/WITH query.
   - The executor may append a LIMIT automatically; that is expected.

C) sql_to_df(sql, name=..., limit=...)
   - Executes ONE read-only SELECT/WITH query and stores the result as a pandas DataFrame.
   - The DataFrame is stored under the given name.
   - Use this for any visualization or when you need pandas operations.

D) python_viz(code, dataframes=[...])
   - Runs plotting code using DataFrames fetched via sql_to_df.
   - IMPORTANT: The value of "name" in the sql_to_df output is the EXACT variable name
     Example (name round-trip is mandatory):
          1) Call:
               sql_to_df(sql="...", name="example_df")
             Tool returns (excerpt):
               {{\"name\": \"example_df\", ...}}
        
          2) Then call:
               python_viz(code="...", dataframes=["example_df"])
        
          3) Inside python_viz code you MUST reference:
               df = example_df.copy()

   - DataFrames are available only as variables by name (e.g., example_df).
   - Always add title and labels when required

E) search_mplsoccer_docs(query)
   - Search tool for mplsoccer documentation. 
   - ALWAYS call this if you are unsure of an argument (e.g., how to change pitch color, how to use Pitch, or how to plot a heatmap).
   - It returns multiple results with snippets and links to ensure technical accuracy.

============================================================
HARD RULES (must follow)
============================================================
- Use less steps to get the job done not much tool calling
- Always use table_colums first to understand the table structure before query
1) Never invent numbers, columns, or match_ids. Always query.
2) One SQL statement per tool call. SELECT/WITH only.
3) For names, prefer case-insensitive matching:
   lower(player__name) LIKE '%hakimi%'
4) Resolve match_id(s) using games first when needed.
5) For sequences: ORDER BY event_index ASC is mandatory.
6) Use search_mplsoccer_docs whenever a user asks for a specific "look" or "style" you haven't produced before.

============================================================
VISUALIZATION WORKFLOW (MANDATORY)
============================================================
When the user asks for any plot/map/heatmap/network:

1) Clarify scope in ONE sentence (match_id(s), team(s), player(s), event type(s)).
2) Call sql_to_df to fetch ONLY needed columns (avoid SELECT *).
   - Include event_index if the visualization depends on event order.
3) Validate using sql_to_df output (row_count + preview):
   - confirm match_id coverage (distinct match_id if applicable)
   - confirm entity match (player__name/team__name)
   - confirm event type filter (type__name)
   If anything is off, fix SQL and call sql_to_df again.
4) Call python_viz ONCE with code that:
   - uses df = <dataframe_name>.copy()
   - parses StatsBomb locations (JSON strings like "[x,y]") safely
   - uses mplsoccer Pitch if drawing on a pitch
   - DOES NOT call plt.show()
   - ALWAYS saves at least one figure to {PLOT_DIR} (e.g., plt.savefig('{PLOT_DIR}/plot.png', dpi=150, bbox_inches='tight'))
   - in python_viz, refer to dataframes by their variable name (e.g., player_passes)
   - prefer using arrows for passes and shots vizualisations (using location and end location)

5) Retry python_viz ONLY if:
   - python_viz returned an error OR
   - python_viz returned empty image_paths (treat as incomplete visualization).
6) Final response must include:
   - what was plotted
   - key counts (n events, n matches)
   - match context (teams/date) when relevant

============================================================
ANSWER STYLE
============================================================
- For non-visual answers: return a clean table or bullet list grounded in SQL results.
- If multiple matches are possible: show candidates (match_id, date, teams) and ask user to pick.

""".strip()


# ----------------------------
# Download cache helper
# ----------------------------
def download_json(url: str, out_path: Path) -> object:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")
    return r.json()


# ----------------------------
# DB build helpers (schema-evolving events table)
# ----------------------------
_COL_BAD = re.compile(r"[^0-9a-zA-Z_]")
_COL_STARTS_DIGIT = re.compile(r"^[0-9]")


def sanitize_col(name: str) -> str:
    s = _COL_BAD.sub("_", name)
    if _COL_STARTS_DIGIT.match(s):
        s = "c_" + s
    return s


def coerce_complex_to_json(df: pd.DataFrame) -> pd.DataFrame:
    def conv(v):
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False)
        return v

    for c in df.columns:
        if df[c].dtype == "object":
            needs = df[c].map(lambda x: isinstance(x, (dict, list))).any()
            if needs:
                df[c] = df[c].map(conv)
    return df


def ensure_columns(conn: sqlite3.Connection, table: str, cols: list[str]) -> None:
    existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()}
    missing = [c for c in cols if c not in existing]
    for c in missing:
        conn.execute(f'ALTER TABLE {table} ADD COLUMN "{c}" TEXT;')
    conn.commit()


def upsert_column_map(conn: sqlite3.Connection, table: str, mapping: dict[str, str]) -> None:
    rows = [(table, orig, san) for orig, san in mapping.items() if orig != san]
    if rows:
        conn.executemany(
            "INSERT INTO column_map(table_name, original_name, sanitized_name) VALUES (?,?,?)",
            rows,
        )
        conn.commit()


@dataclass
class BuildProgress:
    stage: str
    i: int
    total: int


ProgressCb = Callable[[BuildProgress], None]


def build_afcon_db(force: bool = False, progress_cb: Optional[ProgressCb] = None) -> Path:
    """
    Builds data/afcon.sqlite if missing (or force=True).
    Includes games + fully flattened events + lineups.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists() and not force:
        return DB_PATH

    # --------- matches -> games ----------
    matches_url = f"{BASE}/matches/{AFCON_COMP_ID}/{AFCON_SEASON_ID}.json"
    matches_json = download_json(matches_url, DATA_DIR / f"matches_{AFCON_COMP_ID}_{AFCON_SEASON_ID}.json")
    matches_df = pd.json_normalize(matches_json)

    keep = [
        "match_id", "match_date", "kick_off",
        "home_team.home_team_name", "away_team.away_team_name",
        "home_score", "away_score",
        "competition.competition_name", "season.season_name",
        "competition_stage.name",
    ]
    matches_df = matches_df[[c for c in keep if c in matches_df.columns]].copy()
    matches_df.columns = [c.replace(".", "__") for c in matches_df.columns]
    match_ids = matches_df["match_id"].astype(int).tolist()

    # --------- create fresh DB ----------
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    conn.executescript("""
    DROP TABLE IF EXISTS games;
    DROP TABLE IF EXISTS events;
    DROP TABLE IF EXISTS lineups;
    DROP TABLE IF EXISTS column_map;
    """)
    conn.commit()

    matches_df.to_sql("games", conn, index=False)

    conn.execute("""
    CREATE TABLE events (
      event_id TEXT PRIMARY KEY,
      match_id INTEGER,
      event_index INTEGER
    );
    """)
    conn.execute("""
    CREATE TABLE lineups (
      match_id INTEGER,
      team_name TEXT,
      player_id INTEGER,
      player_name TEXT,
      position_name TEXT,
      jersey_number INTEGER
    );
    """)
    conn.execute("""
    CREATE TABLE column_map (
      table_name TEXT,
      original_name TEXT,
      sanitized_name TEXT
    );
    """)
    conn.executescript("""
    CREATE INDEX idx_games_match_id ON games(match_id);
    CREATE INDEX idx_events_match_index ON events(match_id, event_index);
    CREATE INDEX idx_lineups_match_team ON lineups(match_id, team_name);
    """)
    conn.commit()

    # --------- ingest events ----------
    total = len(match_ids)
    for idx, mid in enumerate(match_ids, start=1):
        if progress_cb:
            progress_cb(BuildProgress(stage="Ingest events", i=idx, total=total))

        events_url = f"{BASE}/events/{mid}.json"
        events_json = download_json(events_url, DATA_DIR / "events" / f"{mid}.json")
        df = pd.json_normalize(events_json, sep="__")

        if "match_id" not in df.columns:
            df["match_id"] = mid
        if "id" in df.columns and "event_id" not in df.columns:
            df["event_id"] = df["id"]
        if "index" in df.columns and "event_index" not in df.columns:
            df["event_index"] = df["index"]

        df["raw_json"] = [json.dumps(e, ensure_ascii=False) for e in events_json]

        df = coerce_complex_to_json(df)
        mapping = {c: sanitize_col(c) for c in df.columns}
        df = df.rename(columns=mapping)

        if "event_id" not in df.columns:
            raise ValueError(f"Missing event_id for match {mid}")
        if "event_index" not in df.columns:
            df["event_index"] = range(1, len(df) + 1)

        ensure_columns(conn, "events", list(df.columns))
        upsert_column_map(conn, "events", mapping)

        table_cols = [r[1] for r in conn.execute("PRAGMA table_info(events);").fetchall()]
        for c in table_cols:
            if c not in df.columns:
                df[c] = None
        df = df[table_cols]

        df.to_sql("events", conn, if_exists="append", index=False, chunksize=200)

    # --------- ingest lineups ----------
    def pick_starting_position(positions: list[dict]) -> str | None:
        if not positions:
            return None
        for p in positions:
            if p.get("start_reason") == "Starting XI":
                return p.get("position")
        return positions[0].get("position")

    total = len(match_ids)
    rows: list[dict] = []
    for idx, mid in enumerate(match_ids, start=1):
        if progress_cb:
            progress_cb(BuildProgress(stage="Ingest lineups", i=idx, total=total))

        url = f"{BASE}/lineups/{mid}.json"
        lj = download_json(url, DATA_DIR / "lineups" / f"{mid}.json")

        for team in lj:
            team_name = team.get("team_name")
            for pl in team.get("lineup", []):
                rows.append({
                    "match_id": mid,
                    "team_name": team_name,
                    "player_id": pl.get("player_id"),
                    "player_name": pl.get("player_name"),
                    "position_name": pick_starting_position(pl.get("positions", [])),
                    "jersey_number": pl.get("jersey_number"),
                })

    conn.execute("DELETE FROM lineups;")
    conn.commit()
    pd.DataFrame(rows).to_sql("lineups", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    return DB_PATH
