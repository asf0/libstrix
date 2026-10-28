# SPDX-License-Identifier: MIT
"""One SQLite store for benchmark results, instead of numbers scattered in logs.

The bench scripts (profile_model, kld_check, e2e_check) record the values they
print; the drivers ask `has` whether a config is already done; the collect_*
scripts read it back. Every run keeps the commit, dirty flag, STRIX_* env and
argv, so a number is never separated from the code and kernel set that made it.
Raw logs stay disposable under profiles/logs/.

    python bench/results.py has --model M --phase decode --depth 131072 \\
        --kv-dtype int8_per_token_head --strix-disable all --commit HEAD
    python bench/results.py show --model M --phase kld

`has` exits 0 if a matching row exists, 1 if not. --strix-disable none means
the plugin is on (STRIX_DISABLE unset or empty); omit the flag to match any.
Models are matched by checkpoint directory name. The default DB is
profiles/results.db, overridden by --db or $STRIX_RESULTS_DB. WAL mode and one
short transaction per call keep a second process (a collector while a job
records) from blocking.
"""

import argparse
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "profiles" / "results.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER,
    host TEXT,
    script TEXT,
    argv TEXT,
    env TEXT,
    model TEXT,
    log_path TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    phase TEXT NOT NULL,
    label TEXT,
    depth INTEGER,
    batch INTEGER,
    kv_dtype TEXT,
    strix_disable TEXT NOT NULL DEFAULT '',
    metric TEXT NOT NULL,
    value REAL,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS results_key
    ON results(phase, depth, kv_dtype, strix_disable, metric);
CREATE INDEX IF NOT EXISTS results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS runs_model ON runs(model, git_commit);
"""


def db_path(db: str | os.PathLike | None = None) -> Path:
    return Path(db or os.environ.get("STRIX_RESULTS_DB") or DEFAULT_DB)


def connect(db: str | os.PathLike | None = None) -> sqlite3.Connection:
    path = db_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def model_name(model: str | os.PathLike | None) -> str | None:
    """Checkpoint directory name, so a path and a bare name match."""
    return Path(str(model).rstrip("/")).name if model else None


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--untracked-files=no"))


def current_strix_disable() -> str:
    return os.environ.get("STRIX_DISABLE", "")


def start_run(
    script: str,
    model: str | None = None,
    argv: list[str] | None = None,
    log_path: str | None = None,
    db: str | os.PathLike | None = None,
) -> str:
    """Insert a run row and return its id. $STRIX_LOG_PATH names the log."""
    run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    env = {k: v for k, v in os.environ.items() if k.startswith("STRIX_")}
    con = connect(db)
    with con:
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                git_commit(),
                int(git_dirty()),
                socket.gethostname(),
                Path(script).name,
                json.dumps(sys.argv if argv is None else argv),
                json.dumps(env),
                model_name(model),
                log_path or os.environ.get("STRIX_LOG_PATH"),
            ),
        )
    con.close()
    return run_id


def record(
    run_id: str,
    phase: str,
    metric: str,
    value: float,
    label: str | None = None,
    depth: int | None = None,
    batch: int | None = None,
    kv_dtype: str | None = None,
    strix_disable: str | None = None,
    extra: dict | None = None,
    db: str | os.PathLike | None = None,
) -> None:
    """Add one number to a run; strix_disable defaults to the live env."""
    if strix_disable is None:
        strix_disable = current_strix_disable()
    con = connect(db)
    with con:
        con.execute(
            "INSERT INTO results (run_id, phase, label, depth, batch, kv_dtype,"
            " strix_disable, metric, value, extra) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                phase,
                label,
                depth,
                batch,
                kv_dtype,
                strix_disable,
                metric,
                float(value),
                None if extra is None else json.dumps(extra),
            ),
        )
    con.close()


def query(
    db: str | os.PathLike | None = None,
    model: str | None = None,
    phase: str | None = None,
    label: str | None = None,
    depth: int | None = None,
    batch: int | None = None,
    kv_dtype: str | None = None,
    strix_disable: str | None = None,
    commit: str | None = None,
    metric: str | None = None,
) -> list[dict]:
    """Rows joined with their run, oldest first. None means no filter.

    commit is a hash prefix; strix_disable "" selects the plugin-on rows.
    """
    where, args = [], []
    for col, val in (
        ("r.model", model_name(model)),
        ("x.phase", phase),
        ("x.label", label),
        ("x.depth", depth),
        ("x.batch", batch),
        ("x.kv_dtype", kv_dtype),
        ("x.strix_disable", strix_disable),
        ("x.metric", metric),
    ):
        if val is not None:
            where.append(f"{col} = ?")
            args.append(val)
    if commit is not None:
        where.append("r.git_commit LIKE ?")
        args.append(commit + "%")
    sql = (
        "SELECT x.*, r.started_at, r.git_commit, r.git_dirty, r.host, r.script,"
        " r.argv, r.env, r.model, r.log_path FROM results x"
        " JOIN runs r USING (run_id)"
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY r.started_at, x.id"
    )
    con = connect(db)
    rows = [dict(r) for r in con.execute(sql, args)]
    con.close()
    return rows


def has_result(**filters) -> bool:
    return bool(query(**filters))


def latest(rows: list[dict], key) -> dict:
    """Newest row per key(row); rows come back oldest first."""
    return {key(r): r for r in rows}


def _filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model")
    p.add_argument("--phase")
    p.add_argument("--label")
    p.add_argument("--depth", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--kv-dtype")
    p.add_argument("--strix-disable", help="e.g. all; 'none' = plugin on")
    p.add_argument("--commit", help="hash prefix, or HEAD for the current commit")
    p.add_argument("--metric")


def _kwargs(a: argparse.Namespace) -> dict:
    sd = a.strix_disable
    commit = git_commit() if a.commit in ("HEAD", "current") else a.commit
    return {
        "db": a.db,
        "model": a.model,
        "phase": a.phase,
        "label": a.label,
        "depth": a.depth,
        "batch": a.batch,
        "kv_dtype": a.kv_dtype,
        "strix_disable": "" if sd == "none" else sd,
        "commit": commit,
        "metric": a.metric,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", help=f"default {DEFAULT_DB} or $STRIX_RESULTS_DB")
    sub = p.add_subparsers(dest="cmd", required=True)
    _filters(sub.add_parser("has", help="exit 0 if a matching result exists"))
    s = sub.add_parser("show", help="print matching results")
    _filters(s)
    s.add_argument("--limit", type=int, default=200, help="newest N rows")
    a = p.parse_args()
    kw = _kwargs(a)
    if a.cmd == "has":
        return 0 if has_result(**kw) else 1
    rows = query(**kw)[-a.limit :]
    print(
        "| started | commit | model | phase | label | kv | disable | metric | value |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | ---: |")
    for r in rows:
        dirty = "+" if r["git_dirty"] else ""
        print(
            f"| {r['started_at']} | {(r['git_commit'] or '?')[:8]}{dirty} | "
            f"{r['model']} | {r['phase']} | {r['label'] or ''} | "
            f"{r['kv_dtype'] or ''} | {r['strix_disable'] or '-'} | "
            f"{r['metric']} | {r['value']:.6g} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
