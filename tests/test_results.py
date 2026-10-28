# SPDX-License-Identifier: MIT
"""The results store: record/query/has_result and the `has` exit codes (CPU)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import results

SCRIPT = Path(results.__file__)


def _fill(db):
    rid = results.start_run("profile_model.py", "/m/Model-A/", argv=["x"], db=db)
    for kv, sd, ms in (("int8_per_token_head", "all", 40.0), ("auto", "", 30.0)):
        results.record(
            rid,
            "decode",
            "decode_ms",
            ms,
            label="depth131072#0",
            depth=131072,
            kv_dtype=kv,
            strix_disable=sd,
            extra={"k": 1},
            db=db,
        )
    return rid


def test_record_query(tmp_path):
    db = tmp_path / "sub" / "r.db"
    rid = _fill(db)
    rows = results.query(db=db, model="Model-A", phase="decode")
    assert [r["value"] for r in rows] == [40.0, 30.0]
    assert rows[0]["run_id"] == rid and rows[0]["model"] == "Model-A"
    assert rows[0]["extra"] == '{"k": 1}'
    assert results.query(db=db, strix_disable="")[0]["kv_dtype"] == "auto"
    assert results.query(db=db, model="Other") == []


def test_strix_disable_from_env(tmp_path, monkeypatch):
    db = tmp_path / "r.db"
    monkeypatch.setenv("STRIX_DISABLE", "all")
    rid = results.start_run("s.py", "M", db=db)
    results.record(rid, "kld", "kld", 0.1, db=db)
    row = results.query(db=db)[0]
    assert row["strix_disable"] == "all"
    assert "STRIX_DISABLE" in row["env"]


def test_has_result_and_commit(tmp_path):
    db = tmp_path / "r.db"
    _fill(db)
    kw = {"db": db, "model": "Model-A", "phase": "decode", "depth": 131072}
    assert results.has_result(**kw, kv_dtype="auto", strix_disable="")
    assert not results.has_result(**kw, kv_dtype="auto", strix_disable="all")
    assert not results.has_result(**kw, metric="ttft_s")
    assert not results.has_result(**kw, commit="deadbeef0")
    commit = results.query(db=db)[0]["git_commit"]
    if commit:
        assert results.has_result(**kw, commit=commit[:8])


def _cli(db, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli(tmp_path):
    db = tmp_path / "r.db"
    _fill(db)
    has = ["has", "--model", "Model-A", "--phase", "decode", "--depth", "131072"]
    on = ["--strix-disable", "none"]
    assert _cli(db, *has, "--kv-dtype", "auto", *on).returncode == 0
    assert _cli(db, *has, "--strix-disable", "all").returncode == 0
    assert (
        _cli(db, *has, "--kv-dtype", "auto", "--strix-disable", "all").returncode == 1
    )
    assert _cli(db, *has, "--metric", "ttft_s").returncode == 1
    shown = _cli(db, "show", "--model", "Model-A")
    assert shown.returncode == 0 and "decode_ms" in shown.stdout
