"""Build a DuckDB query index from verified result bundles and read call details."""

import os
from pathlib import Path

import duckdb

from modulo.shared.serialization import loads

from .result_bundle import verify_bundle


def rebuild_index(directory: Path) -> Path:
    verify_bundle(directory)
    target = directory / "query.duckdb"
    temporary = directory / ".query.duckdb"
    if temporary.exists():
        temporary.unlink()
    try:
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("SET threads = 1")
            connection.execute("""CREATE TABLE calls (
                call_key VARCHAR PRIMARY KEY, call_id VARCHAR, model VARCHAR,
                platform VARCHAR, status VARCHAR,
                verdict VARCHAR, reported_input_tokens VARCHAR, input_tokens VARCHAR,
                input_delta VARCHAR, payload JSON)""")
            with (directory / "results.jsonl").open(encoding="utf-8") as stream:
                batch = []
                for line in stream:
                    row = loads(line)
                    call, count = row["call"], row["measurement"]
                    values = [
                        count.get(k)
                        for k in ("reported_input_tokens", "input_tokens", "input_delta")
                    ]
                    batch.append(
                        (
                            row["call_key"],
                            call["call_id"],
                            call["model"],
                            call.get("platform"),
                            call["status"],
                            count["verdict"],
                            *(None if v is None else str(v) for v in values),
                            line,
                        )
                    )
                    if len(batch) >= 500:
                        connection.executemany(
                            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)", batch
                        )
                        batch.clear()
                if batch:
                    connection.executemany("INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)", batch)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def details(directory: Path, call_id: str | None = None) -> list[dict]:
    # Rebuild from verified facts instead of silently trusting a mutable cache.
    path = rebuild_index(directory)
    with duckdb.connect(str(path), read_only=True) as connection:
        if call_id is None:
            rows = connection.execute("SELECT payload FROM calls ORDER BY call_key").fetchall()
        else:
            rows = connection.execute(
                "SELECT payload FROM calls WHERE call_id = ? ORDER BY call_key", [call_id]
            ).fetchall()
    return [loads(row[0]) for row in rows]
