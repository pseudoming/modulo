"""Build a DuckDB query index from verified result bundles and read call details."""

import csv
import os
import tempfile
from pathlib import Path

import duckdb

from modulo.shared.serialization import loads

from .result_bundle import verify_bundle


def rebuild_index(directory: Path, *, memory_mib: int = 64) -> Path:
    verify_bundle(directory)
    target = directory / "query.duckdb"
    temporary = directory / ".query.duckdb"
    if temporary.exists():
        temporary.unlink()
    try:
        with (
            tempfile.TemporaryDirectory(prefix=".query-rows-", dir=directory) as staging,
            duckdb.connect() as connection,
        ):
            connection.execute("SET threads = 1")
            connection.execute("SET memory_limit = ?", [f"{memory_mib}MiB"])
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET enable_external_file_cache = false")
            connection.execute("SET write_buffer_row_group_count = 1")
            connection.execute(
                "SET write_buffer_row_group_memory_limit = ?", [f"{max(1, memory_mib // 8)}MiB"]
            )
            # Small row groups bound the JSON column's write/compression buffers.
            # ATTACH does not accept a bound path; escape its SQL string literal.
            quoted_path = str(temporary).replace("'", "''")
            connection.execute(f"ATTACH '{quoted_path}' AS result_index (ROW_GROUP_SIZE 2048)")
            connection.execute("USE result_index")
            # Identity uniqueness is already enforced in the disk-backed core
            # flow. A second in-memory ART index would grow with the call count.
            connection.execute("""CREATE TABLE calls (
                call_key VARCHAR, call_id VARCHAR, model VARCHAR,
                platform VARCHAR, status VARCHAR,
                verdict VARCHAR, input_verdict VARCHAR,
                reported_input_tokens VARCHAR, input_tokens VARCHAR,
                input_delta VARCHAR, reported_output_tokens VARCHAR, output_tokens VARCHAR,
                output_delta VARCHAR, output_verdict VARCHAR, payload JSON)""")
            rows_file = Path(staging) / "rows.csv"
            max_line_size, previous_offset = 1, 0
            with (
                (directory / "results.jsonl").open(encoding="utf-8") as stream,
                rows_file.open("w", encoding="utf-8", newline="") as output,
            ):
                # Explicit VARCHAR columns preserve arbitrary-sized integers.
                # Unquoted nulls remain distinct from quoted empty strings.
                writer = csv.writer(output, quoting=csv.QUOTE_NOTNULL, lineterminator="\n")
                for line in stream:
                    row = loads(line)
                    call, count = row["call"], row["measurement"]
                    values = [
                        count.get(k)
                        for k in (
                            "reported_input_tokens",
                            "input_tokens",
                            "input_delta",
                            "reported_output_tokens",
                            "output_tokens",
                            "output_delta",
                        )
                    ]
                    writer.writerow(
                        (
                            row["call_key"],
                            call["call_id"],
                            call["model"],
                            call.get("platform"),
                            call["status"],
                            # Older bundles used verdict for input only. Keep
                            # that meaning; never label it an overall result.
                            count["verdict"] if "input_verdict" in count else None,
                            count.get("input_verdict", count["verdict"]),
                            *(None if v is None else str(v) for v in values),
                            count.get("output_verdict"),
                            line.rstrip("\r\n"),
                        )
                    )
                    offset = output.tell()
                    max_line_size = max(max_line_size, offset - previous_offset)
                    previous_offset = offset
            if previous_offset:
                connection.execute(
                    """COPY calls FROM ? (FORMAT CSV, HEADER false, AUTO_DETECT false,
                        DELIMITER ',', QUOTE '"', ESCAPE '"', NULL '',
                        ALLOW_QUOTED_NULLS false, """
                    f"MAX_LINE_SIZE {max_line_size}, "
                    f"BUFFER_SIZE {max(1024 * 1024, max_line_size * 4)})",
                    [str(rows_file)],
                )
            connection.execute("""CREATE VIEW output_candidates AS
                SELECT calls.call_key, calls.call_id, calls.model, calls.platform,
                    json_extract_string(candidate.value, '$.index') AS candidate_index,
                    json_extract_string(candidate.value, '$.execution') AS execution,
                    json_extract_string(candidate.value, '$.tokens') AS output_tokens,
                    json_extract_string(candidate.value, '$.reason') AS reason,
                    candidate.value AS payload
                FROM calls, json_each(calls.payload, '$.measurement.output_candidates') AS candidate
                """)
            connection.execute("""CREATE VIEW input_images AS
                SELECT calls.call_key, calls.call_id, calls.model, calls.platform,
                    json_extract_string(image.value, '$.message_index') AS message_index,
                    json_extract_string(image.value, '$.part_index') AS part_index,
                    json_extract_string(image.value, '$.execution') AS execution,
                    json_extract_string(image.value, '$.input_tokens') AS input_tokens,
                    json_extract_string(image.value, '$.reason') AS reason,
                    image.value AS payload
                FROM calls, json_each(calls.payload, '$.measurement.input_images') AS image
                """)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def details(directory: Path, call_id: str | None = None) -> list[dict]:
    return list(iter_details(directory, call_id))


def iter_details(directory: Path, call_id: str | None = None):
    # Rebuild from verified facts instead of silently trusting a mutable cache.
    path = rebuild_index(directory)
    with duckdb.connect(str(path), read_only=True) as connection:
        connection.execute("SET memory_limit = '64MiB'")
        connection.execute("SET threads = 1")
        if call_id is None:
            connection.execute("SELECT payload FROM calls ORDER BY call_key")
        else:
            connection.execute(
                "SELECT payload FROM calls WHERE call_id = ? ORDER BY call_key", [call_id]
            )
        while row := connection.fetchone():
            yield loads(row[0])
