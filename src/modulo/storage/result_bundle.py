"""Archive source logs, publish complete result bundles and verify saved files."""

import base64
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from modulo.core.domain.bindings import Bindings
from modulo.core.domain.data_models import BatchResult, RawRecord
from modulo.shared.file_hash import file_sha256
from modulo.shared.serialization import bytes_digest, digest, dumps, iter_dumps, loads

AUTHORITY_FILES = frozenset(
    {
        "raw.jsonl",
        "results.jsonl",
        "diagnostics.jsonl",
        "summary.json",
        "execution.json",
        "bindings.json",
        "assets-lock.json",
    }
)


def verify_bundle(directory: Path) -> dict:
    manifest = loads((directory / "manifest.json").read_bytes())
    if manifest.get("schema_version") != "bundle.v1":
        raise ValueError("unsupported bundle schema")
    if set(manifest.get("files", {})) != AUTHORITY_FILES:
        raise ValueError("bundle has an incomplete or unexpected authority file set")
    if (
        manifest.get("run_id")
        != digest({"plan": manifest["plan"], "files": manifest["files"]})[:24]
    ):
        raise ValueError("bundle run id does not match plan and files")
    for name, expected in manifest["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"bundle hash mismatch: {name}")
    return manifest


class BundleWriter:
    def __init__(self, output_root: Path):
        self.output_root = output_root

    def __enter__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.output_root))
        try:
            self.raw = (self.directory / "raw.jsonl").open("w", encoding="utf-8")
        except BaseException:
            shutil.rmtree(self.directory)
            raise
        return self

    def record(self, record: RawRecord) -> dict[str, str]:
        reference = {"locator": record.locator, "sha256": bytes_digest(record.data)}
        self.raw.write(
            dumps(
                {
                    **reference,
                    "encoding": "base64",
                    "data": base64.b64encode(record.data).decode(),
                }
            )
            + "\n"
        )
        return reference

    def publish(
        self,
        plan: dict,
        result: BatchResult,
        *,
        execution: dict,
        bindings: Bindings,
        asset_lock: dict,
        rebuild_index: Callable[[Path], Path],
    ) -> tuple[Path, bool]:
        self.raw.close()
        temporary, output_root = self.directory, self.output_root
        rows, diagnostics, summary = result.rows, result.diagnostics, result.summary
        status = summary["run_status"]
        for name, values in (("results.jsonl", rows), ("diagnostics.jsonl", diagnostics)):
            with (temporary / name).open("w", encoding="utf-8") as stream:
                for row in values:
                    stream.write(dumps(row) + "\n")
        for name, value in (
            ("summary.json", summary),
            ("execution.json", execution),
            ("bindings.json", bindings.model_dump(by_alias=True)),
            ("assets-lock.json", asset_lock),
        ):
            with (temporary / name).open("w", encoding="utf-8") as stream:
                stream.writelines(iter_dumps(value))
                stream.write("\n")
        files = {name: file_sha256(temporary / name) for name in sorted(AUTHORITY_FILES)}
        run_id = digest({"plan": plan, "files": files})[:24]
        manifest = {
            "schema_version": "bundle.v1",
            "run_id": run_id,
            "status": status,
            "created_at": datetime.now(UTC).isoformat(),
            "plan": plan,
            "files": files,
        }
        (temporary / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
        target = output_root / run_id
        if target.exists():
            prior = verify_bundle(target)
            # Decimal plan values are stored as exact strings; compare the
            # persisted representation, not its in-memory Python types.
            if prior["files"] != files or dumps(prior["plan"]) != dumps(plan):
                raise ValueError("run id collision")
            return target, True
        rebuild_index(temporary)
        temporary.rename(target)
        return target, False

    def __exit__(self, *_exc):
        try:
            self.raw.close()
        finally:
            if self.directory.exists():
                shutil.rmtree(self.directory)
