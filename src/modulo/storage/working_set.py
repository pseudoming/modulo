"""Disk-backed import intermediates and an optional durable source journal."""

import sqlite3
import tempfile
from collections.abc import MutableMapping, Sequence
from pathlib import Path

from modulo.core.domain.data_models import RawRecord
from modulo.shared.serialization import bytes_digest, digest, loads, ordered_json


class WorkingSet:
    def __init__(self, directory: Path | None = None, *, memory_mib: int = 64):
        if type(memory_mib) is not int or memory_mib < 8:
            raise ValueError("memory-mib must be an integer of at least 8")
        self.directory, self.memory_mib = directory, memory_mib
        self.temporary = None

    def __enter__(self):
        try:
            return self._open()
        except BaseException:
            self._close()
            raise

    def _open(self):
        if self.directory is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="modulo-work-")
            self.directory = Path(self.temporary.name)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        # A separate SQLite lock releases automatically even after process death.
        self.lease = sqlite3.connect(self.directory / "lock.sqlite", timeout=0)
        try:
            self.lease.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            self.lease.close()
            raise ValueError("import state is already in use") from exc
        self.connection = sqlite3.connect(self.directory / "state.sqlite")
        self.connection.execute(f"PRAGMA cache_size = -{self.memory_mib * 1024}")
        self.connection.execute("PRAGMA temp_store = FILE")
        self.connection.execute("PRAGMA mmap_size = 0")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS maps (
                name TEXT, key TEXT, value TEXT NOT NULL, PRIMARY KEY (name, key));
            CREATE TABLE IF NOT EXISTS lists (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, group_key TEXT, value TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS list_order ON lists(name, id);
            CREATE INDEX IF NOT EXISTS list_groups ON lists(name, group_key, id);
            CREATE TABLE IF NOT EXISTS raw (
                id INTEGER PRIMARY KEY, identity TEXT UNIQUE NOT NULL,
                locator TEXT NOT NULL, data BLOB NOT NULL);
        """)
        self.writes = 0
        self.autocommit = True
        self.reset()
        return self

    def reset(self):
        """Derived intermediates can be rebuilt; journal/cursors/cache are retained."""
        self.connection.execute("DELETE FROM maps WHERE name NOT LIKE 'saved:%'")
        self.connection.execute("DELETE FROM lists")
        self.connection.commit()

    def changed(self):
        self.writes += 1
        if self.autocommit and self.writes >= 1000:
            self.commit()

    def commit(self):
        self.connection.commit()
        self.writes = 0

    def mapping(self, name):
        return DiskMap(self, name)

    def sequence(self, name):
        return DiskList(self, name)

    def groups(self, name):
        return DiskGroups(self, name)

    def bind(self, context):
        settings = self.mapping("saved:settings")
        fingerprint = digest(context)
        if "context" in settings and settings["context"] != fingerprint:
            raise ValueError("import state has different rules; use a new state directory")
        settings["context"] = fingerprint
        self.commit()

    def remember(self, record):
        identity = digest([record.locator, bytes_digest(record.data)])
        changed = self.connection.execute(
            "INSERT OR IGNORE INTO raw(identity,locator,data) VALUES (?,?,?)",
            (identity, record.locator, record.data),
        ).rowcount
        self.writes += 1
        for key, value in (record.checkpoint or {}).items():
            self.mapping("saved:cursors")[key] = value
        if self.writes >= 1000:
            self.commit()
        return changed

    def raw_records(self):
        for locator, data in self.connection.execute("SELECT locator,data FROM raw ORDER BY id"):
            yield RawRecord(locator, data)

    def raw_count(self):
        return self.connection.execute("SELECT count(*) FROM raw").fetchone()[0]

    def __exit__(self, kind, *_):
        try:
            if kind is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self._close()

    def _close(self):
        for name in ("connection", "lease"):
            connection = getattr(self, name, None)
            if connection is not None:
                connection.close()
        if self.temporary:
            self.temporary.cleanup()


def _key(value):
    parsed = loads(value)
    return tuple(parsed) if isinstance(parsed, list) else parsed


class DiskMap(MutableMapping):
    def __init__(self, workspace, name):
        self.workspace, self.name = workspace, name
        self.connection = workspace.connection

    def __getitem__(self, key):
        row = self.connection.execute(
            "SELECT value FROM maps WHERE name=? AND key=?", (self.name, ordered_json(key))
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return loads(row[0])

    def __setitem__(self, key, value):
        self.connection.execute(
            "INSERT INTO maps VALUES (?,?,?) "
            "ON CONFLICT(name,key) DO UPDATE SET value=excluded.value",
            (self.name, ordered_json(key), ordered_json(value)),
        )
        self.workspace.changed()

    def __delitem__(self, key):
        cursor = self.connection.execute(
            "DELETE FROM maps WHERE name=? AND key=?", (self.name, ordered_json(key))
        )
        if not cursor.rowcount:
            raise KeyError(key)
        self.workspace.changed()

    def __iter__(self):
        for (key,) in self.connection.execute(
            "SELECT key FROM maps WHERE name=? ORDER BY key", (self.name,)
        ):
            yield _key(key)

    def __len__(self):
        return self.connection.execute(
            "SELECT count(*) FROM maps WHERE name=?", (self.name,)
        ).fetchone()[0]

    def clear(self):
        self.connection.execute("DELETE FROM maps WHERE name=?", (self.name,))
        self.workspace.changed()


class DiskList(Sequence):
    def __init__(self, workspace, name, group_key=None):
        self.workspace, self.name, self.group_key = workspace, name, group_key
        self.connection = workspace.connection
        self.where = "name=?" + (" AND group_key=?" if group_key is not None else "")
        self.parameters = (name, group_key) if group_key is not None else (name,)

    def append(self, value):
        self.connection.execute(
            "INSERT INTO lists(name,group_key,value) VALUES (?,?,?)",
            (self.name, self.group_key, ordered_json(value)),
        )
        self.workspace.changed()

    def extend(self, values):
        for value in values:
            self.append(value)

    def __iter__(self):
        for (value,) in self.connection.execute(
            f"SELECT value FROM lists WHERE {self.where} ORDER BY id", self.parameters
        ):
            yield loads(value)

    def __len__(self):
        return self.connection.execute(
            f"SELECT count(*) FROM lists WHERE {self.where}", self.parameters
        ).fetchone()[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(index)
        row = self.connection.execute(
            f"SELECT value FROM lists WHERE {self.where} ORDER BY id LIMIT 1 OFFSET ?",
            (*self.parameters, index),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return loads(row[0])

    @property
    def byte_size(self):
        return self.connection.execute(
            f"SELECT coalesce(sum(length(CAST(value AS BLOB))),0) FROM lists WHERE {self.where}",
            self.parameters,
        ).fetchone()[0]


class DiskGroups:
    def __init__(self, workspace, name):
        self.workspace, self.name = workspace, name

    def append(self, key, value):
        DiskList(self.workspace, self.name, ordered_json(key)).append(value)

    def items(self):
        for (key,) in self.workspace.connection.execute(
            "SELECT group_key FROM lists WHERE name=? GROUP BY group_key ORDER BY min(id)",
            (self.name,),
        ):
            yield _key(key), DiskList(self.workspace, self.name, key)

    def clear(self):
        self.workspace.connection.execute("DELETE FROM lists WHERE name=?", (self.name,))
