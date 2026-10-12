"""Resume source reading and replay one cumulative, immutable audit snapshot."""

from modulo.core.domain.data_models import SourceError
from modulo.core.domain.interfaces import SourceAdapter


class JournalSource(SourceAdapter):
    name = "journal"
    version = "source-journal"

    def __init__(self, source, working, max_records):
        self.source, self.working, self.max_records = source, working, max_records
        self.ordered_records = source.ordered_records
        self.error = None

    def capture(self):
        reader = getattr(self.source, "read_from", None)
        if reader is None:
            raise ValueError("this source does not support resumable import")
        self.working.autocommit = False
        records = reader(self.working.mapping("saved:cursors"))
        added = 0
        try:
            for record in records:
                added += self.working.remember(record)
                if self.max_records and added >= self.max_records:
                    self.error = SourceError("record_budget_reached; continue with the same state")
                    break
        except (SourceError, OSError) as exc:
            self.error = exc
        finally:
            try:
                records.close()
            finally:
                self.working.autocommit = True
        self.working.commit()

    def describe(self):
        return {
            **self.source.describe(),
            "journal_records": self.working.raw_count(),
            "journal_scope": "cumulative",
            "capture_complete": self.error is None,
            "capture_error": str(self.error)
            if isinstance(self.error, SourceError)
            else "io_error"
            if self.error
            else None,
        }

    def read(self):
        yield from self.working.raw_records()
        if self.error:
            raise self.error
