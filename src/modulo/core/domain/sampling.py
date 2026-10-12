"""Sampling configuration; time and randomness are explicit, replayable inputs."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


class SamplingOptions(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, alias_generator=lambda name: name.replace("_", "-")
    )

    mode: Literal["all", "ratio", "count", "window"] = "all"
    ratio: Decimal | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    count: int | None = Field(default=None, ge=1, le=100_000)
    seed: str = Field(default="modulo", min_length=1, max_length=1024)
    window_minutes: int | None = Field(default=None, ge=1, le=525_600)
    before: AwareDatetime | None = None

    @field_validator("count", "window_minutes", mode="before")
    @classmethod
    def integer_option(cls, value):
        if value is None or type(value) is int:
            return value
        if isinstance(value, str):
            return int(value)
        raise ValueError("sampling counts and minutes must be integers")

    @field_validator("before", mode="before")
    @classmethod
    def explicit_datetime(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("sampling before must be an ISO timestamp with timezone")
        return value

    @model_validator(mode="after")
    def validate_strategy(self):
        if (self.ratio is not None) != (self.mode == "ratio"):
            raise ValueError("ratio is required only for ratio sampling")
        if (self.count is not None) != (self.mode in {"count", "window"}):
            raise ValueError("count is required only for count/window sampling")
        if (self.window_minutes is not None) != (self.mode == "window"):
            raise ValueError("window-minutes is required only for window sampling")
        if self.before is not None and self.mode != "window":
            raise ValueError("before applies only to window sampling")
        return self

    def resolve(self, now: datetime):
        """Freeze the latest completed UTC window, or an explicit historical endpoint."""
        if self.mode != "window":
            return self
        if now.tzinfo is None:
            raise ValueError("sampling clock must include a timezone")
        end = self.before
        if end is None:
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
            width = timedelta(minutes=self.window_minutes)
            end = epoch + ((now.astimezone(UTC) - epoch) // width) * width
        return self.model_copy(update={"before": end.astimezone(UTC)})

    def bounds(self):
        if self.mode != "window":
            return None
        if self.before is None:
            raise ValueError("freeze the sampling window before running core")
        return self.before - timedelta(minutes=self.window_minutes), self.before
