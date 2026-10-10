"""New API consumption rows, native debug bodies and captured HTTP exchanges."""

from ..shared.gateway import GatewayProtocol


class NewAPIProtocol(GatewayProtocol):
    name = "newapi"
    version = "newapi.body-log.v2"
    call_types = {2: "unknown", 5: "failed"}
    ignored_types = frozenset({1, 3, 4, 6, 7})
    duration_field = "use_time"
    duration_scale = 1000  # Upstream writes UseTimeSeconds, despite ambiguous UI labels.
    metadata_field = "other"
