"""One API consumption rows and body logs, including optional response instrumentation."""

from modulo.adapters.protocols.shared.gateway.adapter import GatewayProtocol


class OneAPIProtocol(GatewayProtocol):
    name = "oneapi"
    version = "oneapi.body-log.v3"
    call_types = {2: "unknown"}
    ignored_types = frozenset({1, 3, 4, 5})
    duration_field = "elapsed_time"
    duration_scale = 1
