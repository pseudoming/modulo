"""Local synthetic model used by native logging acceptance; never calls a model.

Mounted read-only in a temporary Python container by native logging lab.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

ROOT = Path(os.environ.get("MODULO_ROOT", "/workspace"))
VECTORS = json.loads(
    (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
)["vectors"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true,"synthetic":true}')

    def do_POST(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            parts = []
            while size := int(self.rfile.readline().split(b";", 1)[0].strip(), 16):
                parts.append(self.rfile.read(size))
                assert self.rfile.read(2) == b"\r\n"
            assert self.rfile.readline() == b"\r\n"
            raw = b"".join(parts)
        else:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
        data = json.loads(raw)
        index = int(
            self.headers.get("x-modulo-case", data.get("user", "modulo-case-0")).removeprefix(
                "modulo-case-"
            )
        )
        vector = VECTORS[index]
        common = {
            "id": f"native-case-{index}-{uuid4().hex}",
            "model": data["model"],
            "created": 1791590400,
        }
        usage = {
            "prompt_tokens": len(vector["input_ids"]),
            "completion_tokens": vector["visible_output_tokens"]
            + vector["reasoning_output_tokens"],
        }
        usage["total_tokens"] = sum(usage.values())
        self.send_response(200)
        self.send_header("x-request-id", f"native-case-{index}")
        self.send_header(
            "Content-Type", "text/event-stream" if data.get("stream") else "application/json"
        )
        self.end_headers()
        if not data.get("stream"):
            body = {
                **common,
                "object": "chat.completion",
                "usage": usage,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": vector["output_text"],
                            "reasoning_content": vector["output_reasoning"],
                        },
                    }
                ],
            }
            self.wfile.write(json.dumps(body, ensure_ascii=False).encode())
            return
        deltas = [
            {"role": "assistant", "content": ""},
            {"reasoning_content": vector["output_reasoning"]},
            {"content": vector["output_text"]},
            {},
        ]
        for i, delta in enumerate(deltas):
            chunk = {
                **common,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": "stop" if i == 3 else None}
                ],
            }
            if i == 3:
                chunk["usage"] = usage
            self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
