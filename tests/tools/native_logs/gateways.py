"""Generate native gateway logs using isolated test containers and a local stub.

This configures only the dedicated modulo-native-* instances on loopback ports.
Import the produced files with the ordinary modulo import command.
"""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
PASSWORD = "Modulo-Synthetic-2026!"


def api(client, method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    body = response.json()
    if body.get("success") is not True:
        raise RuntimeError(f"{method} {path}: {body.get('message')}")
    return body.get("data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter", choices=["newapi", "oneapi"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/native-logs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    label = args.adapter
    port = 18081 if args.adapter == "newapi" else 18082
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=60)
    if args.adapter == "newapi":
        setup = api(client, "GET", "/api/setup")
        if not setup["status"]:
            api(
                client,
                "POST",
                "/api/setup",
                json={
                    "username": "root",
                    "password": PASSWORD,
                    "confirmPassword": PASSWORD,
                    "SelfUseModeEnabled": True,
                },
            )
    login = api(
        client,
        "POST",
        "/api/user/login",
        json={"username": "root", "password": PASSWORD if args.adapter == "newapi" else "123456"},
    )
    user = login.get("user", login)
    client.headers["New-Api-User"] = str(user["id"])
    client.headers["UserId"] = str(user["id"])
    if login.get("access_token"):
        client.headers["Authorization"] = "Bearer " + login["access_token"]
    assets = json.loads((ROOT / "static/tokenizer-assets.json").read_text())["assets"]
    vectors = json.loads(
        (ROOT / "tests/fixtures/tokenizers/reasoning_reference_vectors.json").read_text()
    )["vectors"]
    chosen = [
        next((i, v) for i, v in enumerate(vectors) if v["asset"] == asset)
        for asset in ("qwen-3.8-2.4t", "deepseek-v4.1")
    ]
    models = [assets[v["asset"]]["repo"] for _, v in chosen]
    for key in ("ModelRatio", "CompletionRatio"):
        api(
            client,
            "PUT",
            "/api/option/",
            json={"key": key, "value": json.dumps(dict.fromkeys(models, 1))},
        )
    channel = {
        "type": 1,
        "name": "modulo-synthetic",
        "key": "local-synthetic-only",
        "base_url": "http://modulo-native-upstream:8080",
        "models": ",".join(models),
        "group": "default",
        "status": 1,
        "priority": 0,
        "weight": 1,
    }
    channels = api(client, "GET", "/api/channel/")
    channels = channels.get("items", []) if isinstance(channels, dict) else channels
    if not any(c["name"] == channel["name"] for c in channels):
        payload = {"mode": "single", "channel": channel} if args.adapter == "newapi" else channel
        api(client, "POST", "/api/channel/", json=payload)
    tokens = api(client, "GET", "/api/token/")
    tokens = tokens.get("items", []) if isinstance(tokens, dict) else tokens
    token = next((t for t in tokens if t["name"] == "modulo-synthetic"), None)
    if token is None:
        api(
            client,
            "POST",
            "/api/token/",
            json={
                "name": "modulo-synthetic",
                "expired_time": -1,
                "remain_quota": 100000000,
                "unlimited_quota": True,
                "group": "default",
            },
        )
        tokens = api(client, "GET", "/api/token/")
        tokens = tokens.get("items", []) if isinstance(tokens, dict) else tokens
        token = next(t for t in tokens if t["name"] == "modulo-synthetic")
    key = token["key"]
    if args.adapter == "newapi":
        key = api(client, "POST", f"/api/token/{token['id']}/key")["key"]
    key = key if key.startswith("sk-") else f"sk-{key}"
    started = datetime.now(UTC).isoformat()
    received = []
    for i, vector in chosen:
        for stream in (False, True):
            request = {
                "model": assets[vector["asset"]]["repo"],
                "messages": vector["messages"],
                "tools": vector["tools"],
                "stream": stream,
                "user": f"modulo-case-{i}",
                **vector["logged_options"],
            }
            if stream:
                request["stream_options"] = {"include_usage": True}
            response = client.post(
                "/v1/chat/completions", json=request, headers={"Authorization": f"Bearer {key}"}
            )
            if response.status_code != 200:
                raise RuntimeError(f"model call failed: {response.status_code} {response.text}")
            received.append(
                {
                    "request": request,
                    "response": response.text,
                    "request_id": response.headers.get("x-request-id"),
                    "stream": stream,
                }
            )
    logs = subprocess.run(
        ["docker", "logs", "--since", started, f"modulo-native-{label}"],
        capture_output=True,
        check=True,
    )
    (args.output / f"{label}.log").write_bytes(logs.stdout + logs.stderr)
    (args.output / f"{label}-responses.json").write_text(
        json.dumps(received, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "adapter": args.adapter,
                "calls": len(received),
                "native_log_bytes": len(logs.stdout + logs.stderr),
                "path": str(args.output / f"{label}.log"),
            }
        )
    )


if __name__ == "__main__":
    main()
