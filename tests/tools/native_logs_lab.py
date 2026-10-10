"""One developer entry for native producer generation and end-to-end validation.

Daily data imports still use modulo import. This lab never calls a remote model.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["langchain", "dify", "dify-agent", "litellm", "newapi", "oneapi", "verify"],
    )
    parser.add_argument(
        "--runtime", default=sys.executable, help="isolated Python for Dify/LiteLLM"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "runs/native-logs")
    parser.add_argument("--serializer", type=Path)
    parser.add_argument("--case", action="append", help="verify only the named producer case")
    parser.add_argument(
        "--es", help="optional real local ES URL; writes only modulo-native-* indices"
    )
    args = parser.parse_args()
    if args.action == "dify-agent":
        from native_logs.dify_agent_verify import verify

        verify(args.es)
        return
    if args.action == "verify":
        from native_logs.verify import verify

        verify(args.data, args.es, args.case)
        return
    scripts = {
        "langchain": "langchain_runs",
        "dify": "dify",
        "litellm": "litellm_logs",
        "newapi": "gateways",
        "oneapi": "gateways",
    }
    command = [args.runtime, str(ROOT / f"tests/tools/native_logs/{scripts[args.action]}.py")]
    if args.action in {"newapi", "oneapi"}:
        command += [args.action]
    command += ["--output", str(args.data)]
    if args.action == "dify":
        if args.serializer is None:
            parser.error("dify generation requires the pinned --serializer source file")
        command += ["--serializer", str(args.serializer)]
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
