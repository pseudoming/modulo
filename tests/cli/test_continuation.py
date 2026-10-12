import json
from pathlib import Path

from modulo.cli import main
from modulo.cli.configuration import import_options
from modulo.cli.parser import parser


def test_follow_detects_new_matching_files_without_repeating_unchanged_output(
    tmp_path, unit_calculator, dify_record, monkeypatch, capsys
):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr("modulo.cli.commands.make_calculator", lambda *args: unit_calculator)
    polls = 0

    def tick(seconds):
        nonlocal polls
        polls += 1
        if polls == 1:
            (logs / "20261012.log").write_text(json.dumps(dify_record) + "\n")
        if polls == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr("modulo.cli.commands.time.sleep", tick)
    code = main(
        [
            "import",
            "--source",
            "log",
            "--adapter",
            "dify",
            "--path",
            str(logs),
            "--pattern",
            "*.log",
            "--state",
            str(tmp_path / "state"),
            "--follow",
            "--output",
            str(tmp_path / "runs"),
        ]
    )
    assert code == 130
    text = capsys.readouterr().out
    decoder = json.JSONDecoder()
    results = []
    while text.strip():
        value, end = decoder.raw_decode(text.lstrip())
        results.append(value)
        text = text.lstrip()[end:]
    assert [result["summary"]["observed_calls"] for result in results] == [0, 1]
    assert all(Path(result["run"]).is_dir() for result in results)


def test_follow_requires_state_before_reading(capsys):
    assert main(["import", "--source", "log", "--adapter", "dify", "--follow"]) == 1
    assert "requires --state" in capsys.readouterr().err


def test_state_and_limits_can_be_set_in_yaml_and_overridden(tmp_path):
    path = tmp_path / "modulo.yaml"
    path.write_text(
        "source: log\nadapter: dify\nmodels: []\nstate: progress\nfollow: true\n"
        "memory-mib: 32\nmax-records: 0\nmax-record-mib: 16\npoll-seconds: 2\n"
    )
    options = import_options(
        parser().parse_args(
            [
                "import",
                "--config",
                str(path),
                "--no-follow",
                "--memory-mib",
                "64",
            ]
        )
    )
    assert options.state == tmp_path / "progress"
    assert options.follow is False and options.memory_mib == 64
    assert options.max_records == 0 and options.max_record_mib == 16
    assert options.poll_seconds == 2
