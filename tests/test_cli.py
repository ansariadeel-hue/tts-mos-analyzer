import json
import subprocess
import sys
from pathlib import Path

from mos.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ratings.csv"


def test_cli_text_report(capsys):
    assert main(["report", str(EXAMPLE), "--drop-unreliable", "--bootstrap", "300"]) == 0
    out = capsys.readouterr().out
    assert "MOS listening-test report" in out
    assert "Ranking by overall" in out
    assert "1. human_dub" in out


def test_cli_json_with_normalize(capsys):
    assert main(["report", str(EXAMPLE), "--normalize", "--json", "--bootstrap", "200"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["options"]["normalize"] is True
    assert set(data["mos"]) == {"human_dub", "tts_v2", "tts_v1", "baseline"}
    assert data["dropped_raters"] == []


def test_cli_rank_by_dimension(capsys):
    assert main(["report", str(EXAMPLE), "--rank-by", "prosody", "--json", "--bootstrap", "200"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["options"]["rank_by"] == "prosody"


def test_cli_missing_file_returns_2(capsys, tmp_path):
    assert main(["report", str(tmp_path / "nope.csv")]) == 2
    assert "error" in capsys.readouterr().err


def test_module_entry_point_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "mos", "report", str(EXAMPLE), "--bootstrap", "100"],
        capture_output=True, text=True, encoding="utf-8",
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "Pairwise differences" in proc.stdout
