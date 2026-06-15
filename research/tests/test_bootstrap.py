"""Tests for research/bootstrap.sh — syntax validity and the side-effect-free
subcommands. Never invokes --apply (which mutates git/exec bits). Stdlib-only."""
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bootstrap.sh"


def test_script_exists():
    assert SCRIPT.exists()


def test_bash_syntax_valid():
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_help_is_side_effect_free():
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0


def test_cron_prints_pasteable_lines():
    r = subprocess.run(["bash", str(SCRIPT), "--cron"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "crontab" in r.stdout
    assert "research/cron_runner.sh" in r.stdout
    assert "liveness_cron.sh" in r.stdout
