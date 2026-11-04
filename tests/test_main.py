import sys
from pathlib import Path

import pytest

from trajectory_datasets import main as cli


def test_list_prints_the_datasets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["trajectory-datasets", "list"])
    cli.main()
    assert "geolife\n" in capsys.readouterr().out


def test_build_passes_names_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, list[str]]] = []

    def fake_build(output: Path, names: list[str]) -> list[str]:
        calls.append((output, names))
        return []

    monkeypatch.setattr(cli, "build_datasets", fake_build)
    monkeypatch.setattr(
        sys, "argv", ["trajectory-datasets", "build", "a", "b", "--output-dir", "out"]
    )
    cli.main()
    assert calls == [(Path("out"), ["a", "b"])]


def test_build_defaults_to_everything_into_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, list[str]]] = []

    def fake_build(output: Path, names: list[str]) -> list[str]:
        calls.append((output, names))
        return []

    monkeypatch.setattr(cli, "build_datasets", fake_build)
    monkeypatch.setattr(sys, "argv", ["trajectory-datasets", "build"])
    cli.main()
    assert calls == [(Path("builds"), [])]


def test_a_failed_dataset_gives_an_error_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "build_datasets", lambda output, names: ["geolife"])
    monkeypatch.setattr(sys, "argv", ["trajectory-datasets", "build"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
