from pathlib import Path

import pytest

from trajectory_datasets import config


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DS_DIR", str(tmp_path / "cache" / "{0}"))
    monkeypatch.setattr(config, "DS_RAW_DIR", str(tmp_path / "cache" / "{0}" / "raw_data"))


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "raw"
    folder.mkdir()
    return folder
