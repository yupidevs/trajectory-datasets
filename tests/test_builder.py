import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from yupi import Trajectory
from yupi.core import JSONSerializer

from trajectory_datasets import builder

NAMES = {
    "animals",
    "cma_bst",
    "geolife",
    "hurdat2",
    "mnist_stroke",
    "stochastic_models",
    "traffic",
    "uci_characters",
    "uci_gotrack",
    "uci_movement_libras",
    "uci_pen_digits",
}


def _trajs() -> tuple[list[Trajectory], list[str]]:
    return [Trajectory(x=[0.0, 1.0], y=[0.0, 2.0]), Trajectory(x=[1.0, 2.0], y=[3.0, 4.0])], [
        "a",
        "b",
    ]


def _read(path: Path, name: str) -> dict[str, Any]:
    with zipfile.ZipFile(path / f"{name}.zip") as file:
        assert file.namelist() == [f"{name}.json"]
        data: dict[str, Any] = json.loads(file.read(f"{name}.json"))
    return data


def _fail() -> tuple[list[Trajectory], list[str]]:
    raise RuntimeError("source is down")


def test_every_recipe_is_found() -> None:
    assert set(builder.available_recipes()) == NAMES


def test_recipes_declare_a_version() -> None:
    assert all(isinstance(r.VERSION, int) for r in builder.available_recipes().values())


def test_the_published_format_is_kept(tmp_path: Path) -> None:
    builder.build_recipe(tmp_path, "demo", 3, _trajs)
    data = _read(tmp_path, "demo")
    assert sorted(data) == ["labels", "trajs", "version"]
    assert data["version"] == 3
    assert data["labels"] == ["a", "b"]
    assert [traj["id"] for traj in data["trajs"]] == ["0", "1"]
    assert JSONSerializer.from_json(data["trajs"][1]).r.tolist() == [[1.0, 3.0], [2.0, 4.0]]


def test_a_cached_dataset_is_not_rebuilt_but_still_zipped(tmp_path: Path) -> None:
    builder.build_recipe(tmp_path / "first", "demo", 1, _trajs)
    builder.build_recipe(tmp_path / "second", "demo", 1, _fail)
    assert _read(tmp_path / "second", "demo")["version"] == 1


def test_a_new_version_is_rebuilt(tmp_path: Path) -> None:
    builder.build_recipe(tmp_path, "demo", 1, _trajs)
    builder.build_recipe(tmp_path, "demo", 2, _trajs)
    assert _read(tmp_path, "demo")["version"] == 2


def test_labels_must_match_the_trajectories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be equal"):
        builder.build_recipe(tmp_path, "demo", 1, lambda: (_trajs()[0], ["a"]))


def test_trajectories_need_two_points(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2 points"):
        builder.build_recipe(tmp_path, "demo", 1, lambda: ([Trajectory(x=[0.0])], ["a"]))


def _recipes(monkeypatch: pytest.MonkeyPatch) -> None:
    recipes = {
        "good": SimpleNamespace(VERSION=0, build=_trajs),
        "bad": SimpleNamespace(VERSION=0, build=_fail),
    }
    monkeypatch.setattr(builder, "available_recipes", lambda: recipes)


def test_without_names_every_dataset_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _recipes(monkeypatch)
    assert builder.build_datasets(tmp_path, []) == ["bad"]
    assert (tmp_path / "good.zip").exists()


def test_only_the_named_datasets_are_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _recipes(monkeypatch)
    assert builder.build_datasets(tmp_path, ["good"]) == []
    assert not (tmp_path / "bad.zip").exists()


def test_unknown_names_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _recipes(monkeypatch)
    with pytest.raises(ValueError, match="nope"):
        builder.build_datasets(tmp_path, ["nope"])
