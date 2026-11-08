import numpy as np
import pytest

from trajectory_datasets.recipes import stochastic_models


@pytest.fixture(autouse=True)
def few_trajectories(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stochastic_models, "N_TRAJS", 2)


def test_every_model_is_generated() -> None:
    trajs, labels = stochastic_models.build()
    assert labels == ["Langevin"] * 2 + ["RandomWalk"] * 2 + ["DiffDiff"] * 2
    assert all(traj.dim == stochastic_models.DIM for traj in trajs)


def test_builds_are_reproducible() -> None:
    first, _ = stochastic_models.build()
    second, _ = stochastic_models.build()
    assert all(np.array_equal(a.r, b.r) for a, b in zip(first, second, strict=True))
