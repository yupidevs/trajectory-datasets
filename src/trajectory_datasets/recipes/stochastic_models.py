import logging
from typing import Any

import numpy as np
from yupi import Trajectory
from yupi.generators import DiffDiffGenerator, LangevinGenerator, RandomWalkGenerator

logger = logging.getLogger(__name__)

# Dataset metadata
NAME = "stochastic_models"
VERSION = 0

# Trajectory-specific parameters
DIM = 2
RANG_T_SCALE = [1, 100]
RANG_R_SCALE = [0.1, 10]
RANG_DT = [1e-3, 1e-1]
RANG_T = [8, 20]
N_TRAJS = 1000


class _Seeds:
    def __init__(self) -> None:
        self.current = 0

    def next(self) -> int:
        self.current += 1
        return self.current


# Build a new instance of the dataset
def build() -> tuple[list[Trajectory], list[Any]]:
    trajs: list[Trajectory] = []
    labels: list[Any] = []
    seeds = _Seeds()
    model_builders = _build_langevin, _build_random_walk, _build_diffdiff
    logger.info("Generating stochastic models dataset")
    for mb in model_builders:
        tl, ll = mb(N_TRAJS, seeds)
        trajs += tl
        labels += ll
    return trajs, labels


def _build_langevin(n_trajs: int, seeds: _Seeds) -> tuple[list[Trajectory], list[Any]]:
    rng = np.random.default_rng(seeds.current)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    total = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    r = rng.uniform(*RANG_R_SCALE, size=n_trajs)
    sigma = rng.permutation(r) / (tau * np.sqrt(tau))
    gamma = 1 / tau

    trajs: list[Trajectory] = []
    labels: list[Any] = []
    for duration, dt_, gamma_, sigma_ in zip(total, dt, gamma, sigma, strict=True):
        lg_gen = LangevinGenerator(
            T=duration, dim=DIM, dt=dt_, gamma=gamma_, sigma=sigma_, seed=seeds.next()
        )
        trajs.append(*lg_gen.generate(1))
        labels.append("Langevin")

    return trajs, labels


def _build_diffdiff(n_trajs: int, seeds: _Seeds) -> tuple[list[Trajectory], list[Any]]:
    rng = np.random.default_rng(seeds.current)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    total = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    r = rng.uniform(*RANG_R_SCALE, size=n_trajs)
    sigma = rng.permutation(r) / (tau * np.sqrt(tau))

    trajs: list[Trajectory] = []
    labels: list[Any] = []
    for duration, dt_, tau_, sigma_ in zip(total, dt, tau, sigma, strict=True):
        dfdf_gen = DiffDiffGenerator(
            T=duration, dim=DIM, dt=dt_, tau=tau_, sigma=sigma_, seed=seeds.next()
        )
        trajs.append(*dfdf_gen.generate(1))
        labels.append("DiffDiff")

    return trajs, labels


def _build_random_walk(n_trajs: int, seeds: _Seeds) -> tuple[list[Trajectory], list[Any]]:
    rng = np.random.default_rng(seeds.current)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    total = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    rand_vec = rng.uniform(0, 10, size=(n_trajs, DIM, 3))
    prob = rand_vec / rand_vec.sum(-1)[..., None]

    trajs: list[Trajectory] = []
    labels: list[Any] = []
    for duration, dt_, prob_ in zip(total, dt, prob, strict=True):
        rw_gen = RandomWalkGenerator(
            T=duration, dim=DIM, dt=dt_, actions_prob=prob_, seed=seeds.next()
        )
        trajs.append(*rw_gen.generate(1))
        labels.append("RandomWalk")

    return trajs, labels
