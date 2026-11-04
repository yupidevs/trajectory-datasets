import logging
from typing import Any, List, Tuple

import numpy as np
from yupi import Trajectory
from yupi.generators import DiffDiffGenerator, LangevinGenerator, RandomWalkGenerator

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


# Build a new instance of the dataset
def build() -> Tuple[List[Trajectory], List[Any]]:
    trajs, labels = [], []
    model_builders = _build_langevin, _build_random_walk, _build_diffdiff
    logging.info("Generating stochastic models dataset")
    for mb in model_builders:
        tl, ll = mb(N_TRAJS)
        trajs += tl
        labels += ll
    return trajs, labels


seed = 0


def get_seed() -> int:
    global seed
    seed += 1
    return seed


def _build_langevin(n_trajs: int) -> Tuple[List[Trajectory], List[Any]]:
    rng = np.random.default_rng(seed)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    T = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    r = rng.uniform(*RANG_R_SCALE, size=n_trajs)
    sigma = rng.permutation(r) / (tau * np.sqrt(tau))
    gamma = 1 / tau

    trajs, labels = [], []
    for T_, dt_, gamma_, sigma_ in zip(T, dt, gamma, sigma):
        lg_gen = LangevinGenerator(
            T=T_,
            dim=DIM,
            N=1,
            dt=dt_,
            gamma=gamma_,
            sigma=sigma_,
            seed=get_seed(),
        )
        trajs.append(*lg_gen.generate())
        labels.append("Langevin")

    return trajs, labels


def _build_diffdiff(n_trajs: int) -> Tuple[List[Trajectory], List[Any]]:
    rng = np.random.default_rng(seed)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    T = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    r = rng.uniform(*RANG_R_SCALE, size=n_trajs)
    sigma = rng.permutation(r) / (tau * np.sqrt(tau))

    trajs, labels = [], []
    for T_, dt_, tau_, sigma_ in zip(T, dt, tau, sigma):
        dfdf_gen = DiffDiffGenerator(
            T=T_,
            dim=DIM,
            N=1,
            dt=dt_,
            tau=tau_,
            sigma=sigma_,
            seed=get_seed(),
        )
        trajs.append(*dfdf_gen.generate())
        labels.append("DiffDiff")

    return trajs, labels


def _build_random_walk(n_trajs: int) -> Tuple[List[Trajectory], List[Any]]:
    rng = np.random.default_rng(seed)

    tau = rng.uniform(*RANG_T_SCALE, size=n_trajs)
    T = tau * rng.uniform(*RANG_T, size=n_trajs)
    dt = tau * rng.uniform(*RANG_DT, size=n_trajs)
    rand_vec = rng.uniform(0, 10, size=(n_trajs, DIM, 3))
    prob = rand_vec / rand_vec.sum(-1)[..., None]

    trajs, labels = [], []
    for T_, dt_, prob_ in zip(T, dt, prob):
        rw_gen = RandomWalkGenerator(
            T=T_, dim=DIM, N=1, dt=dt_, actions_prob=prob_, seed=get_seed()
        )
        trajs.append(*rw_gen.generate())
        labels.append("RandomWalk")

    return trajs, labels


if __name__ == "__main__":
    build()
