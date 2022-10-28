from random import Random
from typing import Any, List, Tuple

from yupi import Trajectory
from yupi.generators import LangevinGenerator, DiffDiffGenerator, RandomWalkGenerator


# Dataset metadata
NAME = "stochastic_models"
VERSION = 1

# Trajectory-specific parameters
DIMENSIONS = 3
AVERAGE_DURATION = 10
AVERAGE_SAMPLE_TIME = 0.1
DURATION_STD = 2
SAMPLE_TIME_STD = 0.01
TRAJECTORIES_PER_MODEL = 1000

# Model-specific parameters (TODO: UPDATE with meaninful values)
LANGEVIN_MIN_SIGMA = 0.01
LANGEVIN_MAX_SIGMA = 0.1
LANGEVIN_MIN_GAMMA = 0.01
LANGEVIN_MAX_GAMMA = 0.1
DIFFDIFF_MIN_SIGMA = 0.01
DIFFDIFF_MAX_SIGMA = 0.1
DIFFDIFF_MIN_TAU = 0.01
DIFFDIFF_MAX_TAU = 0.1


# Build a new instance of the dataset
def build() -> Tuple[List[Trajectory], List[Any]]:
    trajs, labels = [], []
    model_builders = _build_langevin, _build_random_walk, _build_diffdiff
    for mb in model_builders:
        tl, ll = mb(TRAJECTORIES_PER_MODEL)
        trajs += tl
        labels += ll
    #TODO: Shuffle before return
    return trajs, labels


seed = 0


def get_seed() -> int:
    global seed
    seed += 1
    return seed


def _build_langevin(count: int) -> Tuple[List[Trajectory], List[Any]]:
    trajs, labels = [], []

    rng = Random(0)

    for _ in range(count):
        gamma = rng.uniform(LANGEVIN_MIN_GAMMA, LANGEVIN_MAX_GAMMA)
        sigma = rng.uniform(LANGEVIN_MIN_SIGMA, LANGEVIN_MAX_SIGMA)
        dt = abs(rng.normalvariate(AVERAGE_SAMPLE_TIME, SAMPLE_TIME_STD))
        T = abs(rng.normalvariate(AVERAGE_DURATION, DURATION_STD))
        lg_gen = LangevinGenerator(
            dim=DIMENSIONS,
            N=1,
            dt=dt,
            T=T,
            seed=get_seed(),
            gamma=gamma,
            sigma=sigma,
        )
        trajs.append(lg_gen.generate()[0])
        labels.append("Langevin")

    return trajs, labels


def _rand_unit_vect(rng: Random, size=3) -> List[float]:
    random_vector = [rng.uniform(0, 10) for i in range(size)]
    normalized = [i/sum(random_vector) for i in random_vector] 
    normalized[-1] += 1 - sum(normalized)
    return normalized


def _build_random_walk(count: int) -> Tuple[List[Trajectory], List[Any]]:
    trajs, labels = [], []

    rng = Random(0)
    for _ in range(count):
        dt = abs(rng.normalvariate(AVERAGE_SAMPLE_TIME, SAMPLE_TIME_STD))
        T = abs(rng.normalvariate(AVERAGE_DURATION, DURATION_STD))
        prob = [_rand_unit_vect(rng) for i in range(DIMENSIONS)]
        lg_gen = RandomWalkGenerator(dim=DIMENSIONS, N=1, dt=dt, T=T, seed=get_seed(), actions_prob=prob)
        trajs.append(lg_gen.generate()[0])
        labels.append("RandomWalk")

    return trajs, labels


def _build_diffdiff(count: int) -> Tuple[List[Trajectory], List[Any]]:
    trajs, labels = [], []

    rng = Random(0)

    for _ in range(count):
        tau = rng.uniform(DIFFDIFF_MIN_TAU, DIFFDIFF_MAX_TAU)
        sigma = rng.uniform(DIFFDIFF_MIN_SIGMA, DIFFDIFF_MAX_SIGMA)
        dt = abs(rng.normalvariate(AVERAGE_SAMPLE_TIME, SAMPLE_TIME_STD))
        T = abs(rng.normalvariate(AVERAGE_DURATION, DURATION_STD))
        lg_gen = DiffDiffGenerator(
            dim=DIMENSIONS,
            N=1,
            dt=dt,
            T=T,
            seed=get_seed(),
            tau=tau,
            sigma=sigma,
        )
        trajs.append(lg_gen.generate()[0])
        labels.append("DiffDiff")

    return trajs, labels

if __name__ == '__main__':
    build()