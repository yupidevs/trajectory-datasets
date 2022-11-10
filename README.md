# trajectory-datasets

A curated datasets list of raw trajectories that can be used for trajectory
classification.

## Datasets

| Name                  | Description                                           | Availability                                                                                       | Classification Goal                                                                                                      |
|-----------------------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------- |
| `geolife`             | Records of people outdoor movements                   | [microsoft.com](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52367)               | **Transportation mode:**<br> walk • bike • bus • car • subway • train • airplane • boat • run • motorcycle   |
| `mnist_stroke`        | Sequences of strokes representing handwritten digits  | [edwin-de-jong.github.io](https://edwin-de-jong.github.io/blog/mnist-sequence-data/)               | **Decimal digits:**<br> 1 • 2 • 3 • 4 • 5 • 6 • 7 • 8 • 9 • 0                                                |
| `uci_gotrack`         | Cars and buses GPS trayectories                       | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/GPS+Trajectories#)                           | **Transportation mode:**<br> bus • car                                                                       |
| `uci_pen_digits`      | Pen-Based Recognition of Handwritten Digits           | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) | **Decimal digits:**<br> 1 • 2 • 3 • 4 • 5 • 6 • 7 • 8 • 9 • 0                                                |
| `uci_characters`      | Character Trajectories Data Set                       | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Character+Trajectories)                      | **Characters:**<br> a • b • c • d • e • g • h • l • m • n • o • p • q • r • s • u • v • w • y • z            |
| `uci_movement_libras` | LIBRAS (brazilian signal language) movement dataset   | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Libras+Movement)                             | **Movement type:**<br> curved swing • horizontal swing • vertical swing • anti-clockwise arc • clockwise arc • circle • horizontal straight-line • vertical straight-line • horizontal zigzag • vertical zigzag • horizontal wavy • vertical wavy • face-up curve • face-down curve  • tremble |
| `stochastic_models`   | Trajectories generated using statistical models       | [here](recipes/stochastic_models.py)                                                               | **Model used:**<br> Random Walk • Langevin Ecquation • Diffusing Diffusivity                                 |

### Standarized versions of the above datasets

Since each dataset offers its data in a quite different format, we included
some scripts for fetching and transforming them into a standard format. This
will allow testing different analysis tools independently of the dataset.

A sandarized version is a single json file for each dataset, that contains the
following keys:
- **name:** Name of the dataset
- **version:** Integer that indicates the version
- **trajs:** List of trajectories contained in the dataset
- **labels:** List of the labels associated to each trajectory

The standarized versions of the datasets are available in the [releases
page](https://github.com/yupidevs/trajectory-datasets/releases) of this
repository. Moreover, you could generate the standarized versions yourself by
cloning this repo, and running [build.py](build.py).

## Loading trajectories from standarized datasets

Since the standarized format is a plain-text json file, it can be loaded in a
vast variety of programming languages and json-compatible tools.

However, we recommend you to use [yupi](https://github.com/yupidevs/yupi) to
load the datasets if you are using Python. A sample script could be:

```python
import json
import yupi

with open('geolife.json', "r", encoding="utf-8") as f:
    dataset = json.load(f)
    name, version = dataset['name'], dataset['version']
    trajs = [yupi.core.JSONSerializer.from_json(traj) for traj in dataset['trajs']]
    labels = dataset['labels']    
```

This approach will populate `trajs` as a list of `yupi.Trajectory` objects,
which you can use with all the resources offered by [yupi
library](https://github.com/yupidevs/yupi).

If you are planning to use a dataset for Trajectory Classification, you could
use [pactus library](https://github.com/yupidevs/pactus) instead of yupi. It is
a framework designed to evaluate Trajectory Classification methods and it is
(and will always be) compatible with all the datasets in this repository, by
simply doing:

```python
import pactus

geolife_dataset = pactus.Dataset.geolife()  
trajs, labels = geolife_dataset.trajs, geolife_dataset.labels
```

You don't need to download the dataset in advance. The library will do it for
you only the first time you use a dataset. Here, `trajs` is also a list of
`yupi.Trajectory` objects.

## Adding datasets to this repository

New datasets are always welcome to this repository. We only need to ensure that
those can be freely accessed and are relevant for raw-trajectory classification.

If you know about a potentially interesting dataset which is not already in
this repository, you can open a Github Issue providing the information and we
will integrate it as soon as possible.

Otherwise, you can integrate it yourself by:

0. Forking this project.
1. Writting a 'recipe' for the dataset. A Python script that downloads the original dataset and converts it
to the standarized version. You can take a look at the existing recipes into the [recipies folder](recipes/).
2. Store your recipe script in the [recipies folder](recipes/) and run [build.py](build.py).
3. Make sure you got no errors and [build.py](build.py) successfully generated your compressed json file in the builds folder.
4. Add the dataset metadata to the table at the begining of this README.
5. Create a pull-request from your fork.

