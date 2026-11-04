<div align="center">

<img src="assets/logo.svg" alt="" width="96">

# trajectory-datasets

**Standardized raw trajectory datasets for trajectory classification.**

[![QA](https://github.com/gvieralopez/trajectory-datasets/actions/workflows/qa.yml/badge.svg)](https://github.com/gvieralopez/trajectory-datasets/actions/workflows/qa.yml)
[![Release](https://img.shields.io/github/v/release/gvieralopez/trajectory-datasets?label=release)](https://github.com/gvieralopez/trajectory-datasets/releases/latest)
[![Python 3.15+](https://img.shields.io/badge/python-3.15%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
<br>
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

<!-- Badges to move into the rows above once they apply:
[![PyPI](https://img.shields.io/pypi/v/trajectory_datasets)](https://pypi.org/project/trajectory_datasets/)
[![Docker image](https://img.shields.io/badge/docker-ghcr.io-2496ED?logo=docker&logoColor=white)](https://github.com/gvieralopez/trajectory-datasets/pkgs/container/trajectory-datasets)
[![Docs](https://readthedocs.org/projects/trajectory-datasets/badge/?version=latest)](https://trajectory-datasets.readthedocs.io)
-->

[Datasets](#datasets) · [Using the data](#using-the-data) · [Building](#building-the-datasets) · [Releases](#how-releases-work) · [Adding a dataset](#adding-a-dataset) · [Development](#development)

</div>

---

A curated collection of raw trajectory datasets for trajectory classification. Each dataset comes
from a different source and format, and this repository converts them all to one standard format.
You can use the converted files directly, or rebuild them from the original sources.

## Datasets

| Name                  | Description                                           | Availability                                                                                       | Classification Goal                                                                                            |
|-----------------------|-------------------------------------------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------- |
| `geolife`             | Records of people outdoor movements                   | [microsoft.com](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52367)               | **Transportation mode:**<br> walk • bike • bus • car • subway • train • airplane • boat • run • motorcycle     |
| `animals`             | Elk, deer and cattle dataset. [Starkey Project](https://www.fs.usda.gov/research/treesearch/21989)         | [github.com](https://github.com/ardywibowo/RanchHand)                                              | **Animal species**<br> Elk • Deer • Cattle                                                                     |
| `hurdat2`             | Atlantic hurricane database                           | [nhc.noaa.gov](https://www.nhc.noaa.gov/data/)                                                     | **Huracane intensity (Saffir-Simpson scale):**<br> 0 • 1 • 2 • 3 • 4 • 5 <br>*(Zero means was not a huracane)* |
| `mnist_stroke`        | Sequences of strokes representing handwritten digits  | [edwin-de-jong.github.io](https://edwin-de-jong.github.io/blog/mnist-sequence-data/)               | **Decimal digits:**<br> 1 • 2 • 3 • 4 • 5 • 6 • 7 • 8 • 9 • 0                                                  |
| `cma_bst`             | Western North Pacific tropical cyclone database       | [tcdata.typhoon.org.cn](https://tcdata.typhoon.org.cn/en/zjljsjj_zlhq.html)                        | **Huracane intensity:**<br> Weaker or unknown • Tropical Depression • Tropical Storm • Severe Tropical Storm • Typhoon • Severe Typhoon • Super Typhoon • Extratropical Cyclone |
| `uci_gotrack`         | Cars and buses GPS trayectories                       | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/GPS+Trajectories#)                           | **Transportation mode:**<br> bus • car                                                                         |
| `uci_pen_digits`      | Pen-Based Recognition of Handwritten Digits           | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) | **Decimal digits:**<br> 1 • 2 • 3 • 4 • 5 • 6 • 7 • 8 • 9 • 0                                                  |
| `uci_characters`      | Character Trajectories Data Set                       | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Character+Trajectories)                      | **Characters:**<br> a • b • c • d • e • g • h • l • m • n • o • p • q • r • s • u • v • w • y • z              |
| `uci_movement_libras` | LIBRAS (brazilian signal language) movement dataset   | [ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/Libras+Movement)                             | **Movement type:**<br> curved swing • horizontal swing • vertical swing • anti-clockwise arc • clockwise arc • circle • horizontal straight-line • vertical straight-line • horizontal zigzag • vertical zigzag • horizontal wavy • vertical wavy • face-up curve • face-down curve  • tremble |
| `traffic` | Traffic dataset over a road section | [zen-traffic-data](https://zen-traffic-data.net/english/outline/dataset.html)                             | **Vehicle type:**<br> normal • large |
| `stochastic_models`   | Trajectories generated using statistical models       | [here](src/trajectory_datasets/recipes/stochastic_models.py)                                                               | **Model used:**<br> Random Walk • Langevin Ecquation • Diffusing Diffusivity                                   |

### Where to download them

Each dataset is a zip file attached to a [GitHub release](https://github.com/yupidevs/trajectory-datasets/releases).
A release only contains the datasets built for it, so the latest copy of a dataset can be in an older release.

| Dataset | Latest published in |
|---|---|
| `traffic` | [v0.2.0](https://github.com/yupidevs/trajectory-datasets/releases/tag/v0.2.0) |
| All the others | [v0.1.0b2](https://github.com/yupidevs/trajectory-datasets/releases/tag/v0.1.0b2) |

## Using the data

### The standard format

Each zip file contains one JSON file, `<dataset name>.json`, with these keys:

| Key | Content |
|---|---|
| `version` | Integer, the version of the recipe that built the dataset |
| `trajs` | List of trajectories, written with yupi's `JSONSerializer` |
| `labels` | List with the label of each trajectory, in the same order |

Being plain JSON, the files can be read from any language.

### With pactus

For trajectory classification, [pactus](https://github.com/yupidevs/pactus) downloads the datasets for you the
first time you use them:

```python
import pactus

geolife_dataset = pactus.Dataset.geolife()
trajs, labels = geolife_dataset.trajs, geolife_dataset.labels
```

### With yupi

```python
import json

from yupi.core import JSONSerializer

with open("geolife.json", encoding="utf-8") as file:
    dataset = json.load(file)

trajs = [JSONSerializer.from_json(traj) for traj in dataset["trajs"]]
labels = dataset["labels"]
```

In both cases, `trajs` is a list of `yupi.Trajectory` objects.

## Building the datasets

Most people don't need this, because the built files are already published. It's for updating a dataset or
checking how one is made.

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.15.
- `unrar` on your system, for the datasets distributed as `.rar` (`sudo apt-get install unrar` on Debian or Ubuntu).
- Enough disk space: the raw data of some datasets, such as `geolife` and `mnist_stroke`, is large.

### Commands

```bash
git clone https://github.com/yupidevs/trajectory-datasets.git
cd trajectory-datasets
uv sync

uv run trajectory-datasets list                     # show the datasets that can be built
uv run trajectory-datasets build hurdat2 traffic    # build some of them
uv run trajectory-datasets build                    # build all of them
uv run trajectory-datasets build --output-dir out   # write the zips somewhere else (default: builds/)
```

If a dataset fails, usually because its source is offline, the others still build, and the command lists the
failures and exits with an error.

### The cache

Downloads and converted data are kept in `.cache/` in the current folder. Set the `TRAJ_CACHE_PATH` environment
variable to use another location.

A dataset is converted again only if its recipe's `VERSION` is higher than the cached one. Otherwise the zip is
written from the cache, so delete the dataset's cache folder to force a rebuild.

### Sources that have changed

- `mnist_stroke`: the labels are no longer on Yann LeCun's site, so they come from a mirror of the same files.
- `cma_bst`: its source could not be reached when this was last checked (November 2026), so the published file is
  the one to use for now.

### Known limitations of the published files

These come from the recipes, and changing them would change the published data, so they need a new `VERSION`:

- `hurdat2`, `cma_bst` and `traffic` leave out the last storm or vehicle of each source file.
- `cma_bst` keeps latitude and longitude as written in its source, in tenths of a degree.
- `hurdat2` does not make southern latitudes negative. Its Atlantic data is in the northern hemisphere.
- `geolife` measures time within each labelled segment, and wraps around if one lasts more than a day.

## How releases work

Code and data are released separately:

- **Code:** every merge to `main` that passes QA creates a new GitHub release automatically, usually with no files.
- **Data:** datasets are built only on request. In the repository's **Actions** tab, open **Build datasets**, choose
  **Run workflow**, and type the datasets to build separated by spaces, or leave it empty to build all of them.
  The zips are attached to the latest release, replacing any with the same name. If a dataset fails, the ones
  that worked are still attached, and the run is marked as failed.

### How the files are found

[pactus](https://github.com/yupidevs/pactus) goes through the tags from newest to oldest and uses the first release
that has `<dataset name>.zip`. So:

- Attaching a new zip to the latest release is all it takes to update a dataset.
- **Don't delete old releases or rename the zip files.** Most datasets live only in older releases.

## Adding a dataset

New datasets are welcome, as long as they can be freely accessed and are useful for raw trajectory classification.
You can [open an issue](https://github.com/yupidevs/trajectory-datasets/issues) with the details, or add it yourself:

1. Fork this repository.
2. Add a recipe in [`src/trajectory_datasets/recipes/`](src/trajectory_datasets/recipes/). A recipe is a module with:

   | Name | What it is |
   |---|---|
   | `NAME` | The dataset name, which is also the name of the zip file |
   | `VERSION` | An integer. Increase it whenever the output changes |
   | `build()` | Returns `(trajectories, labels)`, a list of `yupi.Trajectory` and a list of the same length |

   Every trajectory needs at least two points. `trajectory_datasets.downloads.download_dataset` downloads and
   extracts the raw files for you.
3. Add tests in [`tests/recipes/`](tests/recipes/) that recreate a small sample of the raw format.
4. Run `uv run trajectory-datasets build <name>` and check the zip in `builds/`.
5. Add the dataset to the table at the top of this README and open a pull request.

## Development

Setup, QA, docs and releases: [DEVELOPMENT.md](DEVELOPMENT.md).

## Contributing

Issues and pull requests are welcome; `make qa` must pass.

## License

The code is MIT: see [LICENSE](LICENSE). Each dataset keeps the terms of its original source, linked in the [table above](#datasets).

---

<sub>🚀 Project starter provided by [Cookie Pyrate v1.1.2](https://github.com/gvieralopez/cookie-pyrate/tree/v1.1.2)</sub>
