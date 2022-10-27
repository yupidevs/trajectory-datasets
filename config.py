"""
General configurations
"""
import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Download configs
# -----------------------------------------------------------------------------
DOWNLOAD_CHUNCK_SIZE = 4096
PROGRESS_BAR_LENGTH = 50

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------

# Dataset structure: (mainly for downloadable datasets)
#  [build dir]
#  └── datasets
#      └── [dataset_name].zip (compressed json file containing version, trajs and labels)
#
#  [cache_dir]
#  └── datasets
#      └── [dataset_name]
#          ├── [unzipped dataset files]
#          └── yupi_data.json
#

CACHE_PATH = os.environ.get("TRAJ_CACHE_PATH", str(Path(__file__).parent / ".cache"))
DS_BASE_DIR = CACHE_PATH + "/datasets"
DS_DIR = DS_BASE_DIR + "/{0}"
DS_RAW_DIR = DS_DIR + "/raw_data"
