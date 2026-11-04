"""
General configurations
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Download configs
# -----------------------------------------------------------------------------
DOWNLOAD_CHUNCK_SIZE = 4096
DOWNLOAD_TIMEOUT_SECONDS = 60
PROGRESS_BAR_LENGTH = 50

# -----------------------------------------------------------------------------
# Dataset configs
# -----------------------------------------------------------------------------

CACHE_PATH = os.environ.get("TRAJ_CACHE_PATH", str(Path.cwd() / ".cache"))
DS_BASE_DIR = CACHE_PATH + "/datasets"
DS_DIR = DS_BASE_DIR + "/{0}"
DS_RAW_DIR = DS_DIR + "/raw_data"
