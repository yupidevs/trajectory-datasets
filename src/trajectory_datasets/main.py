import argparse
import logging
import sys
from pathlib import Path

from trajectory_datasets.builder import available_recipes, build_datasets

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(sys.argv[1:])
    if args.command == "list":
        sys.stdout.write("".join(f"{name}\n" for name in sorted(available_recipes())))
        return
    failed = build_datasets(args.output_dir, args.names)
    if failed:
        logger.error("Failed datasets: %s", ", ".join(failed))
        sys.exit(1)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trajectory-datasets", description="Build standardized trajectory datasets."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show the datasets that can be built")
    build = commands.add_parser("build", help="build datasets into zip files")
    build.add_argument("names", nargs="*", help="datasets to build (all of them by default)")
    build.add_argument("--output-dir", type=Path, default=Path("builds"), help="where to write zips")
    return parser.parse_args(argv)
