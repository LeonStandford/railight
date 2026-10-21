#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env import MissingEnvironmentVariable, expand_tokens, load_env, log_dir
from src.utils.logging_setup import setup_logging

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_INTERRUPTED = 130


@dataclass(frozen=True, slots=True)
class Command:
    module: str
    summary: str


COMMANDS: Mapping[str, Command] = {
    "train": Command(
        module="src.services.train",
        summary="Train RAILIGHT from a YAML config",
    ),
    "test": Command(
        module="src.services.test",
        summary="Evaluate a RAILIGHT checkpoint from a YAML config",
    ),
    "visualize": Command(
        module="src.services.visualize",
        summary="Render Grad-CAM pairs for a day and a night image",
    ),
    "convert-dataset": Command(
        module="src.services.convert_dataset",
        summary="Convert a YOLO-format dataset to RAILIGHT list files",
    ),
    "cut-frames": Command(
        module="src.services.cut_frames",
        summary="Cut frames from target-domain videos",
    ),
}


def command_table() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name:<{width}}  {command.summary}" for name, command in COMMANDS.items()]
    return "commands:\n" + "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        usage="main.py [--verbose] <command> [command options]",
        description="RAILIGHT command line. Run main.py <command> --help for the "
        "options of one command.",
        epilog=command_table(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Lower the console log level to DEBUG",
    )
    return parser


def split_argv(*, argv: Sequence[str]) -> tuple[list[str], str, list[str]]:
    for index, token in enumerate(argv):
        if token in COMMANDS:
            return list(argv[:index]), token, list(argv[index + 1 :])
    return list(argv), "", []


def start_logging(*, app_name: str, verbose: bool) -> None:
    load_env(start=ROOT)
    try:
        directory = log_dir()
    except MissingEnvironmentVariable:
        return
    if directory.resolve().is_relative_to(ROOT):
        raise SystemExit(f"log directory must live outside the project: {directory}")
    log_file = setup_logging(
        app_name=app_name,
        log_dir=directory,
        level=logging.DEBUG if verbose else logging.INFO,
    )
    logging.getLogger(__name__).info("log file %s", log_file)


def run_command(*, command: Command, argv: Sequence[str]) -> int:
    module = importlib.import_module(command.module)
    return int(module.main(list(argv)) or EXIT_OK)


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    leading, name, forwarded = split_argv(argv=tokens)
    parser = build_parser()
    if not name:
        parser.parse_args(leading)
        parser.print_help()
        return EXIT_USAGE_ERROR
    options = parser.parse_args(leading)

    start_logging(
        app_name=name.replace("-", "_"),
        verbose=options.verbose,
    )
    try:
        return run_command(
            command=COMMANDS[name],
            argv=expand_tokens(tokens=forwarded),
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("interrupted")
        return EXIT_INTERRUPTED
    except (FileNotFoundError, MissingEnvironmentVariable, ValueError) as error:
        logging.getLogger(__name__).error("%s: %s", type(error).__name__, error)
        return EXIT_USAGE_ERROR
    except SystemExit as error:
        return int(error.code or EXIT_OK)


if __name__ == "__main__":
    raise SystemExit(main())
