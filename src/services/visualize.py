from __future__ import annotations

import argparse
from typing import Sequence

DESCRIPTION = "Render Grad-CAM pairs for a day source image and a night target image."


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the Grad-CAM renderer, see --help of that tool",
    )


def run(*, args: argparse.Namespace) -> int:
    from src.utils.visualize import main as render_gradcam

    render_gradcam(list(args.forwarded))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from src.utils.visualize import main as render_gradcam

    render_gradcam(None if argv is None else list(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
