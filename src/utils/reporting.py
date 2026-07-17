from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

VIZ_METHOD: str = "RAILIGHT (railway, real-target dark)"
SECTION_MARKER: str = "§"


def viz_method() -> str:
    """Name of the method, printed on every chart."""
    return VIZ_METHOD


def viz_config(
    args_ns: argparse.Namespace,
    backbone: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run description shown in chart captions."""
    out: Dict[str, Any] = {
        "backbone": backbone or getattr(args_ns, "backbone", None) or args_ns.model,
        "exp": args_ns.num_exp,
        "batch": args_ns.batch_size,
        "nc": args_ns.nc,
        "dark_src": "synthetic",
    }
    if extra:
        out.update(extra)
    return out


def format_val_table(title: str, rows: List[Tuple[str, str]]) -> str:
    """Render rows as a box table; a row keyed ``§`` becomes a section header."""
    data = [(k, v) for (k, v) in rows if k != SECTION_MARKER]
    label_w = max((len(k) for (k, _) in data), default=4)
    value_w = max((len(v) for (_, v) in data), default=4)
    sect_w = max((len(v) for (k, v) in rows if k == SECTION_MARKER), default=0)
    content = max(label_w + 3 + value_w, sect_w, len(title))
    inner = content + 2

    top = "╔" + "═" * inner + "╗"
    mid = "╠" + "═" * inner + "╣"
    sep = "╟" + "─" * inner + "╢"
    bot = "╚" + "═" * inner + "╝"

    lines = [top, "║ " + title.center(content) + " ║", mid]
    first_section = True
    for k, v in rows:
        if k == SECTION_MARKER:
            if not first_section:
                lines.append(sep)
            lines.append("║ " + v.ljust(content) + " ║")
            lines.append(sep)
            first_section = False
            continue
        body = f"{k:<{label_w}s} : {v:>{value_w}s}"
        lines.append("║ " + body.ljust(content) + " ║")
    lines.append(bot)

    return "\n".join(lines)
