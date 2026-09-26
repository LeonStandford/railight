from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TEX = Path("/home/a00161/stacy.en14/models/railight/charts/test/class_wise.tex")
BASE = "/home/a00161/stacy.en14/models/railight/charts/test/railight/vgg16"
P4 = "exp4.uda.batch8.weak_strong_augmentation"
P3 = "exp3.da.batch8"

TABLES = {
    "tab:class_wise_night_supervised": ("target",
    {
        "RAILIGHT": f"{P3}.caotulab_server.h100",
        "+ W/S": f"{P3}.weak_strong_augmentation.caotulab_server.h100",
        "+ W/S + CAT": f"{P3}.weak_strong_augmentation.cat.caotulab_server.h100",
        "+ W/S + ConCal": f"{P3}.weak_strong_augmentation.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CDRC": f"{P3}.weak_strong_augmentation.cdrc.caotulab_server.h100",
        "+ W/S + CAT + ConCal": f"{P3}.weak_strong_augmentation.cat.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CAT + CDRC": f"{P3}.weak_strong_augmentation.cat.cdrc.caotulab_server.h100",
        "+ W/S + ConCal + CDRC": f"{P3}.weak_strong_augmentation.cdrc.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CAT + ConCal + CDRC": f"{P3}.weak_strong_augmentation.cat.cdrc.concal.caotulab_server.h100_hpd_off",
        "RAILIGHT + Adam": f"{P3}.caotulab_server.h100_adam",
        "+ W/S + Adam": f"{P3}.weak_strong_augmentation.caotulab_server.h100_adam",
        "+ W/S + CAT + Adam": f"{P3}.weak_strong_augmentation.cat.caotulab_server.h100_adam",
        "+ W/S + ConCal + Adam": f"{P3}.weak_strong_augmentation.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CDRC + Adam": f"{P3}.weak_strong_augmentation.cdrc.caotulab_server.h100_adam",
        "+ W/S + CAT + ConCal + Adam": f"{P3}.weak_strong_augmentation.cat.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CAT + CDRC + Adam": f"{P3}.weak_strong_augmentation.cat.cdrc.caotulab_server.h100_adam",
        "+ W/S + ConCal + CDRC + Adam": f"{P3}.weak_strong_augmentation.cdrc.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CAT + ConCal + CDRC + Adam": f"{P3}.weak_strong_augmentation.cat.cdrc.concal.caotulab_server.h100_hpd_off_adam",
        "+ Bal + Adam": f"{P3}.caotulab_server.h100_adam_balanced",
        "+ Bal + CAT + Adam": f"{P3}.cat.caotulab_server.h100_adam_balanced",
        "+ Bal + ConCal + Adam": f"{P3}.concal.caotulab_server.h100_hpd_off_adam_balanced",
        "+ Bal + CAT + ConCal + Adam": f"{P3}.cat.concal.caotulab_server.h100_hpd_off_adam_balanced",
    }),
    "tab:class_wise_night_adapted": ("target", {
        "RAILIGHT": "exp4.uda.batch8.caotulab_server.h100",
        "+ weak/strong": f"{P4}.caotulab_server.h100",
        "+ CAT": f"{P4}.cat.caotulab_server.h100_fix_icrm",
        "+ ConCal": f"{P4}.concal.caotulab_server.h100_hpd_off",
        "+ CDRC": f"{P4}.cdrc.caotulab_server.h100",
        "+ CAT + ConCal": f"{P4}.cat.concal.caotulab_server.h100_hpd_off",
        "+ CAT + CDRC": f"{P4}.cat.cra_cdrc_bidirection_mixing.caotulab_server.h100",
        "+ ConCal + CDRC": f"{P4}.cdrc.concal.caotulab_server.h100_hpd_off",
        "+ CAT + ConCal + CDRC": f"{P4}.cat.cdrc.concal.caotulab_server.h100_hpd_off",
        "RAILIGHT + Adam": "exp4.uda.batch8.caotulab_server.h100_adam",
        "+ weak/strong + Adam": f"{P4}.caotulab_server.h100_adam",
        "+ CAT + Adam": f"{P4}.cat.caotulab_server.h100_fix_icrm_adam",
        "+ ConCal + Adam": f"{P4}.concal.hpd_off.caotulab_server.h100_adam",
        "+ CDRC + Adam": f"{P4}.cdrc.caotulab_server.h100_adam",
        "+ CAT + ConCal + Adam": f"{P4}.cat.concal.caotulab_server.h100_hpd_off_adam",
        "+ CAT + CDRC + Adam": f"{P4}.cat.cra_cdrc_bidirection_mixing.caotulab_server.h100_adam_v2",
        "+ ConCal + CDRC + Adam": f"{P4}.cdrc.concal.caotulab_server.h100_hpd_off_adam",
        "+ CAT + ConCal + CDRC + Adam": f"{P4}.cat.cdrc.concal.caotulab_server.h100_hpd_off_adam",
    }),
    "tab:class_wise_day_supervised": ("source",
    {
        "RAILIGHT": f"{P3}.caotulab_server.h100",
        "+ W/S": f"{P3}.weak_strong_augmentation.caotulab_server.h100",
        "+ W/S + CAT": f"{P3}.weak_strong_augmentation.cat.caotulab_server.h100",
        "+ W/S + ConCal": f"{P3}.weak_strong_augmentation.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CDRC": f"{P3}.weak_strong_augmentation.cdrc.caotulab_server.h100",
        "+ W/S + CAT + ConCal": f"{P3}.weak_strong_augmentation.cat.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CAT + CDRC": f"{P3}.weak_strong_augmentation.cat.cdrc.caotulab_server.h100",
        "+ W/S + ConCal + CDRC": f"{P3}.weak_strong_augmentation.cdrc.concal.caotulab_server.h100_hpd_off",
        "+ W/S + CAT + ConCal + CDRC": f"{P3}.weak_strong_augmentation.cat.cdrc.concal.caotulab_server.h100_hpd_off",
        "RAILIGHT + Adam": f"{P3}.caotulab_server.h100_adam",
        "+ W/S + Adam": f"{P3}.weak_strong_augmentation.caotulab_server.h100_adam",
        "+ W/S + CAT + Adam": f"{P3}.weak_strong_augmentation.cat.caotulab_server.h100_adam",
        "+ W/S + ConCal + Adam": f"{P3}.weak_strong_augmentation.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CDRC + Adam": f"{P3}.weak_strong_augmentation.cdrc.caotulab_server.h100_adam",
        "+ W/S + CAT + ConCal + Adam": f"{P3}.weak_strong_augmentation.cat.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CAT + CDRC + Adam": f"{P3}.weak_strong_augmentation.cat.cdrc.caotulab_server.h100_adam",
        "+ W/S + ConCal + CDRC + Adam": f"{P3}.weak_strong_augmentation.cdrc.concal.caotulab_server.h100_hpd_off_adam",
        "+ W/S + CAT + ConCal + CDRC + Adam": f"{P3}.weak_strong_augmentation.cat.cdrc.concal.caotulab_server.h100_hpd_off_adam",
        "+ Bal + Adam": f"{P3}.caotulab_server.h100_adam_balanced",
        "+ Bal + CAT + Adam": f"{P3}.cat.caotulab_server.h100_adam_balanced",
        "+ Bal + ConCal + Adam": f"{P3}.concal.caotulab_server.h100_hpd_off_adam_balanced",
        "+ Bal + CAT + ConCal + Adam": f"{P3}.cat.concal.caotulab_server.h100_hpd_off_adam_balanced",
    }),
}

YES, NO, NA = "\\checkmark", "", "--"
MARKUP = re.compile(r"\\(?:textbf|underline)\{([^}]*)\}")
FLAG_COLS = 6


def flags(num_exp: str) -> List[str]:
    parts = num_exp.split(".")
    concal = any(p == "concal" or p.startswith("concal") for p in parts)
    cdrc = "cdrc" in parts or "cra_cdrc_bidirection_mixing" in parts
    return [
        YES if "weak_strong_augmentation" in parts else NO,
        YES if "cat" in parts else NO,
        YES if concal else NO,
        YES if cdrc else NO,
        YES if "_balanced" in num_exp else NO,
        "Adam" if "_adam" in num_exp else "SGD",
    ]


def strip(cell: str) -> str:
    return MARKUP.sub(r"\1", cell).strip()


def rebuild(text: str, label: str, split: str, registry: Dict[str, str]) -> str:
    anchor = text.find(f"\\label{{{label}}}")
    start = text.rfind("\\begin{table*}", 0, anchor)
    stop = text.find("\\end{table*}", anchor) + len("\\end{table*}")
    block = text[start:stop]
    lines = block.splitlines()

    head = next(i for i, l in enumerate(lines) if l.startswith("Method &"))
    header = [strip(c) for c in lines[head].rstrip(" \\\\").split("&")]
    FLAG_NAMES = {"W/S", "Aug", "CAT", "ConCal", "CDRC", "HPD", "Bal", "Balanced", "Optim"}
    old_flags = 0
    while 2 + old_flags < len(header) and header[2 + old_flags] in FLAG_NAMES:
        old_flags += 1
    if old_flags:
        for i, line in enumerate(lines):
            if not (line.rstrip().endswith("\\\\") and " & " in line):
                continue
            raw = line.rstrip()[:-2].rstrip()
            prefix = ""
            if raw.startswith("\\rowcolor{railightrow}"):
                prefix = "\\rowcolor{railightrow}"
                raw = raw[len(prefix):]
            cells = [c.strip() for c in raw.split("&")]
            cells = cells[:2] + cells[2 + old_flags:]
            lines[i] = prefix + " & ".join(cells) + " \\\\"
        lines = [
            re.sub(r"\{ll" + "c" * old_flags + r"(c+)\}",
                   lambda m: "{ll" + m.group(1) + "}", l)
            if l.startswith("\\begin{tabular}") else l
            for l in lines
        ]
        header = [strip(c) for c in lines[head].rstrip(" \\\\").split("&")]
    names = header[2:]
    lines[head] = (
        "Method & Detector & W/S & CAT & ConCal & CDRC & Balanced & Optim & "
        + " & ".join(names) + " \\\\"
    )
    lines = [
        re.sub(r"\{ll(c+)\}", lambda m: "{llcccccc" + m.group(1) + "}", l)
        if l.startswith("\\begin{tabular}") else l
        for l in lines
    ]

    out: List[str] = []
    missing: List[str] = []
    seen_colour = False
    for line in lines:
        if not (line.rstrip().endswith("\\\\") and " & " in line) or line.startswith("Method &"):
            out.append(line)
            continue
        if line.startswith("\\rowcolor{railightrow}"):
            if seen_colour:
                continue
            seen_colour = True
            for name, num_exp in registry.items():
                path = f"{BASE}/{num_exp}/metrics.json"
                if not os.path.isfile(path):
                    missing.append(name)
                    cells = (["RAILIGHT", "DSFD-VGG16"] + flags(num_exp)
                             + [NO] * 9)
                    out.append("\\rowcolor{railightrow}" + " & ".join(cells) + " \\\\")
                    continue
                section = json.load(open(path))[split]
                per = [f"{v['ap50'] * 100:.1f}" for v in section["per_class"].values()]
                cells = (["RAILIGHT", "DSFD-VGG16"] + flags(num_exp) + per
                         + [f"{section['macro_map50'] * 100:.1f}"])
                out.append("\\rowcolor{railightrow}" + " & ".join(cells) + " \\\\")
            continue
        raw = line.rstrip()[:-2].rstrip()
        cells = [strip(c) for c in raw.split("&")]
        cells = cells[:2] + [NA] * FLAG_COLS + cells[2:]
        out.append(" & ".join(cells) + " \\\\")

    data = [i for i, l in enumerate(out)
            if l.rstrip().endswith("\\\\") and " & " in l and not l.startswith("Method &")]
    parsed: Dict[int, Tuple[str, List[str]]] = {}
    for i in data:
        raw = out[i].rstrip()[:-2].rstrip()
        prefix = ""
        if raw.startswith("\\rowcolor{railightrow}"):
            prefix = "\\rowcolor{railightrow}"
            raw = raw[len(prefix):]
        parsed[i] = (prefix, [c.strip() for c in raw.split("&")])
    width = len(next(iter(parsed.values()))[1])
    for col in range(2 + FLAG_COLS, width):
        vals: Dict[int, float] = {}
        for i, (_p, cells) in parsed.items():
            try:
                vals[i] = float(strip(cells[col]))
            except ValueError:
                pass
        if not vals:
            continue
        ranked = sorted(set(vals.values()), reverse=True)
        best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        for i, v in vals.items():
            plain = strip(parsed[i][1][col])
            if v == best:
                parsed[i][1][col] = "\\textbf{%s}" % plain
            elif v == second:
                parsed[i][1][col] = "\\underline{%s}" % plain
            else:
                parsed[i][1][col] = plain
    for i, (prefix, cells) in parsed.items():
        out[i] = prefix + " & ".join(cells) + " \\\\"
    if missing:
        print(f"   {label}: chờ kết quả {missing}")
    return text[:start] + "\n".join(out) + text[stop:]


def main() -> None:
    text = TEX.read_text()
    for lbl in ("tab:config_night_supervised", "tab:config_night_adapted",
                "tab:config_day_supervised"):
        while True:
            m = re.search(r"\n?\\begin\{table\}(?:(?!\\end\{table\}).)*?"
                          + re.escape(f"\\label{{{lbl}}}")
                          + r"(?:(?!\\end\{table\}).)*?\\end\{table\}", text, re.S)
            if not m:
                break
            text = text[:m.start()] + text[m.end():]
    if "amssymb" not in text:
        text = text.replace("\\usepackage{booktabs}",
                            "\\usepackage{amssymb}\n\\usepackage{booktabs}", 1)
    if "\\captionsetup" not in text:
        text = text.replace(
            "\\pagestyle{empty}",
            "\\usepackage{caption}\n"
            "\\captionsetup{skip=2pt,font=small}\n"
            "\\setlength{\\textfloatsep}{4pt}\n"
            "\\setlength{\\floatsep}{4pt}\n"
            "\\setlength{\\intextsep}{4pt}\n"
            "\\setlength{\\tabcolsep}{2pt}\n"
            "\\renewcommand{\\arraystretch}{0.82}\n"
            "\\pagestyle{empty}", 1)
    text = text.replace("\\begin{table*}[t]", "\\begin{table*}[!htbp]")
    for label, (split, registry) in TABLES.items():
        if f"\\label{{{label}}}" not in text:
            print(f"không thấy {label}")
            continue
        text = rebuild(text, label, split, registry)
        print(f"   {label}: xong")
    TEX.write_text(text if text.endswith("\n") else text + "\n")


if __name__ == "__main__":
    main()
