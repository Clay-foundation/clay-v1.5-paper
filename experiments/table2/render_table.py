#!/usr/bin/env python3
"""Render the body of Table 2 from table2_cells.csv into DRAFT.md and latex/sections/02_model.tex.

    python3 experiments/table2/render_table.py            # check: the files hold exactly what the CSV renders
    python3 experiments/table2/render_table.py --write    # replace the table body in both files
    python3 experiments/table2/render_table.py --print md|tex

Only the table body (header row to last summary row) is generated; the caption and the footnote are written by hand.
% Best and the summary rows come from summarize.py, so the printed numbers and the aggregates share one code path.
Every peer tag printed in the table is checked against the footnote of both files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize as sm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRAFT = ROOT / "DRAFT.md"
TEX = ROOT / "latex" / "sections" / "02_model.tex"
CSV = HERE / "table2_cells.csv"

HEADER = ["Benchmark", "Sensor and metric", "Clay v1.5", "% Best", "Best other EO", "Best general"]
# CSV panel codes A-D are blocks; the table prints two panels: Panel A = blocks A, B (SatML leaderboard),
# Panel B = blocks C, D (benchmarks used by peers). A block title that opens a printed panel starts with "Panel".
PANEL_TITLE = {
    "A": "Panel A: SatML leaderboard. Classification, linear probe, 15 core datasets",
    "B": "Segmentation, UPerNet head (Clay: our runs in the leaderboard's harness)",
    "C": "Panel B: Benchmarks used by peers (our runs). OlmoEarth v1.3, GEO-Bench and Copernicus-Bench tasks, linear probe or fine-tune",
    "D": "AlphaEarth / TESSERA suite, 15 tasks, 10-shot / Max",
}
DEC_ONE = {"A", "B"}  # panels whose values are formatted from the leaderboard floats at one decimal


class Md:
    sep = " / "

    def code(self, s):
        return f"`{s}`"

    def bold(self, s):
        return f"**{s}**"

    def pm(self):
        return " ± "

    def num(self, s):
        return s

    def mark(self, m):
        return {"♮": "♮", "◊": "◊", "★": "\\star", "†": "†", "‡": "‡", "∗": "\\*", "§": "§", "♭": "♭"}[m]

    def marks(self, ms):
        return "".join(self.mark(m) for m in ms)

    def dash(self):
        return "–"

    def pct(self, s):
        return s

    def text(self, s):
        return s

    def panel(self, title):
        return "| " + " | ".join([f"*{title}*"] + [""] * (len(HEADER) - 1)) + " |"

    def row(self, cells):
        return "| " + " | ".join(cells) + " |"

    def head(self):
        return [self.row(HEADER), "|---|---|---:|---:|---|---|---|"]

    def table(self, lines):
        return lines


class Tex(Md):
    sep = " \\slsh{} "

    def code(self, s):
        s = s.replace("_", "\\_").replace(". (", ".\\ (")
        return f"\\texttt{{{s}}}"

    def bold(self, s):
        return f"\\textbf{{{s}}}"

    def pm(self):
        return " $\\pm$ "

    def num(self, s):
        return ("\\text{--}" + s[1:]) if s.startswith("-") else s

    def mark(self, m):
        return {"♮": "\\natural", "◊": "\\diamond", "★": "\\star", "†": "\\dagger", "‡": "\\ddagger", "∗": "\\ast", "§": "\\S", "♭": "\\flat"}[m]

    def marks(self, ms):
        if not ms:
            return ""
        body = "".join(self.mark(m) for m in ms)
        return f"$^{{{body}}}$" if len(ms) > 1 else f"$^{body}$"

    def dash(self):
        return "--"

    def pct(self, s):
        return s.replace("%", "\\%")

    def text(self, s):
        return s.replace("&", "\\&").replace("%", "\\%")

    def panel(self, title):
        return f"\\multicolumn{{{len(HEADER)}}}{{@{{}}l@{{}}}}{{\\emph{{{title}}}}} \\\\"

    def row(self, cells):
        return " & ".join(cells) + " \\\\"

    def head(self):
        return ["\\toprule", " & ".join([HEADER[0], HEADER[1], f"\\textbf{{{HEADER[2]}}}", "\\textbf{\\% Best}",
                                        HEADER[4], HEADER[5]]) + " \\\\", "\\midrule"]


def note(r, key):
    return sm.csv_note(r, key) if hasattr(sm, "csv_note") else next(
        (p[len(key) + 1:] for p in r["notes"].split(";") if p.startswith(key + "=")), "")


def value_text(r, panel):
    if panel in DEC_ONE:
        return f"{float(r['value']):.1f}"
    return r["value"]


def model_cell(F, cell, model, slots, panel, bolds):
    parts, tags = [], []
    for s in slots:
        r = cell.get((model, s))
        if sm.has_value(r):
            t = F.num(value_text(r, panel))
            if bolds.get(s) is not None and float(r["value"]) >= bolds[s] - 1e-9:
                t = F.bold(t)
            if model == "clay" and panel == "A" and note(r, "se"):
                t += F.pm() + f"{float(note(r, 'se')):.1f}"
            t += F.marks(note(r, "marks"))
            parts.append(t)
            if model != "clay" and note(r, "tag"):
                tags.append(note(r, "tag") + (f", {note(r, 'tie')}" if note(r, "tie") else ""))
        elif r is not None and r["source"] == "pending":
            parts.append("?")
        else:
            parts.append("n/a" + F.marks(note(r, "marks") if r is not None else ""))
    if all(p.startswith("n/a") for p in parts):
        return parts[0]
    txt = F.sep.join(parts)
    if tags:
        txt += f" ({tags[0] if len(set(tags)) == 1 else '/'.join(tags)})"
    return txt


def render(F, path=CSV):
    order, cells = sm.load(path)
    agg = sm.summarize(path)
    lines = F.head()
    panels = sorted({k[0] for k in order})
    for p in panels:
        lines.append(F.panel(PANEL_TITLE[p]))
        for key in [k for k in order if k[0] == p]:
            cell = cells[key]
            first = next(iter(cell.values()))
            slots = ["first", "second"] if any(s != "single" for (_, s) in cell) else ["single"]
            bolds = {}
            for s in slots:
                vals = [float(x["value"]) for m in sm.MODELS if sm.has_value(x := cell.get((m, s)))]
                bolds[s] = max(vals) if vals else None
            lm = next((note(r, "labelmarks") for r in cell.values() if note(r, "labelmarks")), "")
            label = F.code(key[1]) + F.marks(lm)
            pb = sm.pct_best(cell)
            clay_rows = [cell[("clay", s)] for s in slots if ("clay", s) in cell]
            rank = next((note(r, "rank") for r in clay_rows if note(r, "rank")), "")
            cells_ = [label, F.text(first["sensor"]), model_cell(F, cell, "clay", slots, p, bolds),
                      "n/a" if pb is None else F.pct(f"{pb:.0f}%"),
                      model_cell(F, cell, "other_eo", slots, p, bolds), model_cell(F, cell, "general", slots, p, bolds)]
            lines.append(F.row(cells_))
    d = agg["all_panels"]  # one summary row; the per-panel medians stay in summarize.py for the text
    n, k = d["rows"], d["rows_with_clay_value"]
    med = "?" if d["median_pct_best"] is None else f"{d['median_pct_best']:.0f}%"
    fam = agg["families"]  # the Clay column is one checkpoint; each comparison column is a composite of models
    lines.append(F.row([F.bold("Median, all panels"), f"{n} rows" + ("" if k == n else f" ({k} with a Clay value)"),
                        "1 checkpoint", F.bold(F.pct(med)) if med != "?" else "?",
                        f"{len(fam['other_eo'])} model families", f"{len(fam['general'])} model families"]))
    return lines


def tex_block(lines):
    spec = "@{\\extracolsep{\\fill}}ll r r l l@{}"
    out = ["\\begin{tabular*}{0.99\\textwidth}{" + spec + "}"]
    i = 0
    # insert \midrule before every panel header after the first and before the summary rows of each panel
    body = lines[:3]
    rest = lines[3:]
    for j, l in enumerate(rest):
        if l.startswith("\\multicolumn") and "{Panel" in l and j > 0:
            body.append("\\midrule")
        body.append(l)
    # the summary row gets a rule above it
    final = []
    for l in body:
        if l.startswith("\\textbf{Median"):
            final.append("\\midrule")
        final.append(l)
    return out + final + ["\\bottomrule", "\\end{tabular*}"]


def md_block(lines):
    return lines


def used_tags(path=CSV):
    order, cells = sm.load(path)
    tags = set()
    for c in cells.values():
        for r in c.values():
            for part in r["notes"].split(";"):
                if part.startswith(("tag=", "tie=")):
                    tags.add(part[4:])
    return tags


def tag_defined(tag, foot):
    """A tag is defined if the "Model tags" note names it, names its family pattern, or it is a leaderboard model name."""
    import from_leaderboard as fl
    if tag in foot:
        return True
    if re.fullmatch(r"Olmo-v[\d.]+-[A-Z]", tag):
        return "Olmo-v<V>-<S>" in foot
    if re.fullmatch(r"TM(-[A-Z])?(-RGB)?", tag):
        return "TM-<S>" in foot and "-RGB" in foot
    if tag in fl.TAGS.values() and tag in fl.TAGS:  # tag identical to a leaderboard model name
        return "leaderboard model names" in foot
    return False


def splice_md(text, block):
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("| Benchmark"))
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    return "\n".join(lines[:start] + block + lines[end:]), "\n".join(lines[start:end])


def splice_tex(text, block):
    a = text.index("\\begin{tabular*}{0.99\\textwidth}")
    b = text.index("\\end{tabular*}", a) + len("\\end{tabular*}")
    return text[:a] + "\n".join(block) + text[b:], text[a:b]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--print", choices=["md", "tex"])
    a = ap.parse_args()
    md = md_block(render(Md()))
    tex = tex_block(render(Tex()))
    if a.print:
        print("\n".join(md if a.print == "md" else tex))
        return 0
    d_text, t_text = DRAFT.read_text(), TEX.read_text()
    new_d, old_d = splice_md(d_text, md)
    new_t, old_t = splice_tex(t_text, tex)
    if a.write:
        DRAFT.write_text(new_d)
        TEX.write_text(new_t)
        print("table body written to DRAFT.md and latex/sections/02_model.tex")
        return 0
    bad = []
    if old_d != "\n".join(md):
        bad.append("DRAFT.md table body differs from the CSV rendering")
    if old_t != "\n".join(tex):
        bad.append("latex/sections/02_model.tex table body differs from the CSV rendering")
    for name, txt in (("DRAFT.md", d_text), ("02_model.tex", t_text)):
        foot = txt[txt.index("Model tags"):] if "Model tags" in txt else ""
        missing = [t for t in sorted(used_tags()) if not tag_defined(t, foot)]
        if missing:
            bad.append(f"{name}: tags not defined in the footnote 'Model tags' note: {missing}")
    for b in bad:
        print("FAIL", b)
    print("table body matches the CSV; all tags defined" if not bad else "")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
