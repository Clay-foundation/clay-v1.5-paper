#!/usr/bin/env python3
"""Deterministic build and quality-gate script for the Clay v1.5 paper.

Maintains and verifies the dual Markdown + LaTeX workflow:
  - `DRAFT.md` (canonical human-editable Markdown) <-> `latex/main.tex` + `latex/sections/*.tex`
  - Outputs: `Clay-v1.5-arxiv.pdf`, `Clay-v1.5-arxiv-source.tar.gz`, `experiments/results/build_manifest.json`

Every gate either passes, raises, or (only with an explicit --arxiv-check skip or --allow-label-drift) is recorded as skipped
in the manifest. A missing tool is an error, never a silent skip.

Steps
  1. (optional, --regenerate-figures) regenerate figures; otherwise the committed figure is used as is.
  2. Placeholder check: unresolved TODO/placeholder macros in DRAFT.md and latex/.
  3. DRAFT.md <-> latex/ parity: citation keys, table and figure counts, Table 2 row labels.
  4. Compile `latex/main.tex` with tectonic (XeTeX engine) with SOURCE_DATE_EPOCH/FORCE_SOURCE_DATE set.
  5. Log gates: no undefined citation/reference, no LaTeX error, no overfull hbox > 2 pt.
  6. PDF gates (pdftotext/pdfinfo required): section headings present and in order, no ligature code points,
     no unresolved `??`, main-text length measured at the "References" heading, bibliography entry count in
     the compiled .bbl equals the number of cited keys.
  7. Package a deterministic arXiv tarball (sorted entries, fixed mtime, uid/gid 0, gzip mtime 0).
  8. Tarball checks: unpack to a clean directory and (a) compile it again with tectonic, (b) compile it with
     pdflatex (the engine arXiv uses) via a local pdflatex or the docker image named in --texlive-image.

Requirements: tectonic, pdftotext and pdfinfo (poppler) on PATH (or TECTONIC=/path); for step 8(b) either
pdflatex or docker. Usage: `python build.py --help`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
LATEX_DIR = HERE / "latex"
FIG_DIR = HERE / "figures"
SECTIONS = [
    "01_introduction.tex",
    "02_model.tex",
    "03_comparative.tex",
    "04_discussion.tex",
    "05_conclusion.tex",
]
FIGURE_FILES = ["fig1_architecture_lineage.pdf"]
DEFAULT_TEXLIVE_IMAGE = "texlive/texlive:latest"
FALLBACK_EPOCH = 1735689600  # 2025-01-01T00:00:00Z, used only when neither SOURCE_DATE_EPOCH nor git is available

# Unresolved-placeholder markers (neutral: macros and tags an author leaves while drafting).
PLACEHOLDER_PATTERNS = [
    r"\\authorcheck",
    r"\\torun",
    r"\[TO RUN",
    r"PLACEHOLDER_FOR_EXACT",
    r"\bTODO\b",
    r"\bFIXME\b",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(tool: str, env: str | None = None) -> str:
    found = (os.environ.get(env) if env else None) or shutil.which(tool)
    if not found:
        raise RuntimeError(f"required tool `{tool}` not found on PATH" + (f" (or set {env})" if env else "") + "; install it, the gate cannot be skipped")
    return found


def source_date_epoch() -> int:
    if os.environ.get("SOURCE_DATE_EPOCH"):
        return int(os.environ["SOURCE_DATE_EPOCH"])
    try:
        return int(subprocess.check_output(["git", "-C", str(HERE), "log", "-1", "--format=%ct"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return FALLBACK_EPOCH


# --------------------------------------------------------------------------- placeholder + parity gates
def run_placeholder_check() -> int:
    files = [HERE / "DRAFT.md", LATEX_DIR / "main.tex", *(LATEX_DIR / "sections" / s for s in SECTIONS)]
    hits = []
    for p in files:
        txt = p.read_text(errors="replace")
        for pat in PLACEHOLDER_PATTERNS:
            for m in re.finditer(pat, txt):
                hits.append(f"{p.relative_to(HERE)}: unresolved placeholder `{m.group(0)}`")
    if hits:
        raise RuntimeError("Placeholder check FAILED:\n  - " + "\n  - ".join(hits))
    return 0


def _norm_label(s: str) -> str:
    s = re.sub(r"~?\\cite\{[^}]*\}", "", s)
    s = s.replace("?", "QMARK").replace("\u00b1", "\\pm")  # a pending `?` cell must match a pending `?` cell, not a blank
    s = s.replace("\\cdot", "").replace("·", "")
    s = s.replace("\\star", "").replace("\\ge", "").replace("\\le", "")
    s = re.sub(r"\[`[^\]]*`\]", "", s)
    s = re.sub(r"\\(texttt|textbf|text|emph|textit)\b", "", s)
    s = s.replace("\\,", "").replace("\\ ", "").replace("\\slsh", "/").replace("\\%", "%")
    s = re.sub(r"\$\^[^$]*\$", "", s)  # footnote-style superscripts, e.g. $^{\ast\ddagger}$
    return re.sub(r"[^0-9a-zA-Z%/.]", "", s).lower()


def table2_row_labels(draft_text: str, tex_text: str) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    md_rows: list[tuple[str, str]] = []
    lines = draft_text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("| Benchmark |"))
    for l in lines[start + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[1]:
            md_rows.append(tuple(cells))
    tex_rows: list[tuple[str, str]] = []
    body = tex_text[tex_text.index("Benchmark & Sensor and metric"):]
    body = body[body.index("\\midrule"): body.index("\\bottomrule")]
    for row in re.split(r"\\\\\s*\n", body):
        row = re.sub(r"\\(midrule|toprule)\s*", "", row).strip()
        if not row or row.startswith("\\multicolumn"):
            continue
        cells = [c.strip() for c in row.split("&")]
        if len(cells) >= 2:
            tex_rows.append(tuple(cells))
    return md_rows, tex_rows


def verify_draft_latex_consistency(allow_label_drift: bool) -> dict[str, object]:
    draft_text = (HERE / "DRAFT.md").read_text()
    full_tex = "\n".join([(LATEX_DIR / "main.tex").read_text()] + [(LATEX_DIR / "sections" / s).read_text() for s in SECTIONS])
    bib_text = (LATEX_DIR / "references.bib").read_text()

    bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib_text))
    tex_cites: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", full_tex):
        for k in m.group(1).split(","):
            tex_cites.add(k.strip())

    md_cites: set[str] = set()
    for m in re.finditer(r"\[((?:`[A-Za-z0-9_:\-]+`(?:,\s*)?)+)\]", draft_text):
        for k in re.findall(r"`([A-Za-z0-9_:\-]+)`", m.group(1)):
            md_cites.add(k)

    missing_in_bib = (tex_cites | md_cites) - bib_keys
    if missing_in_bib:
        raise RuntimeError(f"Cited keys missing from references.bib: {sorted(missing_in_bib)}")
    if tex_cites ^ md_cites:
        raise RuntimeError(
            f"Citation key mismatch between DRAFT.md and latex/: "
            f"only in tex={sorted(tex_cites - md_cites)}, only in md={sorted(md_cites - tex_cites)}"
        )

    md_tables = re.findall(r"^\*\*Table (\d+):", draft_text, flags=re.MULTILINE)
    tex_tables = re.findall(r"\\label\{(tab:[^}]+)\}", full_tex)
    if not md_tables or len(md_tables) != len(tex_tables):
        raise RuntimeError(f"Table count mismatch: DRAFT.md has {len(md_tables)}, latex/ has {len(tex_tables)}")
    md_figs = re.findall(r"^!\[Figure (\d+):", draft_text, flags=re.MULTILINE)
    tex_figs = re.findall(r"\\label\{(fig:[^}]+)\}", full_tex)
    if not md_figs or len(md_figs) != len(tex_figs):
        raise RuntimeError(f"Figure count mismatch: DRAFT.md has {len(md_figs)}, latex/ has {len(tex_figs)}")

    md_rows, tex_rows = table2_row_labels(draft_text, (LATEX_DIR / "sections" / "02_model.tex").read_text())
    mismatches = []
    if len(md_rows) != len(tex_rows):
        mismatches.append(f"row count: DRAFT.md {len(md_rows)} vs latex {len(tex_rows)}")
    for i, (a, b) in enumerate(zip(md_rows, tex_rows)):
        if _norm_label(a[0]) != _norm_label(b[0]) or _norm_label(a[1]) != _norm_label(b[1]):
            mismatches.append(f"row {i + 1}: DRAFT.md {a} vs latex {b}")
    # Every cell, not only the two label columns: a number changed in one format alone fails here.
    cell_mismatches = []
    for i, (a, b) in enumerate(zip(md_rows, tex_rows)):
        if len(a) != len(b):
            cell_mismatches.append(f"row {i + 1} ({a[0]}): {len(a)} cells in DRAFT.md vs {len(b)} in latex")
            continue
        for j, (x, y) in enumerate(zip(a, b)):
            if _norm_label(x.replace("\u2212", "-")) != _norm_label(y.replace("\u2212", "-")):
                cell_mismatches.append(f"row {i + 1} ({a[0]}) column {j + 1}: DRAFT.md {x!r} vs latex {y!r}")
    if cell_mismatches:
        raise RuntimeError("Table 2 cells differ between DRAFT.md and latex:\n  - " + "\n  - ".join(cell_mismatches[:20]))
    if mismatches and not allow_label_drift:
        raise RuntimeError("Table 2 row-label drift between DRAFT.md and latex (use --allow-label-drift to record only):\n  - " + "\n  - ".join(mismatches))

    return {
        "bib_entries_total": len(bib_keys),
        "cited_entries_count": len(tex_cites),
        "tables_verified": len(tex_tables),
        "figures_verified": len(tex_figs),
        "table2_rows_compared": min(len(md_rows), len(tex_rows)),
        "table2_cells_compared": sum(len(r) for r in md_rows),
        "table2_row_label_mismatches": len(mismatches),
        "table2_row_label_mismatch_detail": mismatches,
        "draft_latex_parity": not mismatches,
    }


# --------------------------------------------------------------------------- PDF measurements
def pdf_pages(pdf: Path) -> int:
    out = subprocess.check_output([need("pdfinfo"), str(pdf)], text=True)
    return int(re.search(r"^Pages:\s+(\d+)", out, flags=re.M).group(1))


def main_text_pages(pdf: Path) -> dict[str, object]:
    """Main-text length in pages, measured at the line that reads exactly "References".

    Two-column layout: if the heading is in the left column the text before it fills that fraction of the left
    column, if it is in the right column the whole left column plus that fraction of the right column. Pages
    entirely before the heading's page count as 1.0. Frame heights use the text area between the page margins.
    """
    html = subprocess.check_output([need("pdftotext"), "-bbox-layout", str(pdf), "-"], text=True)
    pages = re.split(r"<page ", html)[1:]
    for idx, pg in enumerate(pages):
        w = float(re.search(r'width="([0-9.]+)"', pg).group(1))
        h = float(re.search(r'height="([0-9.]+)"', pg).group(1))
        for m in re.finditer(r'<line xMin="([0-9.]+)" yMin="([0-9.]+)" xMax="[0-9.]+" yMax="([0-9.]+)">(.*?)</line>', pg, flags=re.S):
            words = re.findall(r">([^<>]+)</word>", m.group(4))
            if " ".join(words).strip() == "References":
                x, y = float(m.group(1)), float(m.group(2))
                top, bottom = 0.60 * 72, h - 0.60 * 72  # geometry in latex/main.tex: top/bottom margin 0.60 in
                frac = min(max((y - top) / (bottom - top), 0.0), 1.0)
                col = 0.5 * frac if x < w / 2 else 0.5 + 0.5 * frac
                return {"references_page": idx + 1, "references_column": "left" if x < w / 2 else "right",
                        "main_text_pages": round(idx + col, 2), "total_pages": len(pages)}
    raise RuntimeError(f"no line reading exactly 'References' found in {pdf.name}; cannot measure main-text length")


def _title_key(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"\$[^$]*\$", " ", s)
    return re.sub(r"[^a-z0-9]", "", s.lower())


def check_pdf(pdf: Path, max_main_pages: float) -> dict[str, object]:
    text = subprocess.check_output([need("pdftotext"), "-raw", str(pdf), "-"], text=True)
    ligs = re.findall(r"[\uFB00-\uFB04]", text)
    if ligs:
        raise RuntimeError(f"Unicode ligature codepoints found in {pdf.name}: {len(ligs)}")
    if re.search(r"\[\?\]|\?\?", text):
        raise RuntimeError(f"unresolved `??`/`[?]` reference or citation in {pdf.name}")
    # section headings, taken from the LaTeX sources, must appear in order
    titles = []
    for s in SECTIONS:
        titles += re.findall(r"\\(?:sub)?section\*?\{((?:[^{}]|\{[^{}]*\})*)\}", (LATEX_DIR / "sections" / s).read_text())
    flat = re.sub(r"[^a-z0-9]", "", text.lower())
    pos = 0
    for t in titles:
        k = _title_key(t)[:28]
        j = flat.find(k, pos)
        if j < 0:
            raise RuntimeError(f"section heading not found in PDF text, or out of order: {t!r}")
        pos = j + 1
    mt = main_text_pages(pdf)
    if mt["main_text_pages"] > max_main_pages:
        raise RuntimeError(f"main-text budget exceeded in {pdf.name}: {mt['main_text_pages']} > {max_main_pages}")
    return {"unicode_ligatures": len(ligs), "section_headings_checked": len(titles), **mt}


def check_tex_log(log_text: str, label: str) -> dict[str, object]:
    for tok in ("LaTeX Warning: Reference", "LaTeX Warning: Citation", "Package natbib Warning: Citation",
                "LaTeX Error:", "! Undefined control sequence", "Emergency stop", "Fatal error"):
        if tok in log_text:
            raise RuntimeError(f"TeX quality gate failed ({label}): `{tok}` found in log")
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
    severe = [x for x in over if x > 2.0]
    if severe:
        raise RuntimeError(f"Overfull \\hbox > 2pt ({label}): {severe}")
    return {"overfull_hbox_total": len(over), "overfull_hbox_count_gt_2pt": len(severe), "overfull_hbox_max_pt": max(over) if over else 0.0}


# --------------------------------------------------------------------------- compile + package
def compile_main(tectonic: str, outdir: Path, epoch: int, srcdir: Path | None = None) -> tuple[Path, Path, Path]:
    srcdir = srcdir or LATEX_DIR
    env = {**os.environ, "SOURCE_DATE_EPOCH": str(epoch), "FORCE_SOURCE_DATE": "1"}
    subprocess.run([tectonic, "--keep-logs", "--keep-intermediates", "--outdir", str(outdir), str(srcdir / "main.tex")], check=True, env=env, cwd=str(srcdir))
    return outdir / "main.pdf", outdir / "main.log", outdir / "main.bbl"


def build_tarball(out_tar: Path, bbl: Path) -> list[str]:
    entries: list[tuple[Path, str]] = [
        (LATEX_DIR / "main.tex", "main.tex"),
        (bbl, "main.bbl"),
        (LATEX_DIR / "references.bib", "references.bib"),
        *[(LATEX_DIR / "sections" / s, f"sections/{s}") for s in SECTIONS],
        *[(FIG_DIR / f, f"figures/{f}") for f in FIGURE_FILES],
    ]
    entries.sort(key=lambda e: e[1])
    tmp = out_tar.with_suffix(".tmp")
    with open(tmp, "wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0, compresslevel=9) as gz_fh:
            with tarfile.open(mode="w", fileobj=gz_fh, format=tarfile.GNU_FORMAT) as tf:
                for src_path, arcname in entries:
                    data = src_path.read_bytes()
                    ti = tarfile.TarInfo(name=arcname)
                    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
                    tf.addfile(ti, io.BytesIO(data))
    tmp.replace(out_tar)
    return [a for _, a in entries]


def check_tarball(tar: Path, tectonic: str, epoch: int, arxiv_mode: str, texlive_image: str, max_main_pages: float) -> dict[str, object]:
    info: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="arxiv_stage_") as td:
        stage = Path(td) / "src"
        stage.mkdir()
        with tarfile.open(tar) as tf:
            names = tf.getnames()
            if any(n.startswith("/") or ".." in Path(n).parts for n in names):
                raise RuntimeError("unsafe path in tarball")
            tf.extractall(stage, filter="data")
        info["tarball_files"] = len(names)
        # (a) the tarball alone must compile with tectonic
        out_a = Path(td) / "tectonic"
        out_a.mkdir()
        pdf_a, log_a, _ = compile_main(tectonic, out_a, epoch, srcdir=stage)
        info["tarball_tectonic_pages"] = pdf_pages(pdf_a)
        info["tarball_tectonic_log"] = check_tex_log(log_a.read_text(errors="replace"), "tarball/tectonic")
        # (b) arXiv engine: pdflatex
        if arxiv_mode == "skip":
            info["arxiv_pdflatex_check"] = "skipped (explicit --arxiv-check skip)"
            return info
        pdflatex = shutil.which("pdflatex")
        out_b = Path(td) / "pdflatex"
        out_b.mkdir()
        if arxiv_mode in ("auto", "pdflatex") and pdflatex:
            runner = "local pdflatex"
            for _ in range(2):  # second pass resolves references; .bbl is shipped, so no bibtex is run (as on arXiv)
                subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={out_b}", "main.tex"],
                               check=True, cwd=str(stage), stdout=subprocess.DEVNULL)
        elif arxiv_mode in ("auto", "docker"):
            docker = shutil.which("docker")
            if not docker:
                raise RuntimeError("arXiv check needs pdflatex or docker; neither found (use --arxiv-check skip to record it as skipped)")
            # The tarball goes in on stdin and the output directory comes back as a tar on stdout, so no bind mounts
            # are needed and DOCKER_HOST=ssh://<host> works for a remote engine.
            script = ("mkdir -p /w /out && tar xz -C /w && cd /w && "
                      "pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/out main.tex >/dev/null && "
                      "pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/out main.tex >/dev/null; "
                      "rc=$?; tar c -C /out .; exit $rc")
            proc = subprocess.run([docker, "run", "--rm", "-i", texlive_image, "sh", "-c", script],
                                  input=tar.read_bytes(), stdout=subprocess.PIPE)
            with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as ot:
                ot.extractall(out_b, filter="data")
            runner = f"docker {texlive_image} (DOCKER_HOST={os.environ.get('DOCKER_HOST', 'local')})"
            if proc.returncode != 0:
                tail = (out_b / "main.log").read_text(errors="replace")[-1500:] if (out_b / "main.log").exists() else ""
                raise RuntimeError(f"pdflatex failed in {texlive_image} (exit {proc.returncode}); log tail:\n{tail}")
        else:
            raise RuntimeError("pdflatex not found on PATH")
        pdf_b = out_b / "main.pdf"
        info["arxiv_pdflatex_runner"] = runner
        info["arxiv_pdflatex_pages"] = pdf_pages(pdf_b)
        mt = main_text_pages(pdf_b)
        info["arxiv_pdflatex_main_text_pages"] = mt["main_text_pages"]
        log_text = (out_b / "main.log").read_text(errors="replace")
        info["arxiv_pdflatex_overfull_hbox_pt"] = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
        # Failures are recorded in the manifest (which is still written) and make build.py exit non-zero.
        try:
            info["arxiv_pdflatex_log"] = check_tex_log(log_text, "pdflatex")
            if mt["main_text_pages"] > max_main_pages:
                raise RuntimeError(f"pdflatex main-text budget exceeded: {mt['main_text_pages']} > {max_main_pages}")
            info["arxiv_pdflatex_check"] = "passed"
        except RuntimeError as exc:
            info["arxiv_pdflatex_check"] = f"FAILED: {exc}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--regenerate-figures", action="store_true", help="run generate_figures.py first (default: use the committed figure)")
    ap.add_argument("--max-main-pages", type=float, default=7.0, help="gate on measured main-text pages (default 7.0)")
    ap.add_argument("--allow-label-drift", action="store_true", help="record, but do not fail on, Table 2 row-label differences between DRAFT.md and latex")
    ap.add_argument("--arxiv-check", choices=["auto", "pdflatex", "docker", "skip"], default="auto",
                    help="how to compile the tarball with pdflatex: local binary, docker image, or skip (recorded as skipped)")
    ap.add_argument("--texlive-image", default=os.environ.get("TEXLIVE_IMAGE", DEFAULT_TEXLIVE_IMAGE))
    a = ap.parse_args()

    tectonic = need("tectonic", "TECTONIC")
    need("pdftotext")
    need("pdfinfo")
    epoch = source_date_epoch()

    if a.regenerate_figures:
        subprocess.run([sys.executable, "-B", str(HERE / "generate_figures.py")], cwd=str(HERE), check=True,
                       env={**os.environ, "SOURCE_DATE_EPOCH": str(epoch)})
    placeholders = run_placeholder_check()
    parity = verify_draft_latex_consistency(a.allow_label_drift)

    with tempfile.TemporaryDirectory(prefix="clay_build_") as td:
        outdir = Path(td)
        pdf, log, bbl = compile_main(tectonic, outdir, epoch)
        log_info = check_tex_log(log.read_text(errors="replace"), "main")
        bbl_items = len(re.findall(r"\\bibitem", bbl.read_text(errors="replace")))
        if bbl_items != parity["cited_entries_count"]:
            raise RuntimeError(f"compiled bibliography has {bbl_items} entries but {parity['cited_entries_count']} keys are cited")
        pdf_info = check_pdf(pdf, a.max_main_pages)
        out_pdf = HERE / "Clay-v1.5-arxiv.pdf"
        shutil.copy2(pdf, out_pdf)
        out_tar = HERE / "Clay-v1.5-arxiv-source.tar.gz"
        members = build_tarball(out_tar, bbl)
    tar_info = check_tarball(out_tar, tectonic, epoch, a.arxiv_check, a.texlive_image, a.max_main_pages)

    manifest = {
        "pdf": out_pdf.name,
        "pdf_sha256": sha256_file(out_pdf),
        "arxiv_tar": out_tar.name,
        "arxiv_tar_sha256": sha256_file(out_tar),
        "arxiv_tar_members": members,
        "source_date_epoch": epoch,
        "tectonic_version": subprocess.check_output([tectonic, "--version"], text=True).strip(),
        "figures_regenerated": a.regenerate_figures,
        "unresolved_placeholders": placeholders,
        "compiled_bibliography_entries": bbl_items,
        **log_info,
        **pdf_info,
        **parity,
        **tar_info,
    }
    (HERE / "experiments/results/build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if str(manifest.get("arxiv_pdflatex_check", "")).startswith("FAILED"):
        print("BUILD FAILED: " + manifest["arxiv_pdflatex_check"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
