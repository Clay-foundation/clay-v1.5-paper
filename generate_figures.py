#!/usr/bin/env python3
"""Publication-grade LaTeX/TikZ architecture figure generator for the Clay v1.5 paper (Figure 1, redesign).

The figure is drawn at 1:1 print scale (17.8 cm wide; 6.2 to 7 pt text) in three numbered panels.
It is included at 0.95 of the text width, so printed text size equals source size within about 1%.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"

TIKZ_SOURCE = r"""\documentclass[tikz,border=1pt]{standalone}
\usepackage{fontspec}
\defaultfontfeatures{Ligatures={TeX,NoCommon}}
\setmainfont{lmroman10-regular.otf}[BoldFont=lmroman10-bold.otf,ItalicFont=lmroman10-italic.otf,BoldItalicFont=lmroman10-bolditalic.otf]
\setsansfont{lmsans10-regular.otf}[BoldFont=lmsans10-bold.otf,ItalicFont=lmsans10-oblique.otf]
\usepackage{amsmath,amssymb,graphicx}
\DeclareMathSizes{7}{7}{6.5}{6.5}
\DeclareMathSizes{6.8}{6.8}{6.3}{6.3}
\DeclareMathSizes{6.2}{6.2}{6.2}{6.2}
\usetikzlibrary{arrows.meta,calc}
\definecolor{cB}{HTML}{0072B2}
\definecolor{cO}{HTML}{E69F00}
\definecolor{cP}{HTML}{CC79A7}
\definecolor{ink}{HTML}{1F2933}
\newcommand{\fs}{\sffamily\fontsize{6.8}{7.9}\selectfont}
\newcommand{\fsm}{\sffamily\fontsize{6.2}{7.2}\selectfont}
\newcommand{\fsh}{\sffamily\bfseries\fontsize{7}{8.2}\selectfont}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,font=\fs,text=ink,
  >={Stealth[length=4pt,width=3.4pt]},
  flow/.style={->,line width=0.7pt,draw=ink},
  panel/.style={fill=black!5,draw=none,rounded corners=3pt},
  hdr/.style={font=\fsh,anchor=south},
  mod/.style={draw=cB,fill=cB!14,line width=0.8pt,rounded corners=1.5pt,align=center,inner sep=2pt},
  trf/.style={draw=cO,fill=cO!22,line width=0.8pt,rounded corners=1.5pt,align=center,inner sep=2pt},
  losA/.style={draw=cP,fill=cP!55,line width=1.2pt,rounded corners=1.5pt,align=center,inner sep=2pt},
  losB/.style={draw=cP,fill=cP!8,line width=0.8pt,dashed,rounded corners=1.5pt,align=center,inner sep=2pt},
  gry/.style={draw=black!55,fill=white,line width=0.8pt,rounded corners=1.5pt,align=center,inner sep=2pt},
  op/.style={circle,draw=black!60,fill=white,line width=0.8pt,inner sep=0pt,minimum size=0.5cm},
  img/.style={inner sep=0.5pt,draw=ink,line width=0.5pt}
]
\fill[panel] (0,-0.1) rectangle (5.9,8.15);
\fill[panel] (6.1,-0.1) rectangle (12.2,8.15);
\fill[panel] (12.4,-0.1) rectangle (17.8,8.15);
\node[hdr] at (2.95,7.75) {1\;\;Inputs and patch embedding};
\node[hdr] at (9.15,7.75) {2\;\;Encoder and teacher};
\node[hdr] at (15.1,7.75) {3\;\;Decoder and target};

% ---------------- panel 1
\node[font=\fsm,anchor=south west,align=left] at (0.1,7.0) {\textbf{\fs Wavelength hypernetwork}\\input: band wavelengths in µm (S1: 3.5, 4.0)};
\node[mod,minimum width=1.25cm,minimum height=0.95cm,font=\fsm] (sin) at (0.725,6.4) {Sinusoidal\\encoding\\128-D};
\node[mod,minimum width=1.9cm,minimum height=0.95cm,font=\fsm] (tr) at (2.6,6.4) {Transformer\\1 layer, 4 heads\\128 queries};
\node[mod,minimum width=1.3cm,minimum height=0.95cm,font=\fsm] (ker) at (4.55,6.4) {Kernel\\{\fs $W(\lambda)$}\\8×8};
\draw[flow] (sin) -- (tr);
\draw[flow] (tr) -- (ker);

\node[img] (chip) at (0.8,4.2) {\includegraphics[width=1.2cm,height=1.2cm]{asset_chip_in.png}};
\node[anchor=south,align=center,font=\fsm] at (0.85,4.85) {\textbf{\fs Input chip}\\C = 2 to 21 bands\\GSD 0.1 to 500 m};
\node[op] (conv) at (4.55,4.2) {$\ast$};
\node[op] (add) at (5.35,4.2) {$+$};
\draw[flow] (chip) -- (conv) node[midway,above=1pt,font=\fsm] {8×8 patches};
\draw[flow] (ker) -- (conv);
\draw[flow] (conv) -- (add);
\node[gry,minimum width=3.5cm,minimum height=1.7cm,font=\fsm] (pos) at (4.05,1.95) {\textbf{\fs Position, time, location}\\2D position scaled by GSD\\week / hour\\lon / lat\\metadata dropped 20\% of the time};
\draw[flow] (pos.north -| add) -- (add);

% ---------------- panel 2
\node[img] (sfull) at (6.7,4.2) {\includegraphics[width=0.27cm,height=2.0cm]{asset_strip_full.png}};
\node[anchor=north,font=\fsm] at (6.7,3.15) {Tokens};
\node[img] (svis) at (7.7,4.2) {\includegraphics[width=0.27cm,height=1.5cm]{asset_strip_vis.png}};
\node[anchor=north,align=center,font=\fsm] at (7.55,3.4) {25\% kept\\+ CLS};
\draw[flow] (add.east) -- (sfull.west |- add.east);
\draw[flow] (sfull) -- (svis);

\node[trf,minimum width=2.0cm,minimum height=2.5cm] (enc) at (9.2,4.2) {\textbf{Encoder}\\ViT-L/8, 24 blocks\\16 heads, $D$\,=\,1024\\{\fsm 311.4M parameters}};
\draw[flow] (svis) -- (enc);

\node[draw=cO,line width=0.8pt,fill=cO!12,minimum width=0.9cm,minimum height=1.35cm,inner sep=0pt,font=\fsm] (zlat) at (11.2,4.475) {patch};
\node[draw=cO,line width=0.8pt,fill=cO!55,minimum width=0.9cm,minimum height=0.55cm,inner sep=0pt,font=\fsm] (zcls) at (11.2,3.525) {CLS};
\draw[flow] (enc) -- (enc -| zlat.west);
\node[anchor=south east,align=right,font=\fsm] at (12.05,5.6) {\textbf{Patch tokens:} dense readout;\\central 2×2 for field-scale labels};
\draw[line width=0.5pt,draw=black!60] (11.2,5.6) -- (11.2,5.15);
\node[anchor=north east,font=\fsm] (clslbl) at (11.0,2.75) {\textbf{CLS:} scene-level readout};
\draw[line width=0.5pt,draw=black!60] (11.0,2.62) -- (10.95,3.25);

\node[gry,minimum width=3.5cm,minimum height=1.7cm,font=\fsm] (teach) at (8.05,0.95) {\textbf{\fs Frozen teacher}\\Stage 1: SAM ViT-B/16,\\7 nested Matryoshka heads on CLS\\Stage 2: DINOv2-L/14 with registers,\\dense 1024×1024 projection};
\node[losB,minimum width=1.75cm,minimum height=1.2cm,font=\fsm] (lrep) at (11.2,0.95) {\textbf{Cosine loss}\\CLS vs. teacher\\weight 0.1};
\draw[flow] (teach) -- (lrep);
\draw[flow] (zcls.south) -- (lrep.north);
\draw[flow] (chip.south) -- (0.8,0.95) -- node[pos=0.5,below=1pt,font=\fsm] {pseudo-RGB} (teach.west);

% ---------------- panel 3
\node[trf,minimum width=1.8cm,minimum height=2.2cm] (dec) at (13.45,4.2) {\textbf{Decoder}\\4 blocks, 4 heads\\$D$\,=\,512\\{\fsm 15.9M parameters}};
\draw[flow] (zlat.east |- dec.west) -- (dec.west);
\node[mod,minimum width=1.35cm,minimum height=1.1cm,font=\fsm] (ph) at (15.575,4.2) {\textbf{Pixel head}\\wavelength-\\conditioned};
\node[anchor=south,align=center,font=\fsm] at (15.575,4.85) {mirrors patch\\embedding};
\draw[flow] (dec) -- (ph);
\node[img] (pred) at (17.2,4.2) {\includegraphics[width=1.1cm,height=1.1cm]{asset_chip_rec.png}};
\node[anchor=south,font=\fsm] at (17.2,4.8) {Prediction};
\draw[flow] (ph) -- (pred);
\node[img] (targ) at (17.2,1.9) {\includegraphics[width=1.1cm,height=1.1cm]{asset_chip_in.png}};
\node[anchor=north,font=\fsm] at (17.2,1.3) {Target};
\node[losA,minimum width=2.4cm,minimum height=1.1cm,font=\fsm] (lrec) at (14.85,1.9) {\textbf{L1 loss}\\on 75\% masked\\patches, weight 0.9};
\draw[flow] (targ.west) -- (lrec.east);
\draw[flow] (pred.south) -- (17.2,3.3) -- (14.85,3.3) -- (lrec.north);
\node[font=\fsm,anchor=center] at (15.1,0.4) {\itshape For embeddings: no masking, encoder only.};
\end{tikzpicture}
\end{document}
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build Figure 1 (TikZ via tectonic).")
    ap.add_argument("--out-dir", default=str(FIG_DIR), help="where fig1_architecture_lineage.{pdf,png} are written")
    ap.add_argument("--assets-dir", default=str(FIG_DIR), help="directory holding asset_*.png")
    args = ap.parse_args()
    out_dir, assets_dir = Path(args.out_dir), Path(args.assets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("fig1_architecture_lineage.*"):
        stale.unlink()

    tectonic = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    pdftoppm = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for asset in assets_dir.glob("asset_*.png"):
            shutil.copy2(asset, tmp / asset.name)

        tex_path = tmp / "fig1_architecture_lineage.tex"
        tex_path.write_text(TIKZ_SOURCE)
        subprocess.run([tectonic, str(tex_path)], cwd=str(tmp), check=True)

        pdf_out = out_dir / "fig1_architecture_lineage.pdf"
        shutil.copy2(tmp / "fig1_architecture_lineage.pdf", pdf_out)

        png_stem = out_dir / "fig1_architecture_lineage"
        subprocess.run(
            [pdftoppm, "-png", "-r", "300", "-singlefile", str(pdf_out), str(png_stem)],
            check=True,
        )

    print(f"Generated LaTeX/TikZ Figure 1 (fig1_architecture_lineage.{{pdf,png}}) in {out_dir}")


if __name__ == "__main__":
    main()
