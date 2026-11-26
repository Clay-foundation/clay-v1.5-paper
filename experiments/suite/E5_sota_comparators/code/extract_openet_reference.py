"""Issue #13: extract the published OpenET numbers (and their exact table / section references) from the primary papers.

Inputs are text versions of the two papers (pdftotext of arXiv 2607.03949v2 = TESSERA v2, and arXiv 2507.22291 = AlphaEarth Foundations).
Every value in the output JSON is matched by a regular expression against those texts and stored with the matching line, so the
reference can be audited by search.  Nothing is typed in.

  pdftotext -layout 2607.03949v2.pdf tessera_v2.txt ; pdftotext -layout 2507.22291.pdf alphaearth.txt
  python extract_openet_reference.py --tessera_v2 tessera_v2.txt --alphaearth alphaearth.txt --out openet_published_reference.json
"""
import argparse, datetime, hashlib, json, re


def lines(p): return open(p, encoding="utf-8", errors="replace").read().split("\n")  # split on \n only, so line numbers match grep -n


def find(L, pat, start=0, flags=re.I):
    for i in range(start, len(L)):
        m = re.search(pat, L[i], flags)
        if m: return i, m
    return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tessera_v2", required=True); ap.add_argument("--alphaearth", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); T, A = lines(a.tessera_v2), lines(a.alphaearth)
    out = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "script": "extract_openet_reference.py",
           "inputs_sha256": {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in [("tessera_v2", a.tessera_v2), ("alphaearth", a.alphaearth)]}, "items": {}}
    def put(key, paper, i, value, note=""):
        out["items"][key] = {"paper": paper, "line_no": None if i is None else i + 1, "line": None if i is None else T[i].strip() if paper.startswith("TESSERA") else A[i].strip(),
                             "value": value, "note": note}
    # TESSERA v2: sentence with annual score and AlphaEarth, monthly score, Table 2 caption and rows
    i0, _ = find(T, r"OpenET evapotranspiration ensemble, one of the 15")
    blk = " ".join(x.strip() for x in T[i0:i0 + 12]) if i0 is not None else ""
    m = re.search(r"annual window, the TESSERA v2-2B-L model scores (\d\.\d+) on this task, behind\s+A\s*LPHA\s*E\s*ARTH at (\d\.\d+)", blk, re.I)
    mm = re.search(r"monthly window, with no change to the model or its weights, raises the score to (\d\.\d+) \(Table (\d+)\)", blk, re.I)
    put("tessera_v2_openet_annual_window", "TESSERA v2 arXiv 2607.03949v2, Appendix E 'Inference-time temporal windows'", i0, float(m.group(1)) if m else None,
        "TESSERA v2-2B-L, annual inference window")
    put("tessera_v2_openet_alphaearth_row", "TESSERA v2 arXiv 2607.03949v2, Appendix E", i0, float(m.group(2)) if m else None, "AlphaEarth shown for reference (labelled annual in that table)")
    put("tessera_v2_openet_monthly_window", "TESSERA v2 arXiv 2607.03949v2, Appendix E", i0, float(mm.group(1)) if mm else None, f"cited as Table {mm.group(2) if mm else '?'} in the text")
    for key, pat in [("table2_row_alphaearth", r"A\s*LPHA\s*E\s*ARTH \(annual\)\s+(\d\.\d+)"), ("table2_row_tessera_annual", r"TESSERA v2-2B-L \(annual window\)\s+(\d\.\d+)"),
                     ("table2_row_tessera_monthly", r"TESSERA v2-2B-L \(monthly window\)\s+(\d\.\d+)")]:
        i, m = find(T, pat); put(key, "TESSERA v2 Table 2 (caption: 'Effect of the inference window on the OpenET ensemble')", i, float(m.group(1)) if m else None)
    i, _ = find(T, r"^Table 2: Effect of the inference window"); put("table2_caption", "TESSERA v2", i, None)
    i, m = find(T, r"regression reports max\(0, R2 \)"); put("regression_metric", "TESSERA v2 Section 'Task suite' (Appendix on scaling sweep)", i, "max(0, R2)", "regression scores are floored at 0, so 0.51/0.58/0.69 are R2 values")
    i, m = find(T, r"^\s*OpenET ET\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)")
    put("table16_openet_row_NOT_the_cited_values", "TESSERA v2 Table 16 (per-dataset Matryoshka prefix scores of a preliminary distillation student, maximum shot)", i,
        [float(x) for x in m.groups()] if m else None, "prefix 16/32/64/128; this is a different model and table than the 0.51 / 0.69 quoted in Table 2 / Appendix E")
    # AlphaEarth: OpenET R2 and task definition
    i, m = find(A, r"achiev-?\s*$|R\s*2\s*=\s*0\.58")
    ia, ma = find(A, r"ing 𝑅2 = (0\.\d+) ± (0\.\d+) \(Figure 3\)")
    put("alphaearth_openet_r2_max_trial", "AlphaEarth Foundations arXiv 2507.22291, Section 'Evaluation' text and Figure 3 (max-trial setting)", ia, float(ma.group(1)) if ma else None,
        f"+- {ma.group(2) if ma else '?'}; 'In the max-trial setting ... AEF ... R2 = 0.58'")
    i, m = find(A, r"OpenET\s+ensemble\s+Evapotranspiration\s+regression\s+Western US\s+Monthly\s+(\d+)\s+([\d,]+)")
    put("alphaearth_table1_openet_row", "AlphaEarth Foundations Table 1", i, {"valid_period": "Monthly", "res_m": int(m.group(1)) if m else None, "n_points": m.group(2) if m else None},
        "label valid period is a month; AEF embeddings are static summaries over a conditional valid period (Figure 1B caption)")
    i, m = find(A, r"valid period.*that need not fully intersect"); put("alphaearth_valid_period_conditioning", "AlphaEarth Foundations Figure 1B caption", i, True)
    i, m = find(A, r"calendar month"); put("alphaearth_openet_label_timing", "AlphaEarth Foundations supplement S3.3.9 OpenET", i, "estimates wanted at the conclusion of a given calendar month")
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in out["items"].items(): print(k, "|", v["value"], "| line", v["line_no"])


if __name__ == "__main__":
    main()
