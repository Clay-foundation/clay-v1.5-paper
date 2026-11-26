#!/usr/bin/env python3
"""Recompute every sum in the carbon/compute ledger from the AWS emissions export.

Emissions come from experiments/results/aws_carbon_2024_by_service.csv: the 2024 rows of the AWS
Data Exports table CARBON_EMISSIONS (Customer Carbon Footprint data, carbon model v1.0.0, export
created 2025-10-21, read 2026-11-17), one row per month, service and region. This script checks the
monthly rows in corpus_and_carbon_provenance.json against that export and derives the account-wide
and EC2 us-east-2 totals for August-November 2024. Node-hours come from EC2 usage exports that are
not in the repository; they are summed here, not verified.

The upstream release note (Clay-foundation/model docs/release-notes/specification.md line 128 at
f14e698) gives 0.22 t market-based for October 2024. The export gives 0.022 t, which is used here.

Usage (repo root): python experiments/table1/check_carbon.py [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "suite" / "common"))
from provenance import provenance  # noqa: E402

OCT_MARKET_UPSTREAM_SPEC = 0.22  # specification.md:128 "0.22 MT CO2e market-based" (month of October 2024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "experiments/results/corpus_and_carbon_provenance.json"))
    a = ap.parse_args()
    p = Path(a.json)
    d = json.loads(p.read_text())
    c = d["carbon_and_compute_2024"]

    # The training-only estimate had no derivation and was withdrawn by the authors.
    c.pop("v15_pretraining_alone_estimated_tco2e", None)
    c.pop("author_estimate_v15_pretraining_tco2e", None)
    if "as_stored_totals" not in c:
        c["as_stored_totals"] = {"aug_nov_window_total": c["aug_nov_window_total"],
                                 "full_year_2024_total": c["full_year_2024_total"]}
    c["scope"] = {
        "emissions": ("ACCOUNT-WIDE: all services of the whole AWS account, not only the training instances. "
                      "CCFT is monthly; it cannot isolate the v1.5 pre-training run."),
        "node_hours": "EC2 instance-hours of g6/g5/g6e.48xlarge (8 GPUs each) only, from usage exports",
        "mixed": ("Emissions (all services) and node-hours (3 instance families) do not measure the same thing. "
                  "Aug-Nov also contains tests, S2/NAIP catalog inference and QA (see 'activity')."),
        "region": c["region"],
    }

    months = c["monthly_2024"]
    keys = ("g6_48xlarge_h", "g5_48xlarge_h", "location_based_tco2e", "market_based_tco2e")
    sums = {k: round(sum(m[k] for m in months), 3) for k in keys}
    g6e = c["oct_2024_g6e_48xlarge_hours"]
    node_hours_total = round(sums["g6_48xlarge_h"] + sums["g5_48xlarge_h"] + g6e, 1)

    # Emissions export: account-wide and EC2 us-east-2 totals per month.
    import csv
    win = ("2024-08", "2024-09", "2024-10", "2024-11")
    acct = {m: [0.0, 0.0] for m in win}
    ec2 = {m: [0.0, 0.0] for m in win}
    with open(ROOT / "experiments/results/aws_carbon_2024_by_service.csv") as fh:
        for r in csv.DictReader(fh):
            m = r["usage_period"]
            if m not in acct:
                continue
            lb, mb = float(r["lbm_tco2e"]), float(r["mbm_tco2e"])
            acct[m][0] += lb; acct[m][1] += mb
            if r["product_code"] == "AmazonEC2" and r["region_code"] == "us-east-2":
                ec2[m][0] += lb; ec2[m][1] += mb
    export_rows = [{"month": m, "account_location_tco2e": round(acct[m][0], 4), "account_market_tco2e": round(acct[m][1], 4),
                    "ec2_us_east_2_location_tco2e": round(ec2[m][0], 4), "ec2_us_east_2_market_tco2e": round(ec2[m][1], 4)} for m in win]
    ledger_matches_export = all(
        abs(next(x for x in months if x["month"] == m)["location_based_tco2e"] - round(acct[m][0], 2)) < 0.006
        and abs(next(x for x in months if x["month"] == m)["market_based_tco2e"] - round(acct[m][1], 3)) < 0.0006 for m in win)
    oct_row = next(m for m in months if m["month"] == "2024-10")
    mk_here = oct_row["market_based_tco2e"]
    st = c["as_stored_totals"]
    full_resid = {  # implied rest-of-year (Jan-Jul + Dec), derived from stored full-year minus Aug-Nov
        "g5_48xlarge_h": round(st["full_year_2024_total"]["g5_48xlarge_h"] - sums["g5_48xlarge_h"], 1),
        "location_based_tco2e": round(st["full_year_2024_total"]["location_based_tco2e"] - sums["location_based_tco2e"], 2),
        "market_based_tco2e": round(st["full_year_2024_total"]["market_based_tco2e"] - sums["market_based_tco2e"], 3),
    }
    c["derived"] = {
        "aug_nov_recomputed": sums,
        "aug_nov_matches_stored": {
            "g6_48xlarge_h": sums["g6_48xlarge_h"] == st["aug_nov_window_total"]["g6_48xlarge_h"],
            "g5_48xlarge_h": sums["g5_48xlarge_h"] == st["aug_nov_window_total"]["g5_48xlarge_h"],
            "location_based_tco2e": abs(sums["location_based_tco2e"] - st["aug_nov_window_total"]["location_based_tco2e"]) < 1e-9,
            "market_based_tco2e": abs(sums["market_based_tco2e"] - st["aug_nov_window_total"]["market_based_tco2e"]) < 1e-9,
        },
        "node_hours_total_incl_oct_g6e": node_hours_total,
        "node_hours_note": ("42,031 = g6 33,622.8 + g5 4,438.1 + g6e 3,970.2 (g6e exists only for October; it has "
                            "no monthly row and its share of the Oct emissions is unknown)."),
        "emissions_export": {
            "source": "AWS Data Exports CARBON_EMISSIONS, carbon model v1.0.0, export created 2025-10-21, read 2026-11-17",
            "file": "experiments/results/aws_carbon_2024_by_service.csv",
            "monthly": export_rows,
            "aug_nov_account_wide_location_tco2e": round(sum(acct[m][0] for m in win), 2),
            "aug_nov_account_wide_market_tco2e": round(sum(acct[m][1] for m in win), 3),
            "aug_nov_ec2_us_east_2_location_tco2e": round(sum(ec2[m][0] for m in win), 2),
            "aug_nov_ec2_us_east_2_market_tco2e": round(sum(ec2[m][1] for m in win), 3),
            "ledger_monthly_rows_match_export": ledger_matches_export,
        },
        "october_market_based": {
            "export_tco2e": round(acct["2024-10"][1], 4),
            "ledger_tco2e": mk_here,
            "upstream_specification_md_tco2e": OCT_MARKET_UPSTREAM_SPEC,
            "status": "Resolved against the export: 0.022 t. The upstream release note's 0.22 t is not supported by it.",
        },
        "full_year_minus_aug_nov_implied": full_resid,
        "full_year_note": "Not reproducible from the monthly rows (only Aug-Nov are listed); residual shown for transparency.",
    }
    c["v10_node_hours"] = {
        "may_2024_p5_48xlarge_hours": c["v10_may_2024_p5_48xlarge_hours"],
        "apr_may_2024_p5_48xlarge_hours": c["v10_apr_may_2024_p5_48xlarge_hours"],
        "difference_april_hours": round(c["v10_apr_may_2024_p5_48xlarge_hours"] - c["v10_may_2024_p5_48xlarge_hours"], 1),
        "label": ("887.3 h = May 2024 only; 1,059.8 h = April + May 2024 (the v1.0 window). The manuscript's "
                  "'887 node-hours for v1.0' therefore covers May only."),
    }
    d.setdefault("provenance", {})["check_carbon"] = provenance(__file__, {"json": "<repo json>"}, ROOT)
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps(c["derived"], indent=1)[:2500])


if __name__ == "__main__":
    main()
