#!/usr/bin/env python3
"""Collect every Lesson 3/4/5 number (and the issue-#22 cosine definitions) from the regenerated result files, next to the
pre-audit values, into experiments/results/lessons_345_old_vs_new.json.

Inputs (all under experiments/results/; override the directory with --results):
  v15_clean_metrics.json                         regenerated, documented lat/lon order (what the paper's perturbation runs used)
  v15_clean_metrics_latlon_training.json         optional: same run with the training lat/lon order (sensitivity)
  probe_and_lsvm_results.json                    regenerated probe benchmark
  pre_audit/{v15_clean_metrics,probe_and_lsvm_results}.json   the files as committed before the audit

Run:  python experiments/summarize_lessons_345.py
No numbers are typed in here: every value is read from a JSON by key path, and the key path is stored next to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARAM = ["lsvm", "logistic", "mlp_2layer"]
OTHER = ["random_forest", "hist_gbdt", "knn_cosine", "centroid", "umap2d_knn"]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def g(d, path, default=None):
    for k in path.split("/"):
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return d if d is not None else default


def v15_numbers(d):
    if d is None:
        return None
    B = d["expB_mrl_fossil_audit"]
    ex = B["exchangeability_by_d"]
    vb = B["variance_by_mrl_block"]
    D = d["expD_operational_and_contracts"]
    pc = D["preprocessing_contracts"]
    cliff = D["missing_pixel_cliff_and_rescue"]["22pct_slc_off"]
    C = d["expC_spatial_readout_and_conditioning"]
    q = C["eurosat_2x2_compositional_binding"]
    dense = C["dense_80m_patch_readout"]
    out = {
        "lesson2_expC": {
            "eurosat_2x2_cls_4way_acc_built": {"value": q["built_quadrant"]["cls_1024"]["acc_4way"], "keys": "expC.../eurosat_2x2_compositional_binding/built_quadrant/cls_1024/acc_4way"},
            "eurosat_2x2_cls_4way_acc_water": {"value": q["water_quadrant"]["cls_1024"]["acc_4way"], "keys": ".../water_quadrant/cls_1024/acc_4way"},
            "eurosat_2x2_cls_4way_acc_mean_of_two": {"value": round((q["built_quadrant"]["cls_1024"]["acc_4way"] + q["water_quadrant"]["cls_1024"]["acc_4way"]) / 2, 4), "keys": "mean of the two keys above"},
            "eurosat_2x2_zpatch_concat_4way_acc_built": {"value": q["built_quadrant"]["zpatch_concat_4096"]["acc_4way"], "keys": ".../built_quadrant/zpatch_concat_4096/acc_4way"},
            "eurosat_2x2_zpatch_concat_4way_acc_water": {"value": q["water_quadrant"]["zpatch_concat_4096"]["acc_4way"], "keys": ".../water_quadrant/zpatch_concat_4096/acc_4way"},
            "within_chip_anomaly_r2_zpatch_aligned": {k: dense[k]["z_patch_aligned"]["within_chip_anomaly_r2"] for k in dense},
        },
        "lesson3": {
            "var_first16_over_512_1024": {"value": round(vb["0:16"]["cls_mean_var"] / vb["512:1024"]["cls_mean_var"], 3),
                                          "keys": "expB_mrl_fossil_audit/variance_by_mrl_block/{0:16,512:1024}/cls_mean_var"},
            "var_first16_over_16_1024_mean_per_dim": {"value": round(vb["0:16"]["cls_mean_var"] / (
                (sum(vb[k]["cls_mean_var"] * w for k, w in [("16:32", 16), ("32:64", 32), ("64:128", 64), ("128:256", 128),
                                                          ("256:512", 256), ("512:1024", 512)])) / 1008), 3),
                                                      "keys": "same, weighted by block width"},
            "nlcd_full_1024_mean_r2": {"value": ex["1024"]["cls_prefix_mean_r2"], "keys": "expB.../exchangeability_by_d/1024/cls_prefix_mean_r2"},
            "nlcd_prefix64_mean_r2": {"value": ex["64"]["cls_prefix_mean_r2"], "keys": ".../64/cls_prefix_mean_r2"},
            "nlcd_random64_mean_r2": {"value": ex["64"]["cls_random_mean_r2"], "keys": ".../64/cls_random_mean_r2"},
            "nlcd_pca64_infold_mean_r2": {"value": ex["64"]["cls_infold_pca_mean_r2"], "keys": ".../64/cls_infold_pca_mean_r2"},
            "nlcd_pca64_minus_prefix64": {"value": round(ex["64"]["cls_infold_pca_mean_r2"] - ex["64"]["cls_prefix_mean_r2"], 4),
                                          "keys": "difference of the two keys above"},
            "nlcd_prefix64_imp_ci95": {"value": ex["64"]["cls_prefix_imp_ci95"], "keys": ".../64/cls_prefix_imp_ci95"},
            "nlcd_prefix64_can_ci95": {"value": ex["64"]["cls_prefix_can_ci95"], "keys": ".../64/cls_prefix_can_ci95"},
            "n_chips_nlcd": B["n_chips_nlcd"],
        },
        "lesson5": {
            "baseline_probe_mean_r2": {"value": pc["reference_metadata_boa_harmonized_um"]["matched_probe_mean_r2"],
                                       "keys": "expD.../preprocessing_contracts/reference_metadata_boa_harmonized_um/matched_probe_mean_r2"},
            "wavelength_nm_instead_of_um_unchanged_probe_r2": {"value": pc["wavelength_nm_x1000_mismatch"]["unchanged_ref_probe_mean_r2"],
                                                               "keys": ".../wavelength_nm_x1000_mismatch/unchanged_ref_probe_mean_r2"},
            "no_pb04_offset_unchanged_probe_r2": {"value": pc["unharmonized_pb04_plus1000_offset"]["unchanged_ref_probe_mean_r2"],
                                                  "keys": ".../unharmonized_pb04_plus1000_offset/unchanged_ref_probe_mean_r2"},
            "slc_off_22pct_dn0_probe_r2": {"value": cliff["dn0_unchanged_probe_mean_r2"],
                                           "keys": "expD.../missing_pixel_cliff_and_rescue/22pct_slc_off/dn0_unchanged_probe_mean_r2"},
            "slc_off_22pct_scene_meanfill_probe_r2": {"value": cliff["scene_meanfill_unchanged_probe_mean_r2"],
                                                      "keys": ".../22pct_slc_off/scene_meanfill_unchanged_probe_mean_r2"},
            "slc_off_22pct_global_meanfill_probe_r2": {"value": cliff["global_meanfill_unchanged_probe_mean_r2"],
                                                       "keys": ".../22pct_slc_off/global_meanfill_unchanged_probe_mean_r2"},
            "slc_off_meanfill_gain_over_dn0": {"value": round(cliff["scene_meanfill_unchanged_probe_mean_r2"] - cliff["dn0_unchanged_probe_mean_r2"], 4),
                                               "keys": "difference of scene_meanfill and dn0 keys"},
            "jpeg2000_10to1_probe_r2": {"value": D["jpeg2000_wavelet_compression"]["10to1"]["unchanged_probe_mean_r2"],
                                        "keys": ".../jpeg2000_wavelet_compression/10to1/unchanged_probe_mean_r2"},
            "dn0_normalized_z_min": {"value": min(D["dn0_normalized_z_per_band"]), "keys": ".../dn0_normalized_z_per_band"},
            "latlon_order": d.get("latlon_order", "documented (pre-audit file has no field)"),
        },
        "issue22": {"s1_vs_s2_mean_abs_cosine_stored_definition": d["expA_wavelength_hypernetwork"]["s1_vs_s2_mean_abs_cosine"],
                    "all_definitions": d["expA_wavelength_hypernetwork"].get("s1_vs_s2_cosine_definitions")},
    }
    return out


def probe_numbers(d):
    if d is None:
        return None
    p1 = d["p1_eurosat_multiclass_by_budget"]
    p2 = d["p3_label_noise_stress_k25"]
    p4 = d["p4_colocated_s2_nlcd_spatial_cv"]["continuous_regression_r2"]
    out = {"lesson4": {}}
    L = out["lesson4"]
    for b in ("5", "10", "50", "200"):
        if b in p1:
            L[f"eurosat_f1_budget_{b}"] = {m: p1[b][m]["mean_macro_f1"] for m in PARAM + OTHER + ["umap2d_logistic"] if m in p1[b]}
    budgets = [b for b in ("50", "200") if b in p1]  # the budgets the paper quotes ("50-200 labels/class"); the benchmark also ran 5 and 10
    allp = [(m, b, p1[b][m]["mean_macro_f1"]) for m in PARAM for b in budgets if m in p1[b]]
    mn = min(allp, key=lambda t: t[2]); mx = max(allp, key=lambda t: t[2])
    L["parametric_probe_f1_range_50_to_200_labels"] = {"min": {"method": mn[0], "labels_per_class": mn[1], "f1": mn[2]},
                                                        "max": {"method": mx[0], "labels_per_class": mx[1], "f1": mx[2]}}
    gaps = {}
    for o in OTHER:
        vals = [(b, round(p1[b]["lsvm"]["mean_macro_f1"] - p1[b][o]["mean_macro_f1"], 4)) for b in budgets if o in p1[b]]
        if vals:
            gaps[o] = {"lsvm_minus_other_by_budget": dict(vals), "min": min(v for _, v in vals), "max": max(v for _, v in vals)}
    L["lsvm_minus_other_f1_points"] = gaps
    L["paired_block_bootstrap_contrasts_seed0"] = d.get("p1_paired_contrasts_seed0_by_budget")
    nz = p2.get("noise_30pct", {})
    L["concept_search_k25_30pct_noise_ap"] = {m: nz[m]["mean_ap"] for m in nz}
    L["concept_search_k25_noise_by_level_maxsim_ap"] = {k: v["max_sim_1nn"]["mean_ap"] for k, v in p2.items() if "max_sim_1nn" in v}
    L["nlcd_canopy_r2"] = p4["nlcd_canopy_pct"]
    L["nlcd_impervious_r2"] = p4["nlcd_impervious_pct"]
    L["nlcd_n_scenes"] = d["p4_colocated_s2_nlcd_spatial_cv"]["n_scenes"]
    L["p4_latlon_order"] = d.get("latlon_order_p4", "training (pre-audit file has no field)")
    return out


def pair(old, new):
    return {"pre_audit": old, "regenerated": new}


def main(results: Path):
    ds = {
        "v15_old": load(results / "pre_audit/v15_clean_metrics.json"),
        "v15_new": load(results / "v15_clean_metrics.json"),
        "v15_training_order": load(results / "v15_clean_metrics_latlon_training.json"),
        "probe_old": load(results / "pre_audit/probe_and_lsvm_results.json"),
        "probe_new": load(results / "probe_and_lsvm_results.json"),
    }
    out = {
        "generated_by": "experiments/summarize_lessons_345.py",
        "sources": {k: (None if v is None else "present") for k, v in ds.items()},
        "v15_clean_metrics": {"pre_audit": v15_numbers(ds["v15_old"]), "regenerated_documented_latlon_order": v15_numbers(ds["v15_new"]),
                              "regenerated_training_latlon_order": v15_numbers(ds["v15_training_order"])},
        "probe_and_lsvm": {"pre_audit": probe_numbers(ds["probe_old"]), "regenerated": probe_numbers(ds["probe_new"])},
    }
    (results / "lessons_345_old_vs_new.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", results / "lessons_345_old_vs_new.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    main(Path(ap.parse_args().results))
