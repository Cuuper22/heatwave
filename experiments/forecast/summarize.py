"""Write results/forecast/SUMMARY.md from multiscale.json and hierarchy.json (tables only).

Usage: python experiments/forecast/summarize.py [--dir results/forecast]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL = {"measured": "measured", "public": "public", "simulated": "SIMULATED", "mixed": "mixed"}


def f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}g}"


def cell(d: dict, key: str, ci: bool = True) -> str:
    s = d.get(key)
    if not s:
        return "n/a"
    return f"{f(s['est'])} [{f(s['lo'])}, {f(s['hi'])}]" if ci else f(s["est"])


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def multiscale(res: dict, prov: dict) -> list[str]:
    L = ["# Multi-scale forecasting evaluation", "",
         f"Source: `multiscale.json` (script `{prov['script']}`, git `{prov['git'].get('commit', 'unknown')[:10]}`, data kind {prov['data_kind']}). "
         f"Hardware: {res['hardware']}. Run time {res['runtime_seconds']:.0f} s.", "",
         "Entries are estimate [95% bootstrap CI over windows (origins)]. WQL = 2 x mean pinball loss (21 levels) / mean |y|. "
         "MASE = MAE of the median / in-context seasonal-naive MAE. cov80 and cov98 are the empirical coverage of the 10-90% and 1-99% quantile intervals. "
         "Rows labelled SIMULATED use gpu_stack heatwave simulator data. Timestral rows are zero-shot, without covariates.", ""]
    p = res["protocol"]
    L += ["## Protocol", ""]
    L += table(["scale", "context (steps)", "horizon (steps)", "season (steps)", "buckets (steps)"],
               [[k, str(p["context_steps"][k]), str(p["horizon_steps"][k]), str(p["season_steps"][k]),
                 ", ".join(f"{b[0]}={b[1]}-{b[2]}" for b in p["buckets"][k])] for k in p["context_steps"]])
    L += [f"Timestral 209M: equally spaced subsample of up to {p['n209_windows_per_dataset']} windows per dataset; all methods are also scored on that subset in the 209M subset tables.", ""]
    for scale in ("seconds", "minutes", "hours"):
        L += [f"## Scale: {scale}", ""]
        for name, d in res["datasets"].items():
            if d.get("scale") != scale and "skipped" not in d:
                continue
            if "skipped" in d:
                if d.get("scale", scale) == scale:
                    L += [f"### {name}: skipped ({d['skipped']})", ""]
                continue
            m = d["meta"]
            L += [f"### {name} ({LABEL[d['data_kind']]}; {d['unit']})", "",
                  f"{m['description']}. Windows: {m['n_windows']} from {m['n_series']} series ({m['n_admissible_origins']} admissible origins; stride {m['stride_steps']} steps, cap per series {m['max_origins_per_series']}); "
                  f"step {d['step_s']:g} s; context {m['context_steps']}, horizon {m['horizon_steps']}.", ""]
            meths = {k: v for k, v in d["methods"].items() if "error" not in v}
            L += table(["method", "WQL", "MASE", "MAE", "cov80", "cov98", "ms per series"],
                       [[k, cell(v["buckets"]["all"], "wql"), cell(v["buckets"]["all"], "mase"), cell(v["buckets"]["all"], "mae"),
                         f(v["buckets"]["all"]["cov80"]["est"]), f(v["buckets"]["all"]["cov98"]["est"]), f(v["seconds_per_series"] * 1000, 3)]
                        for k, v in meths.items()])
            for k, v in d["methods"].items():
                if "error" in v:
                    L += [f"`{k}` failed: {v['error']}", ""]
            bl = [b for b in next(iter(meths.values()))["buckets"] if b != "all"] if meths else []
            for stat, title in (("wql", "WQL"), ("mase", "MASE"), ("cov80", "cov80")):
                L += [f"{title} by horizon bucket:", ""]
                L += table(["method"] + bl, [[k, *[cell(v["buckets"][b], stat, ci=stat != "cov80") for b in bl]] for k, v in meths.items()])
            sub = d["subset_209M"]
            if sub["methods"]:
                L += [f"All methods on the {sub['indices_n']} of {sub['of']} windows used for Timestral 209M:", ""]
                L += table(["method", "WQL", "MASE", "cov80", "cov98"],
                           [[k, cell(v["buckets"]["all"], "wql"), cell(v["buckets"]["all"], "mase"), f(v["buckets"]["all"]["cov80"]["est"]),
                             f(v["buckets"]["all"]["cov98"]["est"])] for k, v in sub["methods"].items() if "buckets" in v])
    return L


def hierarchy(res: dict, prov: dict) -> list[str]:
    L = ["# Hierarchy coherence adjustment", "",
         f"Source: `hierarchy.json` (script `{prov['script']}`, git `{prov['git'].get('commit', 'unknown')[:10]}`, data kind {prov['data_kind']}). "
         f"Hardware: {res['hardware']}. Run time {res['runtime_seconds']:.0f} s.", "",
         "Variants: base (independent forecasts), bottom_up, ols, mint_shrink (MinT, shrinkage covariance of median errors at steps 1-10 on training windows), "
         "mint_shrink_allh (same with all 300 steps). Quantile levels are adjusted one at a time with the same matrix (approximation). "
         "Coherence error = sum |y_A - sum of bottom series below A| / sum |y_A| per aggregate level. Trees labelled SIMULATED use gpu_stack heatwave simulator data.", ""]
    p = res["protocol"]
    L += [f"Context {p['context_steps']} s, horizon {p['horizon_steps']} s. Sim test seeds {res['sim_test_seeds']}; MinT covariance from validation seeds {res['sim_val_seeds_for_mint_covariance']}.", ""]
    for tname, t in res["trees"].items():
        L += [f"## Tree {tname} ({LABEL[t['data_kind']]})", "",
              f"Nodes {t['hierarchy']['n_nodes']} (bottom {t['hierarchy']['n_bottom']}; levels {t['hierarchy']['levels']}). Test windows {t['n_test_windows']}, "
              f"training/validation windows for the MinT covariance {t['n_train_windows']}. Coherence error of the actuals: {t['actuals_coherence_error']}.", ""]
        for m, r in t["methods"].items():
            if "error" in r:
                L += [f"### {m}: failed: {r['error']}", ""]
                continue
            V = r["variants"]
            levels = list(V["base"]["levels"].keys())
            agg = levels[:-1]
            L += [f"### {m}", "", f"MinT shrinkage intensity lambda: steps 1-10 {f(r['mint_shrinkage_lambda']['steps_1_10'])}, all steps {f(r['mint_shrinkage_lambda']['all_steps'])}.", ""]
            L += ["Coherence error (median / pooled over 21 quantile levels, after sorting quantiles / pooled before sorting):", ""]
            rows = []
            for v, e in V.items():
                c = e["coherence_error"]
                rows.append([v] + [f"{f(c['median'][lv]['pooled'], 3)} / {f(c['sorted'][lv]['pooled'], 3)} / {f(c['raw'][lv]['pooled'], 3)}" for lv in agg]
                            + [f(e["quantile_crossing_share_before_sort"], 3), f(e["negative_bottom_median_share"], 3)])
            L += table(["variant"] + [f"{lv} vs bottom" for lv in agg] + ["share of crossing quantile curves before sort", "share of negative bottom medians"], rows)
            L += ["Accuracy by level, all horizon steps (estimate [95% CI over windows]); delta = variant minus base with paired CI:", ""]
            rows = []
            for lv in levels:
                for v, e in V.items():
                    b = e["levels"][lv]["buckets"]["all"]
                    d = e["levels"][lv].get("delta_vs_base", {}).get("all", {})
                    rows.append([lv, v, cell(b, "wql"), cell(b, "mase"), f(b["cov80"]["est"]), f(b["cov98"]["est"]),
                                 cell(d, "wql") if d else "-", cell(d, "mase") if d else "-"])
            L += table(["level", "variant", "WQL", "MASE", "cov80", "cov98", "WQL delta", "MASE delta"], rows)
            bl = [b for b in V["base"]["levels"][levels[0]]["buckets"] if b != "all"]
            L += ["WQL by level and horizon bucket:", ""]
            rows = []
            for lv in levels:
                for v, e in V.items():
                    rows.append([lv, v] + [cell(e["levels"][lv]["buckets"][b], "wql", ci=False) for b in bl])
            L += table(["level", "variant"] + bl, rows)
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "forecast"))
    a = ap.parse_args()
    d = Path(a.dir)
    out = ["# Forecast study summary (generated by experiments/forecast/summarize.py; tables only)", ""]
    for fn, fnc in (("multiscale.json", multiscale), ("hierarchy.json", hierarchy)):
        p = d / fn
        if p.exists():
            j = json.loads(p.read_text())
            out += fnc(j["result"], j["provenance"])
        else:
            out += [f"`{fn}` not present.", ""]
    (d / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print("wrote", d / "SUMMARY.md")


if __name__ == "__main__":
    main()
