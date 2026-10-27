"""Generate results/ablation/SUMMARY.md (tables only) from ablation_sim.json and ablation_real.json.

Usage: python experiments/ablation/summarize.py [--sim PATH] [--real PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JSON_ENC = json.JSONEncoder()
JSON_ENC_INDENT = json.JSONEncoder(indent=1)
ROOT = Path(__file__).resolve().parents[2]
ROLE_ORDER = [("baseline", "No covariates"), ("alone", "Each covariate alone"),
              ("forward", "Forward addition (pre-declared order)"), ("all", "All four"), ("leave_out", "Leave one out from all four")]


def g(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}g}"


def dci(e, key="delta_wql"):
    if key not in e:
        return ""
    lo, hi = e[f"{key}_ci95"]
    return f"{e[key]:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def note_line(block: dict) -> str:
    if "windows" not in block:
        return ""
    return (f"Windows per configuration: {block['windows']}. Constant non-binary covariate rows zeroed (rows per group): "
            f"{block['constant_rows_zeroed']} (guard on: {block['const_guard']}). Windows the model rejected, by configuration: "
            f"{block['failed_windows_by_config'] or 'none'}.")


def metric_table(block: dict, labels: list[str]) -> list[str]:
    cfgs = block["configs"]
    rows = ["| config | covariates | " + " | ".join(f"WQL {l}" for l in labels) + " | MASE all | cov80 all | cov98 all |",
            "|---|---|" + "---|" * (len(labels) + 3)]
    for c, e in cfgs.items():
        b = e["buckets"]
        rows.append(f"| {c} | {'+'.join(e['groups']) or '-'} | " + " | ".join(g(b[l]["wql"]) for l in labels)
                    + f" | {g(b['all']['mase'])} | {g(b['all']['cov80'], 3)} | {g(b['all']['cov98'], 3)} |")
    return rows


def delta_table(block: dict, labels: list[str], names=None, cluster: bool = False) -> list[str]:
    cl = " | dWQL all, seed-cluster 95% CI" if cluster else ""
    rows = ["| config | " + " | ".join(f"dWQL {l}" for l in labels) + " | dWQL all | relative dWQL all | dMASE all | share of windows with pinball > 10x none | median per-window pinball ratio" + cl + " |",
            "|---|" + "---|" * (len(labels) + 5 + (1 if cluster else 0))]
    for c, e in block["configs"].items():
        if c == "none" or (names is not None and c not in names):
            continue
        b = e["buckets"]
        rel = b["all"]["rel_delta_wql"]
        lo, hi = b["all"]["rel_delta_wql_ci95"]
        extra = ""
        if cluster:
            clo, chi = b["all"]["delta_wql_cluster_ci95"]
            extra = f" | [{clo:+.4f}, {chi:+.4f}]"
        rows.append(f"| {c} | " + " | ".join(dci(b[l]) for l in labels) + f" | {dci(b['all'])}"
                    + f" | {100 * rel:+.2f}% [{100 * lo:+.2f}%, {100 * hi:+.2f}%] | " + dci(b["all"], "delta_mase")
                    + f" | {g(e.get('blowup_rate_vs_none'), 3)} | {g(e.get('median_pinball_ratio_vs_none'), 3)}" + extra + " |")
    return rows


def sim_section(d: dict) -> list[str]:
    out = ["# Mixed data: SIMULATED cluster (e003 sim-scenario-v3) driven by PUBLIC weather, price and carbon", ""]
    des = d["design"]
    out += [f"Forward-addition order (declared before scoring): {' , '.join(des['forward_order'])}.", ""]
    out += ["| group | covariate | channels |", "|---|---|---|"]
    out += [f"| {k} | {v['name']} | {', '.join(v['channels'])} |" for k, v in des["groups"].items()]
    env = des["environment"]
    out += ["", f"Scenario: {JSON_ENC.encode(des['scenario'])}.", "", "Environment:",
            f"- kind: {env['kind']}", f"- week per seed: {env['week_start_rule']}",
            f"- USD per EUR: {env['usd_per_eur']:.6g} (source: {env['usd_per_eur_source']})"]
    out += [f"- {k}: {v}" for k, v in env["sites"].items()]
    out += ["", "Leakage rules:"] + [f"- {k}: {v}" for k, v in des["leakage_rules"].items()]
    out += ["", f"Bootstrap: {des['bootstrap']}.", f"Seeds: {des['seeds']}. Origin stride multiplier: {des['stride_mult']}. "
            f"Hardware: {JSON_ENC.encode(d['meta'].get('hardware'))}. Run complete: {d.get('complete')}.", ""]
    roles = {n: c["roles"] for n, c in des["configs"].items()}
    for name, ds in d["datasets"].items():
        sc = ds["scale"]
        out += [f"## {name}: {ds['what']} [MIXED: SIMULATED plant, PUBLIC environment]", "",
                f"context {sc['context']} steps, horizon {sc['horizon']} steps of {sc['step']}, "
                f"{ds['subsampling']['n_origins']} origins (stride {ds['subsampling']['origin_stride_steps']} steps), "
                f"seeds {ds['subsampling']['seeds']}, units {ds['subsampling']['units']}, week starts (UTC) {JSON_ENC.encode(ds['subsampling'].get('week_start_utc'))}.", ""]
        for mname, block in ds["methods"].items():
            labels = sc["buckets"][:-1]
            out += [f"### {name} / {mname}", "", note_line(block), "",
                    "Scores per configuration (WQL per horizon bucket; MASE and coverage over the whole horizon):", ""]
            out += metric_table(block, labels)
            if "none" not in block["configs"] or len(block["configs"]) == 1:
                out.append("")
                continue
            out += ["", "Change in WQL against `none` (point estimate [95% CI], moving-block bootstrap over origins):", ""]
            for key, title in ROLE_ORDER[1:]:
                names = [c for c, r in roles.items() if any(x == key or x.startswith(key + "_") for x in r)]
                if key == "forward":
                    names = [c for c, r in roles.items() if any(x.startswith("forward") for x in r)]
                if key == "all":
                    names = [c for c, r in roles.items() if "all" in r]
                if not names:
                    continue
                out += [f"**{title}**", ""] + delta_table(block, labels, names, cluster=True) + [""]
            if ds["subsampling"]["n_origins"] < 100:
                out += [f"Note: with {ds['subsampling']['n_origins']} origins the moving-block interval can exclude its own point estimate; "
                        "quote the seed-cluster interval for this dataset.", ""]
    return out


def real_section(d: dict) -> list[str]:
    out = ["# Real data: public grid/carbon series and MEASURED laptop GPU runs", ""]
    des = d["design"]
    out += [f"Split: {des['split']}.", "", "Leakage rules:"] + [f"- {k}: {v}" for k, v in des["leakage_rules"].items()]
    out += ["", f"Bootstrap: {des['bootstrap']}. Hardware: {JSON_ENC.encode(d['meta'].get('hardware'))}. Run complete: {d.get('complete')}.", ""]
    dg = d.get("diagnostics", {})
    if dg:
        out += ["Diagnostics:", "", "```", JSON_ENC_INDENT.encode(dg), "```", ""]
    for name, ds in d["datasets"].items():
        sc = ds["scale"]
        out += [f"## {name}: {ds['what']} [{ds['data_kind'].upper()}]", "",
                f"context {sc['context']} steps, horizon {sc['horizon']} steps of {sc['step']}, "
                f"{ds['subsampling']['n_origins']} origins (stride {ds['subsampling']['origin_stride_steps']} steps).", "",
                "Covariate groups: " + "; ".join(f"{k} = {v}" for k, v in ds["covariate_groups"].items()), ""]
        for mname, block in ds["methods"].items():
            labels = sc["buckets"][:-1]
            out += [f"### {name} / {mname}", "", note_line(block), ""] + metric_table(block, labels) + [""]
            if len(block["configs"]) > 1:
                out += ["Change in WQL against `none` (point estimate [95% CI]):", ""] + delta_table(block, labels) + [""]
    return out


def diagnosis_section(path: Path) -> list[str]:
    d = json.loads(path.read_text())["result"]
    out = [f"# Diagnosis: where the extra loss of a covariate group comes from ({d['dataset']}, seed {d['seed']})", "",
           "Generated from " + path.name + " (experiments/ablation/diagnose_horizon_steps.py). Each window is classed by the shape of each "
           "covariate row after the constant-row guard: constant over the context or varying in it, and changing inside the horizon or not. "
           "Loss ratio = summed per-window loss with the group / without it.", ""]
    for g, blk in d["groups"].items():
        out += [f"## group {g} ({', '.join(blk['channels'])})", "", "| model | row shape | windows | loss with / without | share of the extra loss |", "|---|---|---|---|---|"]
        for m, classes in blk["by_class"].items():
            for k, v in classes.items():
                if k.startswith("_"):
                    continue
                share = v["share_of_extra_loss"]
                out.append(f"| {m} | {k} | {v['n_windows']} | {g_(v['loss_with_over_without'])} | {'' if share is None else g_(share)} |")
        out.append("")
    return out


def g_(x):
    return f"{x:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=str(ROOT / "results/ablation/ablation_sim.json"))
    ap.add_argument("--real", default=str(ROOT / "results/ablation/ablation_real.json"))
    ap.add_argument("--out", default=str(ROOT / "results/ablation/SUMMARY.md"))
    a = ap.parse_args()
    lines = ["# Covariate ablation summary", "",
             "Generated by experiments/ablation/summarize.py from ablation_sim.json and ablation_real.json. "
             "WQL = weighted quantile loss (lower is lower loss). dWQL = WQL(config) - WQL(none).", ""]
    for path, fn in ((a.sim, sim_section), (a.real, real_section)):
        if Path(path).exists():
            lines += fn(json.loads(Path(path).read_text())["result"]) + [""]
    for dp in sorted((ROOT / "results/ablation/diagnosis").glob("*.json")):
        lines += diagnosis_section(dp) + [""]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
