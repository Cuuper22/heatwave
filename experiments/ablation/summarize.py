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
    rows = ["| config | " + " | ".join(f"dWQL {l}" for l in labels) + " | dWQL all | relative dWQL all | dMASE all" + cl + " |",
            "|---|" + "---|" * (len(labels) + 3 + (1 if cluster else 0))]
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
                    + f" | {100 * rel:+.2f}% [{100 * lo:+.2f}%, {100 * hi:+.2f}%] | " + dci(b["all"], "delta_mase") + extra + " |")
    return rows


def sim_section(d: dict) -> list[str]:
    out = ["# Simulated data (e003 sim-scenario-v2). Every number in this part is SIMULATED.", ""]
    des = d["design"]
    out += [f"Forward-addition order (declared before scoring): {' , '.join(des['forward_order'])}.", ""]
    out += ["| group | covariate | channels |", "|---|---|---|"]
    out += [f"| {k} | {v['name']} | {', '.join(v['channels'])} |" for k, v in des["groups"].items()]
    out += ["", "Variants:"] + [f"- {k}: {v}" for k, v in des["variants"].items()]
    out += ["", "Leakage rules:"] + [f"- {k}: {v}" for k, v in des["leakage_rules"].items()]
    out += ["", f"Bootstrap: {des['bootstrap']}.", f"Seeds: {des['seeds']}. Origin stride multiplier: {des['stride_mult']}. "
            f"Hardware: {JSON_ENC.encode(d['meta'].get('hardware'))}. Run complete: {d.get('complete')}.", ""]
    roles = {n: c["roles"] for n, c in des["configs"].items()}
    for name, ds in d["datasets"].items():
        sc = ds["scale"]
        out += [f"## {name}: {ds['what']} [SIMULATED]", "",
                f"context {sc['context']} steps, horizon {sc['horizon']} steps of {sc['step']}, "
                f"{ds['subsampling']['n_origins']} origins (stride {ds['subsampling']['origin_stride_steps']} steps), "
                f"seeds {ds['subsampling']['seeds']}, units {ds['subsampling']['units']}.", ""]
        for mname, block in ds["methods"].items():
            labels = sc["buckets"][:-1]
            out += [f"### {name} / {mname}", "", "Scores per configuration (WQL per horizon bucket; MASE and coverage over the whole horizon):", ""]
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
            out += [f"### {name} / {mname}", ""] + metric_table(block, labels) + [""]
            if len(block["configs"]) > 1:
                out += ["Change in WQL against `none` (point estimate [95% CI]):", ""] + delta_table(block, labels) + [""]
    return out


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
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
