"""Collect the headline numbers quoted in the READMEs from committed result files.

Every number in results/headline.json carries the file it was read from and
the key path inside it, so each figure in the writeups can be checked.
Usage: python scripts/build_headline.py [--gpu-stack /path/to/gpu_stack checkout]
Writes results/headline.json and results/HEADLINE.md. Missing result files are
reported as missing, never filled in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Book:
    def __init__(self, roots: dict[str, Path]):
        self.roots, self.rows, self.cache = roots, [], {}

    def load(self, repo: str, rel: str):
        key = (repo, rel)
        if key not in self.cache:
            p = self.roots[repo] / rel
            self.cache[key] = json.loads(p.read_text()) if p.exists() else None
        return self.cache[key]

    def get(self, section: str, label: str, repo: str, rel: str, path: list, unit: str = "", kind: str = ""):
        d = self.load(repo, rel)
        val = None
        if d is not None:
            try:
                val = d
                for k in path:
                    val = val[k]
            except (KeyError, IndexError, TypeError):
                val = None
        if kind == "" and d is not None:
            kind = d.get("provenance", {}).get("data_kind", "") if isinstance(d, dict) else ""
        self.rows.append({"section": section, "label": label, "value": val, "unit": unit, "data_kind": kind,
                          "source": f"{repo}:{rel}", "key": "/".join(str(k) for k in path),
                          "status": "ok" if val is not None else ("missing file" if d is None else "missing key")})
        return val


def wql_all(book, label, ds, method, rel="results/forecast/multiscale.json"):
    d = book.load("heatwave", rel)
    kind = (d or {}).get("result", {}).get("datasets", {}).get(ds, {}).get("data_kind", "")
    return book.get("forecasting (multi-scale)", f"{ds}: {method} WQL (all horizons)", "heatwave", rel,
                    ["result", "datasets", ds, "methods", method, "buckets", "all", "wql", "est"], kind=kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-stack", default="/home/claude/gpu_stack-")
    a = ap.parse_args()
    b = Book({"heatwave": ROOT, "gpu_stack": Path(a.gpu_stack)})

    # thermal (measured laptop)
    t = "results/thermal/laptop_rc.json"
    for model in ("one_node", "two_node"):
        b.get("thermal (measured laptop)", f"{model} RC, out-of-sample run2 R^2", "heatwave", t, ["result", "validation", "run2", model, "r2"])
        b.get("thermal (measured laptop)", f"{model} RC, out-of-sample run2 RMSE", "heatwave", t, ["result", "validation", "run2", model, "rmse_c"], "degC")
    b.get("thermal (measured laptop)", "die time constant (2-node)", "heatwave", t, ["result", "two_node", "die_tau_s"], "s")
    b.get("thermal (measured laptop)", "heatsink time constant (2-node)", "heatwave", t, ["result", "two_node", "sink_tau_s"], "s")
    b.get("thermal (measured laptop)", "thermal slowdown onset temperature, 5th pct (run1)", "heatwave", t, ["result", "throttle", "run1", "temp_c_at_sw_thermal_p05"], "degC")
    b.get("thermal (measured laptop)", "SM clock in training, slowdown vs none (run1)", "heatwave", t, ["result", "throttle", "run1", "sm_clock_mhz_training_sw_thermal_median"], "MHz")
    b.get("thermal (measured laptop)", "SM clock in training without slowdown (run1)", "heatwave", t, ["result", "throttle", "run1", "sm_clock_mhz_training_no_thermal_median"], "MHz")
    tw = "results/thermal/throttle_warning.json"
    for m in ("reactive", "ets", "arima", "timestral-5M", "timestral-28M", "timestral-28M+schedule"):
        b.get("throttle warning (measured laptop, test run2)", f"{m} F1", "heatwave", tw, ["result", "methods", m, "test", "run2", "f1"])
        b.get("throttle warning (measured laptop, test run2)", f"{m} median lead", "heatwave", tw, ["result", "methods", m, "test", "run2", "median_lead_s"], "s")
    # live laptop session
    ls = "results/laptop/20261026T214028Z/forecast_score.json"
    for m in ("timestral-5M", "seasonal-naive"):
        for q in ("power_w", "temp_c"):
            b.get("live laptop session (measured, dry-run control)", f"{m} {q} pinball", "heatwave", ls, ["result", "methods", m, q, "pinball_mean"])
    # calibration (public H100)
    c = "results/calibration/datacenter_power.json"
    for k in ("idle_ratio", "compute_fraction_of_limit", "low_phase_ratio", "low_phase_fraction_of_active_time", "temp_per_power_k_per_w"):
        b.get("simulator calibration (public H100)", k, "heatwave", c, ["result", k, "median"])
    b.get("simulator calibration (measured laptop)", "checkpoint/compute power ratio", "heatwave", c, ["result", "laptop_checkpoint", "ratio"])
    # forecasting
    for ds in ("laptop_power", "elsayed_gpu_power", "nlr_job_power", "mit_node_power", "sim_rack_power_1s", "laptop_temp",
               "mit_node_temp", "sim_rack_temp_1min", "caiso_rt_price_1h", "uk_carbon_actual_1h", "sim_site_power_1h"):
        for m in ("timestral-28M", "seasonal-naive", "ets", "arima"):
            wql_all(b, "", ds, m)
    # e003 decisions (gpu_stack)
    cost = "experiments/e003-heatwave/results/cost-v1.json"
    for pair in ("caiso", "de"):
        for pol in ("timestral_28M", "timestral_28M_cov", "seasonal_naive_24", "ets_24", "arima_seasonal", "perfect", "offline_optimum", "reactive"):
            key = f"{pol}|f=0.3|W=24|rho=0.7|q=0.5" if pol not in ("reactive",) else f"{pol}|f=0.3|W=24|rho=0.7|q=na"
            for metric in ("saving_pct_electricity", "saving_pct_co2", "saving_pct_total"):
                b.get(f"cost timing ({pair}, f=0.3, W=24 h, lambda=0.5)", f"{pol} {metric}", "gpu_stack", cost,
                      ["result", "pairs", pair, "curves", key, metric, 5], "%")
    # packing (simulated)
    pk = "experiments/e003-heatwave/results/packing-v1.json"
    for pol in ("reactive", "timestral-5M/q0.5", "timestral-5M/q0.9", "timestral-28M/q0.9", "seasonal-naive/q0.9",
                "ets/q0.9", "arima/q0.9", "oracle"):
        for metric, unit in (("gpu_hours", "GPU-h"), ("energy_above_cap_kwh", "kWh"), ("violation_rate", ""), ("stranded_kwh", "kWh")):
            b.get("power-cap packing (simulated, 6 test seeds)", f"{pol} {metric}", "gpu_stack", pk,
                  ["result", "policies", pol, metric, "mean"], unit)
    for pen in ("2.5", "10", "40", "160"):
        for stat in ("mean", "lo", "hi"):
            b.get("power-cap packing (simulated, 6 test seeds)", f"learned minus reactive net value at penalty {pen} USD/kWh ({stat})",
                  "gpu_stack", pk, ["result", "paired_net_value_difference", pen, "net_value_difference_usd_vs", "reactive", stat], "USD")
        b.get("power-cap packing (simulated, 6 test seeds)", f"best fixed policy at penalty {pen}", "gpu_stack", pk,
              ["result", "best_fixed_by_penalty", pen, "best_fixed_policy_by_mean_net_value_on_test_seeds"])
    # outage (simulated)
    for ver in ("v1", "v2", "v3"):
        oc = f"experiments/e003-heatwave/results/outage-{ver}.json"
        for pol in ("reactive", "e001", "naive", "ets", "arima", "ts5m", "ts28m", "perfect_foresight"):
            for metric, unit in (("useful_gpu_h", "GPU-h"), ("lost_gpu_h", "GPU-h"), ("deficit_gpu_h", "GPU-h")):
                b.get(f"outage response {ver} (simulated, 10 test seeds)", f"{pol} {metric}", "gpu_stack", oc,
                      ["result", "policies", pol, "overall", metric, "mean"], unit)
            if pol in ("reactive", "e001"):  # no warnings, so no recall or false alarms
                continue
            b.get(f"outage response {ver} (simulated, 10 test seeds)", f"{pol} recall", "gpu_stack", oc,
                  ["result", "policies", pol, "overall", "recall", "value"])
            b.get(f"outage response {ver} (simulated, 10 test seeds)", f"{pol} false alarms per day", "gpu_stack", oc,
                  ["result", "policies", pol, "overall", "false_alarms_per_day", "mean"])
            if ver == "v3" and pol != "reactive":
                for metric in ("useful_gpu_h_minus_reactive", "lost_gpu_h_minus_reactive"):
                    for stat in ("mean", "lo", "hi"):
                        b.get(f"outage response {ver} (simulated, 10 test seeds)", f"{pol} {metric} ({stat})", "gpu_stack", oc,
                              ["result", "paired_vs_reactive", pol, metric, stat], "GPU-h")
    # cost audit: README sentence generated from the result so it cannot drift
    ca = b.load("gpu_stack", "experiments/e003-heatwave/results/cost-audit-v1.json")
    sentences = {}
    if ca is not None:
        r = ca["result"]
        base = r["base_case"]
        rows = {x["setting"].split(":")[0]: x for x in r["one_at_a_time"]["H100"] if x["parameter"] == "electricity price"}
        nm, cal = rows["NM"]["electricity_share"], rows["CA"]["electricity_share"]
        b.get("cost audit (published prices, calculator)", "electricity share, README example, defaults", "gpu_stack",
              "experiments/e003-heatwave/results/cost-audit-v1.json", ["result", "base_case", "electricity_share"])
        sentences["readme_cost_range"] = (
            f"For a 7 billion parameter model trained on 2 trillion tokens on 1,024 H100s, power and cooling come to about "
            f"{100 * base['electricity_share']:.1f} percent of the cost. At the industrial electricity prices of different places, with "
            f"the calculator's other defaults ({base['inputs']['useful_life_years']:g}-year server life, PUE {base['inputs']['pue']:g}), the same run "
            f"ranges from about {100 * nm:.0f} percent (New Mexico average) to about {100 * cal:.0f} percent (California average).")
        sentences["readme_cost_range_source"] = ("gpu_stack:experiments/e003-heatwave/results/cost-audit-v1.json "
                                                 "result/base_case/electricity_share and result/one_at_a_time/H100 (electricity price NM, CA)")
    pkd = b.load("gpu_stack", "experiments/e003-heatwave/results/packing-v1.json")
    if pkd is not None:
        pr = pkd["result"]["paired_net_value_difference"]
        m = {k: pr[k]["net_value_difference_usd_vs"]["reactive"]["mean"] for k in pr}
        sentences["packing_learned_vs_reactive"] = (
            f"A small learned policy that picks the quantile per situation (from forecast spread, job value and the over-cap penalty) "
            f"delivers ${m['40']:,.0f} more net value per episode than reacting at a penalty of $40/kWh, and ${m['160']:,.0f} more at $160/kWh; "
            f"at $2.5/kWh reacting wins by ${-m['2.5']:,.0f}.")
    # known-future inputs (covariate ablation): WQL per input set and the paired difference against no inputs
    for abf, sec in [("results/ablation/ablation_real.json", "known-future inputs, real data (test windows)"),
                     ("results/ablation/ablation_sim.json", "known-future inputs, simulated cluster (test seeds)")]:
        ab = b.load("heatwave", abf)
        if ab is None:
            b.get(sec, "result", "heatwave", abf, ["result"])
            continue
        for ds, v in ab["result"]["datasets"].items():
            for m, mv in v["methods"].items():
                for c, cv in (mv.get("configs") or {}).items():
                    bk = [k for k in cv["buckets"] if k.startswith("all")][0]
                    base = ["result", "datasets", ds, "methods", m, "configs", c, "buckets", bk]
                    b.get(sec, f"{ds} {m} {'+'.join(cv['groups']) or 'none'} WQL", "heatwave", abf, base + ["wql"], kind=v.get("data_kind", ""))
                    if cv["buckets"][bk].get("delta_wql") is not None:
                        b.get(sec, f"{ds} {m} {'+'.join(cv['groups'])} minus none (paired)", "heatwave", abf, base + ["delta_wql"], kind=v.get("data_kind", ""))
                        b.get(sec, f"{ds} {m} {'+'.join(cv['groups'])} minus none, interval", "heatwave", abf, base + ["delta_wql_ci95"], kind=v.get("data_kind", ""))
    # fine-tuning (GPU, Colab): zero-shot vs fine-tuned, with and without covariates, per dataset
    ftf = "results/finetune/colab/finetune_colab.json"
    ft = b.load("heatwave", ftf)
    if ft is not None:
        ev = ft["result"]["evaluation"]
        for size in ev:
            for ds, v in ev[size]["datasets"].items():
                kind = v.get("data_kind", "")
                for mode in v["modes"]:
                    for model in v["modes"][mode]["models"]:
                        b.get(f"fine-tuning, Timestral-{size} (test windows)", f"{ds} {mode} {model} WQL", "heatwave", ftf,
                              ["result", "evaluation", size, "datasets", ds, "modes", mode, "models", model, "all", "wql", "est"], kind=kind)
        def _delta(v, mode):
            # fine-tuned minus zero-shot WQL on the same windows, bootstrap interval (from the result file)
            d = v["modes"].get(mode, {}).get("delta_vs_reference", {})
            for dv in d.values():
                w = dv.get("all", {}).get("wql")
                if w:
                    return {"est": w["est"], "lo": w["lo"], "hi": w["hi"]}
            return None
        rows = []
        for size in ev:
            for ds, v in ev[size]["datasets"].items():
                m = {(mode, ("zs" if mm.startswith("zero") else "ft")): v["modes"][mode]["models"][mm]["all"]["wql"]["est"]
                     for mode in v["modes"] for mm in v["modes"][mode]["models"]}
                zs_best = min(x for (mode, w), x in m.items() if w == "zs")
                ft_best = min(x for (mode, w), x in m.items() if w == "ft")
                rows.append({"size": size, "dataset": ds, "data_kind": v.get("data_kind", ""), "zero_shot_best": zs_best,
                             "finetuned_best": ft_best, "ratio_best_ft_over_best_zs": ft_best / zs_best,
                             "zero_shot_none": m.get(("none", "zs")), "zero_shot_cov": m.get(("cov_all", "zs")),
                             "finetuned_none": m.get(("none", "ft")), "finetuned_cov": m.get(("cov_all", "ft")),
                             "paired_delta_none": _delta(v, "none"), "paired_delta_cov": _delta(v, "cov_all")})
        sentences["finetune_rows_note"] = ("ratio_best_ft_over_best_zs compares the best fine-tuned configuration with the best zero-shot "
                                           "configuration (with or without covariates) on the same test windows")
        finetune_table = rows
    else:
        finetune_table = []
    Path(ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/headline.json").write_text(json.dumps({"rows": b.rows, "sentences": sentences, "finetune_table": finetune_table}, indent=1, default=str) + "\n")
    lines = ["# Headline numbers (generated by scripts/build_headline.py; every value cites its file and key)", ""]
    sec = None
    for r in b.rows:
        if r["section"] != sec:
            sec = r["section"]
            lines += ["", f"## {sec}", "", "| quantity | value | unit | data | source | key |", "|---|---|---|---|---|---|"]
        v = r["value"]
        vs = f"{v:.4g}" if isinstance(v, (int, float)) else (r["status"] if v is None else str(v))
        lines.append(f"| {r['label']} | {vs} | {r['unit']} | {r['data_kind']} | {r['source']} | {r['key']} |")
    if finetune_table:
        lines += ["", "## Fine-tuned vs zero-shot, best configuration of each (WQL on test windows)", "",
                  "| size | dataset | data | zero-shot none | zero-shot cov | fine-tuned none | fine-tuned cov | best ft / best zs | ft-zs none [interval] | ft-zs cov [interval] |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda x: "" if x is None else f"{x:.4f}"
        g = lambda d: "" if d is None else f"{d['est']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
        for r in finetune_table:
            lines.append(f"| {r['size']} | {r['dataset']} | {r['data_kind']} | {f(r['zero_shot_none'])} | {f(r['zero_shot_cov'])} | "
                         f"{f(r['finetuned_none'])} | {f(r['finetuned_cov'])} | {r['ratio_best_ft_over_best_zs']:.3f} | "
                         f"{g(r['paired_delta_none'])} | {g(r['paired_delta_cov'])} |")
    if sentences:
        lines += ["", "## Generated sentences (paste verbatim)", ""] + [f"- {k}: {v}" for k, v in sentences.items()]
    (ROOT / "results/HEADLINE.md").write_text("\n".join(lines) + "\n")
    missing = [r for r in b.rows if r["status"] != "ok"]
    print(f"{len(b.rows)} rows, {len(missing)} missing")


if __name__ == "__main__":
    main()
