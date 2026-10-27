# Heatwave

**Forecast-driven power and thermal management for GPU clusters, using Mistral's Timestral forecaster.**

A GPU cluster's power draw swings within a training step, dips at every checkpoint, follows the job schedule over hours, and heats the racks it runs in. The grid's price and carbon intensity swing too. Heatwave asks what a probabilistic forecast of all of that is worth when you act on it: how many more jobs fit under a power cap, how much work survives a cooling failure, and how much cheaper and cleaner training gets when flexible work waits for better hours.

Every forecast here is a set of 21 quantiles (the 1st, 5th, 10th, ..., 95th, 99th percentile of what might happen), which is what Timestral produces. A decision can then plan around a chosen quantile: plan on the 90th percentile of power and you expect to go over about one time in ten.

Heatwave is built on [gpu_stack](https://github.com/Cuuper22/gpu_stack-) (a calculator and explainer for where AI training cost comes from). The cluster simulator, the decision experiments and a new calculator scope live in gpu_stack on its `heatwave` branch as experiment e003; this repository holds the forecasting code, the data pipelines, the laptop tools and the forecasting studies, and imports gpu_stack for the simulator.

**What kind of number is each number.** Every figure below says whether it is **measured** (my laptop's GPU, logged by this repo), **public** (a published dataset, cited in [data/SOURCES.md](data/SOURCES.md)), or **simulated** (the gpu_stack e003 simulator). Every figure is read from a committed result file; [results/HEADLINE.md](results/HEADLINE.md) lists each one with the file and key it came from (`python scripts/build_headline.py` regenerates it).

## What I found

<!-- numbers: results/HEADLINE.md -->

**Zero-shot Timestral had lower forecast loss than the classical forecasts on all eleven series I tried, from 1-second GPU power to hourly grid prices.** Weighted quantile loss over all horizons (lower is better), Timestral-28M with no fine-tuning against the best of seasonal naive, ETS and ARIMA, on held-out data (temperature series for the MIT nodes and the simulated racks are in the source table):

| series | data | Timestral-28M | best classical |
|---|---|---|---|
| H100 GPU power, 1 s (Elsayed et al. 2026) | public | 0.158 | 0.259 (ETS) |
| H100 multi-node job power, 0.2 s (NLR 2026) | public | 0.076 | 0.122 (seasonal naive) |
| V100 node power, 1 s (MIT Supercloud) | public | 0.086 | 0.112 (ARIMA) |
| laptop RTX 3060 power, 1 s | measured | 0.191 | 0.218 (seasonal naive) |
| laptop RTX 3060 temperature, 1 s | measured | 0.164 | 0.180 (ARIMA) |
| CAISO real-time price, 1 h | public | 0.203 | 0.241 (seasonal naive) |
| GB carbon intensity, 1 h | public | 0.185 | 0.248 (seasonal naive) |
| simulated rack power, 1 s | simulated | 0.025 | 0.082 (ARIMA) |
| simulated site power, 1 h | simulated | 0.125 | 0.134 (ETS) |

Source: [results/forecast/multiscale.json](results/forecast/multiscale.json), table form in [results/forecast/SUMMARY.md](results/forecast/SUMMARY.md) (per horizon, with bootstrap intervals and coverage).

**Known-future inputs.** PENDING (T04b covariate ablation: results/ablation/).

**A GPU's temperature is easy to model and its throttling is easy to see coming, by a few seconds.** On my laptop's GPU (measured, two runs, 41 and 79 minutes of scripted training load), a two-node thermal model (die plus heatsink) fitted on the first run predicts the second run's temperature with R² 0.972 (RMSE 2.7 °C); the die's time constant is 1.5 s and the heatsink's 38 s. The GPU's own thermal slowdown flag starts around 84 °C (5th percentile) and halves the clock (900 vs 1,642 MHz median during training). Warning before each slowdown from a 30-second temperature forecast, tuned on run 1 and tested on run 2: F1 0.730 with Timestral-28M (median warning 18.5 s ahead) and 0.741 with ETS (16.5 s) against 0.548 for reacting to the current temperature (10.5 s). On a horizon this short, the classical models do as well as Timestral. Source: [results/thermal/](results/thermal/).

**Packing jobs under a power cap: the quantile to plan around depends on what going over costs.** Simulated, two sites, 6 held-out seeds. Reacting to measured power packs the most work (1,841 GPU-hours) and spends 72% of decision intervals over the cap (42.5 kWh above it). Planning on Timestral-5M's 90th percentile packs 1,531 GPU-hours with 7.1 kWh over the cap; Timestral-28M's 90th percentile, 4.7 kWh; seasonal naive's, 23.2 kWh. A small learned policy that picks the quantile per situation (from forecast spread, job value and the over-cap penalty) delivers $746 more net value per episode than reacting at a penalty of $40/kWh, and $5,575 more at $160/kWh; at $2.5/kWh reacting wins by $295. Source: gpu_stack `experiments/e003-heatwave/results/packing-v1.json`.

**Outage response: forecasts see some trouble coming, and in this simulator that recovers little useful work.** Simulated, 10 held-out seeds, 65 cooling failures, breaker trips, curtailments and GPU faults, 41 of them with no warning sign at all. Reacting after the fact loses 35.1 GPU-hours of work to faults, but its useful output is only 10.7 GPU-hours short of a run with no faults (1,722 vs 1,733 with no events). Watching each GPU's own temperature against its rack (outage v2) lets the forecasters warn before 35 to 46 percent of GPU faults, the ones that announce themselves (half of them do), 12 to 20 minutes ahead on median. Timestral-28M warnings cut the shortfall to 5.8 GPU-hours with 0.4 false alarms a day; the paired difference from reacting, +4.9 GPU-hours, has an interval that includes zero. Even perfect foresight ends 11.5 short: it cuts work lost to faults from 35.1 to 3.5 GPU-hours and spends 36.6 on checkpoints, moves and pauses. A follow-up that redesigns the response is in progress. The e001 rule (each site trains alone, weights averaged every 8 steps) loses only 0.7 GPU-hours to faults but pays for it in throughput: 1,267 useful GPU-hours. Source: gpu_stack `experiments/e003-heatwave/results/outage-v2.json` (v1 alongside).

**Timing flexible work saves electricity and carbon, and barely moves the total bill.** Public CAISO day-ahead prices and CAISO carbon intensity, last 26 weeks held out, 30% of the work flexible within 24 hours: scheduling on Timestral-28M forecasts cuts the electricity cost of that work by 8.6% and its carbon by 10.8%, 96% of what perfect foresight gets (9.0%); seasonal naive gets 8.3%. In Germany (DE-LU prices and carbon) the cut is 12.6% vs 13.7% with perfect foresight. Because hardware dominates the bill under gpu_stack's calculator, total cost per token moves by 0.22% (CAISO) and 1.18% (DE-LU). Source: gpu_stack `experiments/e003-heatwave/results/cost-v1.json`.

**How big a slice electricity is was itself worth checking.** An audit of gpu_stack's cost inputs against 2026 sources (prices, operators' depreciation lives, measured server power, a separate facility term) moved the calculator's example from about 6 to about 7.4 percent electricity. At the industrial electricity prices of different places, with the calculator's other defaults (5.5-year server life, PUE 1.2), the same run ranges from about 5 percent (New Mexico average) to about 16 percent (California average). Source: gpu_stack `experiments/e003-heatwave/results/cost-audit-v1.json`, with the reasoning in `experiments/e003-heatwave/cost_audit/CORRECTION.md`.

**Fine-tuning.** PENDING (T05).

## What is in here

```text
heatwave/
  forecast/     the shared 21-quantile interface, Timestral wrapper, classical baselines, metrics,
                evaluation windows, hierarchy coherence (bottom-up, OLS, MinT), covariate builders
  data/         one on-disk format for series, plus loaders for every public source
  laptop/       live logger, training load, forecast loop, dashboard and power/clock-limit controller
  sim_env.py    drives the gpu_stack simulator with public weather, price and carbon
  finetune/     PENDING
data/           processed public and measured series (sources and licences in data/SOURCES.md)
experiments/    the forecasting, ablation, thermal and calibration studies
results/        every result file the text quotes, each with its provenance
laptop/         Windows instructions and the one-command laptop run
notebooks/      PENDING (Colab fine-tuning)
scripts/        data fetchers, laptop collection, headline-number collector
```

## Reproduce

```bash
pip install -e ".[dev]"            # pulls timestral and gpu_stack (heatwave branch) from GitHub
python -m pytest -q
python scripts/build_headline.py --gpu-stack /path/to/gpu_stack-checkout
```

Each study script in `experiments/` documents its own command line; the heavier ones ran on Google Colab runtimes and say so in their result's provenance.

### The laptop demo

`python -m heatwave.laptop.run --minutes 20` logs GPU power and temperature at 10 Hz, runs a small training load with checkpoints, forecasts the next minute with Timestral-5M on the CPU every few seconds, serves a live dashboard at http://127.0.0.1:8765 with the forecast bands, and runs a controller that sets the GPU's power limit (or SM clock limit, on laptop GPUs that refuse `nvidia-smi -pl`) to keep the forecast's 90th percentile of temperature under a setpoint. It clamps every limit to the GPU's allowed range, and on exit sets the GPU to its maximum limit. Changing limits needs an administrator shell; without one it runs in dry run, logging every command it would have sent. `--dry-run` forces that. Details: [laptop/README.md](laptop/README.md). The committed live session ([laptop/logs/20261026T214028Z](laptop/logs/20261026T214028Z)) ran in dry run (no administrator rights in that shell): 25 minutes, 281 scored forecasts, Timestral-5M pinball loss 0.630 on power and 0.188 on temperature against 0.787 and 0.246 for seasonal naive (measured).

## Data and licences

Public data and their licences are listed in [data/SOURCES.md](data/SOURCES.md). The MIT Supercloud series are CC BY-NC-ND 4.0, so the processed copies are not in this repository; `scripts/fetch_mit_supercloud.py` recreates them. Timestral's code is Apache 2.0 and its weights are under a modified MIT licence ([model card](https://huggingface.co/mistralai/Timestral-209M)).

## How I built it

A lead agent planned the work, split it into self-contained tasks for parallel worker agents, and checked every result against the raw files and gpu_stack's tests and audits before merging. Several times that checking changed a result. Three examples: the simulator's first GPU power levels came from laptop ratios that put a datacenter GPU's idle draw at 76% of its working draw (public H100 data says 21%); a simulator bug kept one GPU's warning sign out of the telemetry a controller sees; and an unrestricted ETS model produced quantiles above 10¹² on one-second power data. Each was fixed and the affected studies were re-run.
