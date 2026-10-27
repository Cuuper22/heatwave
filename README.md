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
| H100 and B200 GPU power, 1 s (Elsayed et al. 2026) | public | 0.158 | 0.259 (ETS) |
| H100 multi-node job power, 0.2 s (NLR 2026) | public | 0.076 | 0.122 (seasonal naive) |
| V100 node power, 1 s (MIT Supercloud) | public | 0.086 | 0.112 (ARIMA) |
| laptop RTX 3060 power, 1 s | measured | 0.191 | 0.218 (seasonal naive) |
| laptop RTX 3060 temperature, 1 s | measured | 0.164 | 0.180 (ARIMA) |
| CAISO real-time price, 1 h | public | 0.203 | 0.241 (seasonal naive) |
| GB carbon intensity, 1 h | public | 0.185 | 0.248 (seasonal naive) |
| simulated rack power, 1 s | simulated | 0.025 | 0.082 (ARIMA) |
| simulated site power, 1 h | simulated | 0.125 | 0.134 (ETS) |

Source: [results/forecast/multiscale.json](results/forecast/multiscale.json), table form in [results/forecast/SUMMARY.md](results/forecast/SUMMARY.md) (per horizon, with bootstrap intervals and coverage).

**Inputs known ahead of time help when they describe what the series is about to do.** Timestral can take inputs whose future values are known at forecast time alongside the history. Each input here becomes visible only once it would have been published: the CAISO day-ahead price from 13:00 Pacific the day before (assumed publication time), a weather forecast 24 hours before the hour it describes. Zero-shot Timestral-209M on held-out real data, loss without and with each input, and the paired difference over the same windows with its 95% interval (moving-block bootstrap):

| series | data | input | without | with | difference (interval) |
|---|---|---|---|---|---|
| CAISO real-time price, 5 min | public | day-ahead price | 0.189 | 0.169 | −0.020 (−0.025 to −0.016) |
| CAISO real-time price, 5 min | public | day-ahead demand forecast | 0.189 | 0.180 | −0.009 (−0.012 to −0.006) |
| CAISO real-time price, 5 min | public | weather forecast | 0.189 | 0.188 | −0.001 (−0.003 to +0.002) |
| CAISO carbon intensity, 5 min | public | day-ahead demand forecast | 0.117 | 0.114 | −0.003 (−0.005 to −0.002) |
| CAISO carbon intensity, 5 min | public | weather forecast | 0.117 | 0.117 | 0.000 (−0.003 to +0.002) |
| GB carbon intensity, 30 min | public | archived carbon forecast | 0.145 | 0.041 | −0.104 (−0.114 to −0.096) |
| GB carbon intensity, 30 min | public | weather forecast | 0.145 | 0.140 | −0.005 (−0.008 to −0.002) |
| laptop GPU power, 1 s | measured | load schedule | 0.096 | 0.045 | −0.052 (−0.077 to −0.030) |
| laptop GPU power, 1 s | measured | checkpoint timing | 0.096 | 0.095 | −0.001 (−0.004 to +0.001) |
| laptop GPU temperature, 1 s | measured | load schedule | 0.036 | 0.022 | −0.014 (−0.029 to +0.006) |
| laptop GPU power, 10 Hz | measured | load schedule | 0.107 | 0.250 | +0.142 (+0.045 to +0.264) |

The GB carbon forecast comes from an archive that does not record when each forecast was issued; I assumed 24 hours ahead, and if it was issued later the gain is overstated. The same load schedule that halves the loss at one reading a second makes the 209M model worse at ten readings a second (the 28M model only slightly: +0.011, interval +0.002 to +0.023), so how a schedule is handed to the model matters as much as the schedule. A version that is told the checkpoint times the run actually logged (not knowable in advance; a comparison only) does no better than the schedule plus predicted checkpoint timing. Source: [results/ablation/ablation_real.json](results/ablation/ablation_real.json) (also Timestral-28M, and the laptop's first run). The same test on the simulated cluster, with its job schedule, training-step phase, weather forecast and price and carbon forecasts, is still running; its result will be `results/ablation/ablation_sim.json`.<!-- PENDING: ablation_sim -->

**On the simulated cluster the job schedule is the input that matters, and a price that steps at the forecast origin breaks the 1-second forecast.** Same test on the simulated plant driven by public weather, price and carbon, zero-shot Timestral-209M, 10 held-out seeds (intervals are over seeds). The planned job schedule lowers the loss on hourly site power (0.1005; −0.0095, interval −0.0172 to −0.0019), per-minute site power (0.0510; −0.0118, −0.0208 to −0.0044) and per-minute rack temperature (0.0187; −0.0018, −0.0037 to −0.0001), and adding weather, price and carbon on top changes little (all four inputs: −0.0103, −0.0124 and −0.0019). On 1-second rack power every input hurts: the schedule by +0.0381 (+0.0297 to +0.0473) on a loss of 0.0167, and price and carbon take the loss from 0.0167 to 1.01 (+0.996, +0.843 to +1.171). That extra loss sits in windows where the hourly price and carbon are flat over the 34-minute context and step to a new value at the forecast origin: on seed 100 those windows carry 99.96% of it, at 158 times the loss of the same windows without the input (28M: 14 times), while windows where the input moves inside the context lose about 3% more. Price does not drive rack power in this simulator, so the model is reacting to a step it has no evidence about in the history. Source: [results/ablation/ablation_sim.json](results/ablation/ablation_sim.json), tables in [results/ablation/SUMMARY.md](results/ablation/SUMMARY.md), the window classes in [results/ablation/diagnosis/](results/ablation/diagnosis/).

**A GPU's temperature is easy to model and its throttling is easy to see coming, by a few seconds.** On my laptop's GPU (measured, two runs, 41 and 79 minutes of scripted training load), a two-node thermal model (die plus heatsink) fitted on the first run predicts the second run's temperature with R² 0.972 (RMSE 2.7 °C); the die's time constant is 1.5 s and the heatsink's 38 s. The GPU's own thermal slowdown flag starts around 84 °C (5th percentile) and halves the clock (900 vs 1,642 MHz median during training). Warning before each slowdown from a 30-second temperature forecast, tuned on run 1 and tested on run 2: F1 0.730 with Timestral-28M (median warning 18.5 s ahead) and 0.741 with ETS (16.5 s) against 0.548 for reacting to the current temperature (10.5 s). On a horizon this short, the classical models do as well as Timestral. Source: [results/thermal/](results/thermal/).

**Packing jobs under a power cap: the quantile to plan around depends on what going over costs.** Simulated, two sites, 6 held-out seeds. Reacting to measured power packs the most work (1,841 GPU-hours) and spends 72% of decision intervals over the cap (42.5 kWh above it). Planning on Timestral-5M's 90th percentile packs 1,531 GPU-hours with 7.1 kWh over the cap; Timestral-28M's 90th percentile, 4.7 kWh; seasonal naive's, 23.2 kWh. A small learned policy that picks the quantile per situation (from forecast spread, job value and the over-cap penalty) delivers $746 more net value per episode than reacting at a penalty of $40/kWh, and $5,575 more at $160/kWh; at $2.5/kWh reacting wins by $295. Source: gpu_stack `experiments/e003-heatwave/results/packing-v1.json`.

**Outage response: forecasts see some trouble coming, and in this simulator that recovers little useful work.** Simulated, 10 held-out seeds, 65 cooling failures, breaker trips, curtailments and GPU faults, 41 of them with no warning sign at all. Reacting after the fact loses 35.1 GPU-hours of work to faults, but its useful output is only 10.7 GPU-hours short of a run with no faults (1,722 vs 1,733 with no events). Watching each GPU's own temperature against its rack (outage v2) lets the forecasters warn before 35 to 46 percent of GPU faults, the ones that announce themselves (half of them do), 12 to 20 minutes ahead on median. Timestral-28M warnings cut the shortfall to 5.8 GPU-hours with 0.4 false alarms a day; the paired difference from reacting, +4.9 GPU-hours, has an interval that includes zero. Even perfect foresight ends 11.5 short: it cuts work lost to faults from 35.1 to 3.5 GPU-hours and spends 36.6 on checkpoints, moves and pauses. A follow-up that redesigns the response is in progress. The e001 rule (each site trains alone, weights averaged every 8 steps) loses only 0.7 GPU-hours to faults but pays for it in throughput: 1,267 useful GPU-hours. Source: gpu_stack `experiments/e003-heatwave/results/outage-v2.json` (v1 alongside).

**Timing flexible work saves electricity and carbon, and barely moves the total bill.** Public CAISO day-ahead prices and CAISO carbon intensity, last 26 weeks held out, 30% of the work flexible within 24 hours: scheduling on Timestral-28M forecasts cuts the electricity cost of that work by 8.6% and its carbon by 10.8%, 96% of what perfect foresight gets (9.0%); seasonal naive gets 8.3%. In Germany (DE-LU prices and carbon) the cut is 12.6% vs 13.7% with perfect foresight. Because hardware dominates the bill under gpu_stack's calculator, total cost per token moves by 0.22% (CAISO) and 1.18% (DE-LU). Source: gpu_stack `experiments/e003-heatwave/results/cost-v1.json`.

**How big a slice electricity is was itself worth checking.** An audit of gpu_stack's cost inputs against 2026 sources (prices, operators' depreciation lives, measured server power, a separate facility term) moved the calculator's example from about 6 to about 7.4 percent electricity. At the industrial electricity prices of different places, with the calculator's other defaults (5.5-year server life, PUE 1.2), the same run ranges from about 5 percent (New Mexico average) to about 16 percent (California average). Source: gpu_stack `experiments/e003-heatwave/results/cost-audit-v1.json`, with the reasoning in `experiments/e003-heatwave/cost_audit/CORRECTION.md`.

**Fine-tuning pays off most where the model has to learn to use a known-future input; elsewhere it trims a few percent.** I fine-tuned all weights of Timestral-28M, 123M and 209M on one Colab GPU (an RTX PRO 6000) on an equal mixture of ten tasks: simulated rack power, rack temperature and site power from 50 training seeds, the public GPU, job and node power series, MIT node temperature, my laptop's first run, CAISO prices and GB carbon. Validation loss chose the learning rate and stopped training (best step 200 to 700, under three minutes of training per model). Below, the test-window loss of the best fine-tuned configuration divided by that of the best zero-shot configuration, each picked from with and without known-future inputs (below 1 means fine-tuning lowered the loss). Test windows were never read in training: simulator seeds 100 to 109, the laptop's second run, grid data from May 2026.

| series | data | 28M | 123M | 209M |
|---|---|---|---|---|
| laptop power, 1 s, with the scripted training schedule as input | measured | 0.524 | 0.498 | 0.448 |
| simulated rack power, 1 s | simulated | 0.820 | 0.802 | 0.803 |
| simulated site power, 1 h | simulated | 0.929 | 0.953 | 0.908 |
| simulated rack temperature, 1 min | simulated | 1.047 | 0.913 | 0.987 |
| laptop temperature, 1 s | measured | 0.991 | 1.011 | 1.022 |
| six public series (range) | public | 0.942 to 0.984 | 0.664 to 0.996 | 0.931 to 1.052 |

Three things qualify this. First, the large laptop and rack-power gains come with the known-future inputs. Without them, the paired difference between fine-tuned and zero-shot loss on laptop power has an interval that includes zero at every size, and zero-shot Timestral reads these inputs poorly here: on simulated rack power, adding them roughly doubles the zero-shot loss (28M: 0.054 against 0.026 without), and on laptop power they help the zero-shot 28M and 123M (0.146 and 0.144 against 0.191 and 0.193) but hurt the 209M (0.227 against 0.200). Part of what fine-tuning buys is a working path for these inputs, not a better model of the series itself. Second, without known-future inputs fine-tuning lowered loss on CAISO price and GB carbon at every size, with paired intervals that exclude zero, and raised it on the H100/B200 series at 209M (+0.0076, interval +0.0049 to +0.0106). Third, the laptop test is 70 windows from one 79-minute run, so its intervals leave out run-to-run variation. A CPU pilot tried smaller set-ups on laptop power with the schedule input: 28M with only its last blocks and head trained reached 0.095 (best zero-shot 0.146); 5M with all weights trained only brought that configuration back to the loss without the input (0.192 against 0.193); 5M with LoRA got worse (0.932, against 0.465 zero-shot with the input).¹ Source: [results/finetune/colab/finetune_colab.json](results/finetune/colab/finetune_colab.json), [results/finetune/cpu_pilot.json](results/finetune/cpu_pilot.json); every row with paired intervals is in the fine-tuning table of [results/HEADLINE.md](results/HEADLINE.md). 123M started from a zero-shot validation loss of 0.118 against 0.057 and 0.050 for 28M and 209M (`training.*.initial_val_macro`), while its zero-shot test losses are in line with the other sizes on every series, so the difference is confined to the validation windows; it reached the same validation loss as the others after training (0.046; validation curves in `training.*.validation_curve`). The fine-tuned weights are not in this repository; [notebooks/finetune_timestral_colab.ipynb](notebooks/finetune_timestral_colab.ipynb) retrains them.

¹ The pilot's weights were trained in an earlier working session on a Colab CPU runtime; the scores in its result file come from re-running the evaluation on those weights at the commit the file records.

## What is in here

```text
heatwave/
  forecast/     the shared 21-quantile interface, Timestral wrapper, classical baselines, metrics,
                evaluation windows, hierarchy coherence (bottom-up, OLS, MinT), covariate builders
  data/         one on-disk format for series, plus loaders for every public source
  laptop/       live logger, training load, forecast loop, dashboard and power/clock-limit controller
  sim_env.py    drives the gpu_stack simulator with public weather, price and carbon
  finetune/     fine-tuning data recipe (fixed splits, leakage-safe inputs), training loop, evaluation
data/           processed public and measured series (sources and licences in data/SOURCES.md)
experiments/    the forecasting, ablation, thermal and calibration studies
results/        every result file the text quotes, each with its provenance
laptop/         Windows instructions and the one-command laptop run
notebooks/      Colab notebook that runs the fine-tune on a GPU runtime
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
