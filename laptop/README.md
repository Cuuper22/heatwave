# Heatwave on the laptop

Logs GPU power and temperature at 10 Hz, runs a small synthetic training load
with periodic checkpoints, forecasts the next minute of power and temperature
with Timestral (5M, on the CPU) next to a seasonal-naive baseline, shows it all
on a live local dashboard, and runs a controller that sets a power limit (or an
SM clock limit) so the forecast quantile of GPU temperature stays under a
setpoint.

Everything is started by one command and written to `laptop/logs/<UTC timestamp>/`.

## Install (Windows, PowerShell)

Needs Python 3.12, `git`, and a CUDA build of torch (the laptop already has
torch 2.6.0+cu124). Timestral is installed from GitHub, so `git` must be on PATH.

```powershell
git clone <this repository> heatwave
cd heatwave
git checkout w/laptop            # or main once merged
pip install -e .
pip install nvidia-ml-py         # NVML telemetry; without it nvidia-smi is parsed instead
```

Or use the wrapper, which creates a venv (it reuses the installed CUDA torch),
installs, and runs a 20-minute dry run:

```powershell
powershell -ExecutionPolicy RemoteSigned -File laptop\run_laptop.ps1
```

## Commands

Live control (changes GPU limits; needs an administrator PowerShell, see below):

```powershell
python -m heatwave.laptop.run --minutes 20 --quantile 0.9 --temp-limit 75
```

Dry run (prints the nvidia-smi commands, runs none of them, needs no admin rights):

```powershell
python -m heatwave.laptop.run --minutes 20 --dry-run
```

Data collection only (logger + load, no forecast, no control, no dashboard):

```powershell
python -m heatwave.laptop.run --log-only --minutes 60
```

Other flags: `--model-size 5M|28M|123M|209M` (Timestral), `--no-load` (log whatever else
is using the GPU), `--power-budget <W>` (also keep the forecast power quantile under a
budget), `--load-size tiny|small|medium`, `--horizon 60`, `--port 8765`,
`--control-forecaster timestral|seasonal-naive`. `python -m heatwave.laptop.run --help` lists all.

Score a run again later (also done automatically at the end of a run):

```powershell
python -m heatwave.laptop.score laptop\logs\<run>
```

Print the load model's parameter counts: `python -m heatwave.laptop.load --events x.csv --describe`.

## Dashboard

Open http://127.0.0.1:8765 while a run is going. It shows power and temperature,
the forecast bands (1-99% and 10-90%), the seasonal-naive median, checkpoint markers
(logged and scheduled), the setpoint and budget lines, the controller's limit and the
enforced power limit. The mode is shown in large letters: DRY RUN, or LIVE CONTROL.
A SIMULATED DATA badge appears for fake-backend runs.

## Live control and administrator rights

Real actuation is the default. `nvidia-smi -pl` and `nvidia-smi -lgc` need
administrator rights, so for live control start PowerShell as administrator:
press Start, type `PowerShell`, right-click "Windows PowerShell", choose
"Run as administrator", then `cd` into the clone and run the live command above.

Without administrator rights the controller does not fail: it stays in DRY RUN and says why
in the console, in `control.jsonl` and on the dashboard.

At start the controller checks what works:

1. It sends `nvidia-smi -i 0 -pl <current limit>` (the current value, so nothing changes).
   If that works, it controls the power limit.
2. If the driver says the power limit is not supported (common on laptop GPUs, where
   `power.limit` reports `[N/A]` and Dynamic Boost manages power), it tries
   `nvidia-smi -i 0 -lgc <min>,<max>` with the full clock range and, if that works,
   controls the SM clock cap instead.
3. If neither works it stays in DRY RUN and logs the driver's message.

The error-message matching is by keyword (permission, not supported); driver wording
differs between versions, and anything unrecognized is treated as "did not work".

### Limits and rate limiting

Every requested limit is clamped to the GPU's reported minimum and maximum and never goes
below a floor of 50% of the default power limit (or of the maximum SM clock in clock mode;
`--floor-frac` changes it). A limit changes at most once per `--min-interval` seconds (10 by default)
and by at most 5 W (100 MHz in clock mode) per change. Every command is printed
before it runs, for example `[control] running: nvidia-smi -i 0 -pl 72`.

### What happens on exit

On exit the controller sets the GPU to its maximum allowed power limit
(`nvidia-smi -i 0 -pl <max>`, with the maximum read from the GPU) or, in clock mode, runs
`nvidia-smi -i 0 -rgc` (clocks back to the default maximum), and leaves it there. It does
not put back the value from before the run. This happens on a normal finish, on Ctrl+C, on SIGTERM and
Windows CTRL_BREAK, and when the program stops on an error (`try/finally`, `atexit` and signal handlers).
If setting the maximum fails, the console prints the exact command to run by hand.
In `--dry-run` nothing was changed, so nothing is run on exit.

If a run is ever stopped in a way that skips all of this (for example the window is closed
by force), set the GPU back by hand from an administrator PowerShell:

```powershell
nvidia-smi -i 0 -rgc
nvidia-smi -i 0 -pl 105      # use the maximum shown by: nvidia-smi -q -d POWER
```

## Files written to `laptop/logs/<run>/`

| file | content |
| --- | --- |
| `telemetry.csv` | 10 Hz: UTC time, monotonic seconds, power, energy, temperature, SM and memory clock, utilization, fan, power limit, enforced limit, memory used, `data_kind` |
| `events.csv` | training load: `config`, every `step`, `ckpt_start`, `ckpt_end`, `end` (UTC time stamps) |
| `forecasts.jsonl` | every forecast: time issued, origin, horizon, 21 quantiles for power and temperature, Timestral and seasonal naive |
| `control.jsonl` | every controller cycle: peaks, decision, command |
| `meta.json`, `summary.json` | run settings, GPU limits, controller mode, summary numbers with provenance |
| `load_summary.json`, `load.log` | the load process's own summary (parameter count, peak CUDA memory) and output |

`results/laptop/<run>/forecast_score.json` holds pinball loss, WQL, coverage of the 80% and 98%
intervals and the MAE of the median, Timestral against seasonal naive, for forecasts whose
whole horizon lies inside the log. The score file records `data_kind` as `measured` for GPU runs
and `simulated` for fake-backend runs.

## Seasonal-naive baseline and covariates

The baseline repeats the last checkpoint cycle (season = the measured checkpoint period in
seconds; plain persistence until two checkpoints have been seen). Timestral gets two
known-future covariates built from `events.csv`: whether a checkpoint is in progress, and the
phase within the checkpoint cycle. Future values come from the measured period, so forecasts
that start before the first two checkpoints carry no schedule information.

## Try it without a GPU (SIMULATED)

```
python -m heatwave.laptop.run --fake --minutes 1 --horizon 12 --forecast-every 3 --min-context 15
```

The fake backend is an RC thermal model with made-up constants (the 80 W default, 1 W minimum and
105 W maximum power limits are the values `nvidia-smi` reported on this laptop's RTX 3060 Laptop GPU).
Everything it writes is labeled `simulated`. `--fake-smi clock|power|noadmin` selects which
`nvidia-smi` behavior it imitates.

## Notes

* The controller gains, step sizes and the 50% floor are tuning defaults, not measurements.
* In `--dry-run` the controller assumes its own commands took effect, so the decisions follow an
  open loop and the logged temperatures do not respond to them.
* The load's VRAM use is reported in `load_summary.json` after a run on the GPU; pick `--load-size small`
  or `tiny` if `medium` does not fit in 6 GB.
