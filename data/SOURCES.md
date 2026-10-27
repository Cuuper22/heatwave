# Data sources (public data collected for Heatwave)

All raw files are downloaded by `scripts/fetch_*.py` into `data/raw/<source>/` (gitignored). Every fetch script is re-runnable and
skips files that are already on disk. The sha256 of each raw file is stored in `data/processed/public_*/raw_manifest.json` (and, per
series, in the `source.raw_sha256` field of the `.meta.json` sidecar). The numbers in the generated tables at the bottom come from
`results/data/public_inventory.json`, written by `scripts/fetch_inventory.py`. Loaders: `heatwave.data.public`.

All data are public data (`data_kind = "public"`). Nothing here is simulated. Where a quantity is computed by us from public columns
(carbon intensity for CAISO, DE and the EIA balancing authorities, wet-bulb temperature, 1 s bins) the series metadata says so.

Retrieval date of every download: 2026-10-26 (UTC), unless the raw manifest shows otherwise.

**Raw data policy.** To stay under the disk limit, large raw downloads (MIT Supercloud CSVs, the Alibaba, NLR, Elsayed and EIA archives) were deleted after processing; the
manifests keep their sha256 and URL, and every fetch script downloads them again (and checks the hash against the manifest) when run. Processed files are written with
`scripts/fetch_compress.py` (zstd, same content). Committed `data/processed` is about 125 MB.

## Which time scale each source serves

| scale | sources |
|---|---|
| sub-second / seconds (GPU power swings) | MIT Supercloud (V100, 1 s bins of ~9.4 Hz samples), Elsayed et al. 2026 (H100/B200 8-GPU nodes, 20 ms), NLR GenAI power 2026 (H100 multi-node jobs, 0.2 s / 0.1 s) |
| minutes (thermal, 5-min prices) | MIT Supercloud temperatures (1 s), Elsayed 2026 temperatures (1 s), CAISO real-time LMP (5 min), CAISO fuel mix / CO2 / demand (5 min) |
| hours (price, carbon, weather) | CAISO day-ahead LMP, DE-LU day-ahead price, GB carbon intensity (30 min), DE generation and CO2 proxy (15 min), EIA-930 hourly demand and generation, NASA POWER and Open-Meteo weather |

## 1. GPU telemetry

### 1.1 MIT Supercloud Dataset (labelled part, 2022-01 release) — main source for per-GPU power and temperature
- Title: The MIT Supercloud Dataset. Paper: Samsi et al., IEEE HPEC 2021, arXiv:2108.02037 (https://arxiv.org/abs/2108.02037).
- Data: `https://mit-supercloud-dataset.s3.amazonaws.com/datacenter-challenge/202201/` (`gpu/<0000-0099>/<jobid>-r<rack>-n<node>.csv`,
  `slurm-log.csv`, `labelled_jobids.csv`, `README.md`, `LICENSE`).
- Licence: **CC BY-NC-ND 4.0** (file `LICENSE` in the bucket). Attribution required, non-commercial only, **no derivatives**: the
  processed 1 s series are an adaptation of the raw samples, so they are NOT in this repository and are never published: `data/processed/public_gpu_mit*` is gitignored and rebuilt locally with `python scripts/fetch_mit_supercloud.py`. Only aggregate results computed from them (forecast scores) are committed. (The sibling folder `2022-hpca/` has stricter terms limiting use to the HPCA'22 paper analysis
  and is not used.)
- Hardware (paper, Section III): nodes with Intel Xeon Gold 6248 and two NVIDIA Volta V100 32 GB. In the file names `r` is the rack and
  `n` the node number (anonymised).
- Raw rate: `nvidia-smi` every ~106 ms per GPU (about 9.4 samples/s; the README says 100 ms). Columns used: `utilization_gpu_pct`,
  `utilization_memory_pct`, `memory_used_MiB`, `temperature_gpu`, `temperature_memory`, `power_draw_W`.
- Processing: samples are binned to **1 s** (mean; maximum for `power_max_w`); seconds without a sample stay NaN. One parquet per
  (job, node) in `data/processed/public_gpu_mit/gpu_<job>_<node>.parquet`, columns `g<k>_<metric>` per GPU plus node aggregates
  (`node_power_w` = sum of GPUs, `node_temp_c_max`, `node_util_pct_mean`, `node_mem_used_mib`): a two-level GPU -> node hierarchy.
  Jobs table: `data/processed/public_gpu_mit_jobs.parquet` (job id, node ids, GPUs, start/end, duration, model label if the job is one of the
  labelled DNN workloads, job type, series names).
- Selection (seeded, `--seed 0`): jobs that ran at least 30 min (observed telemetry >= 1800 s), at most 8 h, at most 4 nodes, node files <= 25 MB,
  spread over nodes and, for labelled jobs, over model types, within a 2 GB raw budget. Bucket totals and the size of the subset are in `results/data/public_mit_selection.json`.
  **The subset is not a random sample of cluster usage** (the dataset's own README says it is a subset of workloads and should not be used to estimate
  system utilisation).
- Not used: `node-data.csv` (2.3 GB, 5-min node load, no power), `cpu/` folders.

### 1.2 Elsayed, Al-Obaidi, Farag (2026): 8-GPU H100 and B200 nodes at 20 ms — newest per-GPU power/temperature source
- Paper: "Characterization of high-resolution AI data center training workloads on single and multiple GPU nodes", Scientific Data 13, 1268
  (2026), doi:10.1038/s41597-026-07496-6. Data: figshare doi:10.6084/m9.figshare.31654879 (`ds.zip`, 209,685,111 bytes, md5 f44c1bccc233ffe11916432d720a8034, verified).
- Licence: CC BY 4.0 (figshare record). The paper article itself is CC BY-NC-ND; the data record is CC BY.
- Content used: `Node_Dataset`, 32 unique 15-minute sessions (16 H100, 16 B200; Llama diffusion image generation and LLM training with DeepSpeed), per-GPU
  power, temperature, utilisation, memory (pynvml), 45,000 samples per session at ~20 ms. The zip stores 42 files; 10 are byte-identical copies in several
  sub-folders and are stored once. `Single_Machine_Dataset` (RTX 3060 workstation, 40 sessions) is in the raw zip but not processed.
- Processing: `data/processed/public_gpu_node_20ms/` (per-GPU `power_w` and `node_power_w` on a 20 ms grid, NaN where no sample fell in a slot) and
  `data/processed/public_gpu_node_1s/` (all metrics, 1 s means, plus 1 s power maxima). Sessions table: `data/processed/public_gpu_node_sessions.parquet`.
- Caveat: timestamps carry no time zone in the files; we interpret them as UTC (not verified). The dataset is generated by the authors' own training scripts,
  not production traffic.

### 1.3 NLR "Dataset of Generative AI Workload Power Profiles" (2026): H100 multi-node jobs at 0.1-0.2 s
- Authors: Vercellino et al., National Laboratory of the Rockies. doi:10.7799/3025227 (https://data.nlr.gov/submissions/312), paper arXiv:2604.07345.
  Download: `https://data.nlr.gov/system/files/312/1775845072-dataset.zip` (1.07 GB, redirects to an S3 bucket).
- Licence: NLR Data Catalog licence (https://data.nlr.gov/node/312/license): free use and copying provided the notice is kept and DOE/NLR/ALLIANCE are
  credited; the names may not be used to endorse products.
- Content used: the authors' aggregated files (`01_aggregated_datasets`): total power = sum over all GPUs (NVML) and CPUs (RAPL) of all nodes of a job.
  41 training runs (Llama-2 70B LoRA and Stable Diffusion, 2/4/8/16 nodes of 4 H100 each, 0.2 s) and a seeded sample of 60 of 1,026 online vLLM serving runs
  (Llama-3 70B, 0.1 s). No temperature in the aggregated files. The index of these series is a **relative** time axis (1970-01-01 + seconds since run start).
- Not processed: the 2 GB of raw per-device logs (`00_raw_datasets`), the offline inference runs, the online-rate runs and the simulated whole-facility profiles.

### 1.4 Public GPU-cluster job traces (job sizes, durations, start times)
- **Alibaba cluster-trace-gpu-v2023** (https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023; Weng et al., USENIX ATC 2023):
  `openb_pod_list_default.csv` (8,152 pods with GPU count, QoS, creation / scheduled / deletion time in seconds on a trace-relative clock) and
  `openb_node_list_gpu_node.csv` (1,213 GPU nodes). Fetched from raw.githubusercontent.com. The repository carries no licence file; its README says the
  traces may be used "for research or study purpose". -> `data/processed/public_job_traces/alibaba2023_*.parquet`.
- **Alibaba cluster-trace-gpu-v2026** (https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2026; Li et al., OSDI 2026): data on
  `tre-clusterdata.oss-cn-hangzhou.aliyuncs.com`. We fetched `asi_opensource_job_execution_summary.zip` (1.19 GB): 40.5 M GPU-pod execution spans with
  job type (training, online/offline inference, dev), GPU request (fractional allowed), duration in hours, priority class, GPU model bucket. It has **no
  start times**; the hourly pod table that does (351 GB) was not fetched. A seeded 200,000-row sample and summary statistics are committed
  (`alibaba2026_gpu_pods_sample.parquet`, `results/data/public_job_trace_summary.json`).
- Use for a job schedule: arrival times and sizes from the v2023 pods, size / duration distributions per job type from v2026.
- Microsoft Philly (2017) and SenseTime Helios (2020) job traces: blocked from the VM, collected from the author's laptop with
  `scripts/fetch_job_traces_extra.py` (sha256 and licences in `data/processed/public_job_traces/extra_manifest.json`; both repositories
  carry CC BY 4.0 licence files). Philly: Jeon et al., USENIX ATC 2019, arXiv:1901.05758, https://github.com/msr-fiddle/philly-traces;
  `philly_jobs.parquet` keeps one row per job (117,325 jobs; per-minute GPU utilization not kept). Helios: Hu et al., SC 2021,
  doi:10.1145/3458817.3476223, https://github.com/S-Lab-System-Group/HeliosData; only the Venus cluster's GPU jobs are committed
  (`helios_venus_jobs.parquet`, 125,303 GPU jobs of 246,708) to stay inside the size budget; Earth, Saturn and Uranus are re-created by the
  script. Other traces listed in the Alibaba README (v2020 PAI GPU trace, microservices) were not needed.

### 1.5 Other 2025-2026 sources found by web search
- Used: 1.2 and 1.3 above.
- Seen, not collected: arXiv:2604.04745 "The Energy Cost of Execution-Idle in GPU Clusters" (no dataset located); arXiv:2508.14318 "Power Stabilization for AI Training
  Datacenters" (paper only); Epoch AI "Data on GPU clusters" (https://epoch.ai/data/gpu-clusters, cluster sizes, not time series). The BUTTER-E energy dataset
  (catalog.data.gov) reports energy per training run, not time series.

## 2. Grid price and carbon intensity

### 2.1 CAISO OASIS prices (day-ahead and real-time), hub TH_NP15_GEN-APND
- URL: `https://oasis.caiso.com/oasisapi/SingleZip` (`PRC_LMP` version 12, market DAM; `PRC_INTVL_LMP` version 3, market RTM), CSV in zip. 30-day requests,
  one request every 6 s. Terms: CAISO public market data, no explicit open licence found; cite CAISO.
- Collected: 2024-01-01 up to the day before retrieval. DAM hourly (LMP, energy, congestion, loss, GHG components), RTM 5-minute (same components).
  Day-ahead and real-time are separate files (`public_caiso_np15_lmp_dam`, `public_caiso_np15_lmp_rtm`) so the day-ahead price can be a known-future covariate.
  Timestamps are the interval start in UTC.

### 2.2 CAISO Today's Outlook history: fuel mix, CO2, demand with day-ahead forecast
- URL: `https://www.caiso.com/outlook/history/<YYYYMMDD>/{fuelsource,co2,demand}.csv`, 5-minute, Pacific wall-clock times (converted to UTC; DST handled).
- `public_caiso_fuelmix` (MW by source), `public_caiso_carbon` (CAISO's CO2 emission rate per resource group in mTCO2/h, the unit printed at
  https://www.caiso.com/todays-outlook/emissions, plus our computed `intensity_kg_per_mwh = 1000 * total CO2 / net supply`), `public_caiso_demand`
  (actual demand, CAISO day-ahead and hour-ahead forecast; the day-ahead forecast is a known-future covariate).
- Import emissions inside CAISO's CO2 file use CAISO's own method; we do not recompute them. The published imports CO2 is negative while California is a net exporter, so the
  total-basis `intensity_kg_per_mwh` is negative in part of the record (inspect with `load_carbon('caiso', basis='total')`).
  `intensity_in_state_kg_per_mwh` (imports excluded, never negative) is the default of `load_carbon('caiso')`.

### 2.3 Great Britain carbon intensity (National Energy System Operator, formerly National Grid ESO)
- API: `https://api.carbonintensity.org.uk` (`/intensity`, `/generation`), half-hourly. Licence: CC BY 4.0 (data of the Carbon Intensity API project). Ranges
  are limited to 31 days; we request 28-day windows.
- `public_uk_carbon_intensity`: `intensity_actual_g_per_kwh` and `intensity_forecast_g_per_kwh`; `public_uk_generation_mix`: percent per fuel.
- **Caveat on the forecast column:** for past periods the API returns one forecast value; the documentation does not state its issue time. Treat it as a
  forecast with unknown vintage, not as a verified day-ahead forecast. Some periods have no forecast value (NaN).

### 2.4 Energy-Charts (Fraunhofer ISE) for DE-LU
- API: `https://api.energy-charts.info` (`/price?bzn=DE-LU`, `/public_power?country=de`, `/public_power_forecast` day-ahead for load, solar, wind onshore and offshore).
  Licence stated in every response: CC BY 4.0, from Bundesnetzagentur | SMARD.de.
- Day-ahead price: hourly until 2025-09-30, 15-minute from 2025-10-01 (the market moved to 15-minute products). `public_de_price_dam_1h` is the hourly mean (identity
  before the switch); `public_de_price_dam_native` has the 15-minute era only.
- `public_de_generation` (15 min, net generation by type, load): includes `intensity_lifecycle_proxy_g_per_kwh`, computed here from IPCC AR5 lifecycle medians
  (WGIII Annex III Table A.III.2, https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf): coal 820 (also applied to lignite and coal-derived gas), gas
  combined cycle 490, dedicated biomass 230, geothermal 38, hydro 24, utility PV 48, wind onshore 11, offshore 12 gCO2eq/kWh. Oil, waste, "others", storage and cross-border rows
  have no factor and are left out; `intensity_factor_coverage_pct` gives the covered share. It is production-based (imports ignored) and is a proxy, not an official figure.
- `public_de_forecast_dayahead`: day-ahead forecasts of load and renewables (known-future covariates).

### 2.5 EIA Hourly Electric Grid Monitor (Form EIA-930), bulk file
- `https://www.eia.gov/opendata/bulk/EBA.zip` (api.eia.gov is blocked, www.eia.gov is reachable). U.S. government data; cite EIA (https://www.eia.gov/about/copyright.php).
- Balancing authorities CISO, ERCO (Texas) and PJM: demand, **day-ahead demand forecast** (`DF`), net generation, interchange, net generation by source; hourly UTC from 2024-01-01.
  `intensity_lifecycle_proxy_g_per_kwh` computed as for DE (coal, gas, nuclear 12, solar, hydro, wind; oil and other excluded). BA-reported values can have gaps and outliers.
  The index runs past the retrieval time because day-ahead forecasts are included.

## 3. Weather (hourly)
Locations: Santa Clara CA (37.35, -121.96; CAISO NP15 / Bay Area), London (51.51, -0.13), Frankfurt (50.11, 8.68).

### 3.1 NASA POWER hourly (actuals; collected)
- API: `https://power.larc.nasa.gov/api/temporal/hourly/point` (T2M, RH2M, T2MDEW, T2MWET wet-bulb, WS10M, ALLSKY_SFC_SW_DWN). Hourly API v2.10.3 (response header). Referencing
  rules: https://power.larc.nasa.gov/docs/referencing/ (cite the project, version and access date; no fee). Fill value -999 is mapped to NaN.
- These are assimilated analysis values, not measurements and not forecasts. The response does not say whether the hour label marks the start or end of the averaging hour.

### 3.2 Open-Meteo (actuals via archive API; archived forecasts via historical-forecast and previous-runs APIs)
- Free tier, CC BY 4.0, non-commercial (https://open-meteo.com/en/license). `scripts/fetch_open_meteo.py` requests `temperature_2m, relative_humidity_2m, dew_point_2m, wind_speed_10m,
  shortwave_radiation`, adds a computed wet-bulb (Stull 2011, doi:10.1175/JAMC-D-11-0143.1) and writes `public_weather_om_{actual,forecast,dayahead}_<region>`.
  `historical_forecast` is the stitched short-lead best-match forecast; `previous_day1` is the forecast issued one day earlier (true day-ahead vintage).
- **Status:** the VM's shared egress address had used up the daily request limit (HTTP 429), so on 2026-10-26 the same
  script was run on the author's laptop (its own network and normal quota) and the processed outputs were copied back. Collected:
  2024-01-01 to 2026-10-25, hourly, for bayarea, london and frankfurt: actuals (archive), archived short-lead forecasts
  (historical-forecast, `kind="forecast"`) and day-ahead vintage forecasts (previous_day1, `vintage="dayahead"`); 24,696 rows each.
  Missing share: 0 for actuals and short-lead forecasts; day-ahead 1.8% (bayarea, frankfurt) and 4.9% (london), as printed by the fetch run.
  Raw sha256 in `data/processed/public_open_meteo/raw_manifest.json`.

## 4. Blocked / not reachable from this VM
The status of each URL checked is in `results/data/public_reachability.json` (written by `scripts/fetch_reachability.py`, plain GET through the allowlist proxy, no workaround attempted).
Summary of what was refused or missing at that time:

| URL | result |
|---|---|
| `github.com/...` HTML pages (alibaba/clusterdata, microsoft/philly-traces, S-Lab/HeliosData) | 403 (blocked); raw.githubusercontent.com works |
| `raw.githubusercontent.com/microsoft/philly-traces/master/README.md` | 404 (wrong owner; the traces live at msr-fiddle/philly-traces, collected from the laptop) |
| `raw.githubusercontent.com/S-Lab/Helios/main/README.md` | 404 (the data lives at S-Lab-System-Group/HeliosData, collected from the laptop) |
| `azurepublicdatasettraces.blob.core.windows.net/` | 400 (no listing; the Azure dataset README on raw.githubusercontent.com is reachable but its traces are VM/LLM-inference traces, not GPU power) |
| `api.eia.gov` | 403 (blocked); the bulk file on www.eia.gov replaces it (www.eia.gov answered 503 on one early check, then 200/206) |
| `unece.org/.../LCA-2.pdf` | 403 |
| `pmc.ncbi.nlm.nih.gov`, `pubmed.ncbi.nlm.nih.gov` | bot check page instead of content (not worked around) |
| `archive-api.open-meteo.com`, `previous-runs-api.open-meteo.com`, `historical-forecast-api.open-meteo.com` | 429 daily limit on the VM's shared address; collected from the author's laptop instead |

<!-- AUTO:BEGIN -->

### Processed series (generated by scripts/fetch_inventory.py from results/data/public_inventory.json)

| series | freq | rows | start (UTC) | end (UTC) | missing (all cells) | worst column missing |
|---|---|---:|---|---|---:|---:|
| `public_caiso_carbon` | 5min | 296,340 | 2024-01-01T08:00:00 | 2026-10-26T06:55:00 | 0.0021 | 0.0037 |
| `public_caiso_demand` | 5min | 296,340 | 2024-01-01T08:00:00 | 2026-10-26T06:55:00 | 0.2502 | 0.9996 |
| `public_caiso_fuelmix` | 5min | 296,340 | 2024-01-01T08:00:00 | 2026-10-26T06:55:00 | 0.0010 | 0.0018 |
| `public_caiso_np15_lmp_dam` | 1h | 24,696 | 2024-01-01T00:00:00 | 2026-10-25T23:00:00 | 0.0000 | 0.0000 |
| `public_caiso_np15_lmp_rtm` | 5min | 296,352 | 2024-01-01T00:00:00 | 2026-10-25T23:55:00 | 0.0000 | 0.0000 |
| `public_de_forecast_dayahead` | 15min | 98,785 | 2024-01-01T00:00:00 | 2026-10-26T00:00:00 | 0.0005 | 0.0010 |
| `public_de_generation` | 15min | 98,785 | 2024-01-01T00:00:00 | 2026-10-26T00:00:00 | 0.0000 | 0.0001 |
| `public_de_price_dam_1h` | 1h | 24,697 | 2024-01-01T00:00:00 | 2026-10-26T00:00:00 | 0.0000 | 0.0000 |
| `public_de_price_dam_native` | 15min | 37,449 | 2025-09-30T22:00:00 | 2026-10-26T00:00:00 | 0.0000 | 0.0000 |
| `public_eia_ciso_hourly` | 1h | 24,728 | 2024-01-01T00:00:00 | 2026-10-27T07:00:00 | 0.0032 | 0.0166 |
| `public_eia_erco_hourly` | 1h | 24,726 | 2024-01-01T00:00:00 | 2026-10-27T05:00:00 | 0.0030 | 0.0069 |
| `public_eia_pjm_hourly` | 1h | 24,725 | 2024-01-01T00:00:00 | 2026-10-27T04:00:00 | 0.0023 | 0.0048 |
| `public_uk_carbon_intensity` | 30min | 49,393 | 2023-12-31T23:30:00 | 2026-10-25T23:30:00 | 0.0011 | 0.0015 |
| `public_uk_generation_mix` | 30min | 49,393 | 2023-12-31T23:30:00 | 2026-10-25T23:30:00 | 0.0502 | 0.0502 |
| `public_weather_nasa_bayarea` | 1h | 24,696 | 2024-01-01T00:00:00 | 2026-10-25T23:00:00 | 0.0164 | 0.0836 |
| `public_weather_nasa_frankfurt` | 1h | 24,696 | 2024-01-01T00:00:00 | 2026-10-25T23:00:00 | 0.0164 | 0.0836 |
| `public_weather_nasa_london` | 1h | 24,696 | 2024-01-01T00:00:00 | 2026-10-25T23:00:00 | 0.0164 | 0.0836 |

### MIT Supercloud GPU subset

- GPU jobs: 259 (78 with a known model label), 296 per-node series, 196 distinct nodes, 29 multi-node jobs, 419 GPUs, 2,975,820 one-second rows, 1172 GPU-hours of telemetry.
- Observed duration per job (seconds): min 1836, median 9398, max 27858; jobs of at least 1800 s: 259.
- Telemetry window 2021-02-14T20:32:48 to 2021-10-01T09:51:06 UTC; mean fraction of seconds without node power: 0.0007.

### Job trace tables

- `alibaba2023_pods`: 8,152 rows
- `alibaba2023_nodes`: 1,213 rows
- `alibaba2026_gpu_pods_sample`: 200,000 rows

### Raw file hashes

Each manifest lists the sha256 and byte size of every raw file used, plus the retrieval time and URL.

| raw set | files | sha256 of the sorted (path, sha256) list |
|---|---:|---|
| `data/processed/public_caiso/raw_manifest.json` | 3146 | `96343abc11a7dd9f1acb317de98e153a745bc4e57dbcde1a9e46bfec50a3dedc` |
| `data/processed/public_eia/raw_manifest.json` | 1 | `2b24e98053282f11f44f761bc4c719a98e9ce29f7eb68dc521f223ec992ed253` |
| `data/processed/public_energy_charts/raw_manifest.json` | 72 | `72bef606ab1d0c9204ef026ebb56900cf973030bccddaca9d155365453a18a2c` |
| `data/processed/public_gpu_mit/raw_manifest.json` (local only, written by the fetch script with the series) | 306 | `af56ec3d9dfa7b300b9b07a222a5d74750c9800fee43d08362e6bf369f429655` |
| `data/processed/public_gpu_nlr/raw_manifest.json` | 1 | `8c8d1051857b51b88a8e968d361307e1f8aecd77a6e32ab982272e58c52718ad` |
| `data/processed/public_gpu_node/raw_manifest.json` | 1 | `f0547b978102523c6c9fd19c87ef9d68362225577a339d69a018a97a392763e8` |
| `data/processed/public_job_traces/raw_manifest.json` | 3 | `ec9dff81f88f4cc24915fe2c13c2fabded526c7bd4dc497d779779f75f6e1325` |
| `data/processed/public_nasa_power/raw_manifest.json` | 9 | `1f5f9efccfc96b823a8932c08bb5541cc47d00f9c9094db4e38967d3f5566cea` |
| `data/processed/public_uk_carbon/raw_manifest.json` | 74 | `6ad4bbe177eedd9ba0408f4e76bd4a96c1fd8648b93a8341803bb2dbae2c8021` |

<!-- AUTO:END -->
