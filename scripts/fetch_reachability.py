#!/usr/bin/env python
"""Record the HTTP status of every public-data host/URL considered, for the "blocked / not reachable" list in data/SOURCES.md.

Plain GET (first byte only where the server honours Range), through the pre-configured allowlist proxy, no retries
around a refusal. Output: results/data/public_reachability.json (via write_result).
Usage: python scripts/fetch_reachability.py
"""

from __future__ import annotations

import sys

import requests

from heatwave.data.public._common import REPO_ROOT, USER_AGENT
from heatwave.provenance import write_result

URLS = {
    # GPU telemetry and job traces
    "mit_supercloud_bucket": "https://mit-supercloud-dataset.s3.amazonaws.com/?list-type=2&delimiter=/",
    "alibaba_clusterdata_raw_readme": "https://raw.githubusercontent.com/alibaba/clusterdata/master/README.md",
    "alibaba_clusterdata_github_page": "https://github.com/alibaba/clusterdata",
    "alibaba_v2026_oss_data": "https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-gpu-v2026/data/asi_opensource_job_execution_summary.zip",
    "microsoft_philly_github_page": "https://github.com/microsoft/philly-traces",
    "microsoft_philly_raw_readme": "https://raw.githubusercontent.com/microsoft/philly-traces/master/README.md",
    "azure_public_dataset_raw_readme": "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/README.md",
    "azure_public_dataset_blob_root": "https://azurepublicdatasettraces.blob.core.windows.net/",
    "helios_github_page": "https://github.com/S-Lab/HeliosData",
    "helios_raw_readme": "https://raw.githubusercontent.com/S-Lab/Helios/main/README.md",
    "acmetrace_raw_readme": "https://raw.githubusercontent.com/InternLM/AcmeTrace/main/README.md",
    "unece_lca_pdf": "https://unece.org/sites/default/files/2021-10/LCA-2.pdf",
    # grid
    "caiso_oasis": "https://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20260101T08:00-0000&enddatetime=20260102T08:00-0000&version=12&market_run_id=DAM&node=TH_NP15_GEN-APND&resultformat=6",
    "caiso_outlook_fuelsource": "https://www.caiso.com/outlook/history/20260101/fuelsource.csv",
    "uk_carbon_intensity_api": "https://api.carbonintensity.org.uk/intensity/2026-01-01T00:00Z/2026-01-02T00:00Z",
    "energy_charts_price": "https://api.energy-charts.info/price?bzn=DE-LU&start=2026-01-01&end=2026-01-02",
    "eia_api": "https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly",
    "eia_www_gridmonitor": "https://www.eia.gov/electricity/gridmonitor/",
    "eia_www_opendata": "https://www.eia.gov/opendata/",
    "eia_www_bulk_download": "https://www.eia.gov/opendata/bulk/EBA.zip",
    # weather
    "open_meteo_forecast": "https://api.open-meteo.com/v1/forecast?latitude=37.35&longitude=-121.96&hourly=temperature_2m&forecast_days=1",
    "open_meteo_archive": "https://archive-api.open-meteo.com/v1/archive?latitude=37.35&longitude=-121.96&start_date=2026-01-01&end_date=2026-01-02&hourly=temperature_2m",
    "open_meteo_historical_forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=37.35&longitude=-121.96&start_date=2026-01-01&end_date=2026-01-02&hourly=temperature_2m",
    "open_meteo_previous_runs": "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=37.35&longitude=-121.96&start_date=2026-01-01&end_date=2026-01-02&hourly=temperature_2m_previous_day1",
    "nasa_power_hourly": "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M&community=RE&longitude=-121.96&latitude=37.35&start=20260101&end=20260102&format=JSON&time-standard=UTC",
    "noaa_api_weather_gov": "https://api.weather.gov/",
    "ncei_global_hourly_csv": "https://www.ncei.noaa.gov/data/global-hourly/access/2026/72494023234.csv",
    "ipcc_ar5_annex_iii_pdf": "https://www.ipcc.ch/site/assets/uploads/2018/02/ipcc_wg3_ar5_annex-iii.pdf",
}


def main() -> int:
    out = {}
    for key, url in URLS.items():
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"}, timeout=40, stream=True, allow_redirects=False)
            body = ""
            if r.status_code in (400, 403, 429, 503):
                body = r.raw.read(200, decode_content=True).decode("utf8", "replace").replace("\n", " ")[:160]
            out[key] = {"url": url, "http_status": r.status_code, "note": body}
            r.close()
        except requests.RequestException as exc:
            out[key] = {"url": url, "http_status": None, "note": type(exc).__name__}
        print(key, out[key]["http_status"], flush=True)
    write_result(REPO_ROOT / "results" / "data" / "public_reachability.json", out, script=__file__, data_kind="public",
                 notes="GET with Range bytes=0-0; 206/200 means reachable, 301/302 are redirects not followed, 403/407 are blocked by the proxy or site, 429 is a rate or quota limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
