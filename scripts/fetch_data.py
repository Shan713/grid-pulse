"""Download aligned Electricity Maps hourly data for the Grid Pulse study.

The API token must be supplied through ELECTRICITYMAPS_API_KEY.  The script
uses short windows because the hourly past-range endpoint limits request size.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


API_ROOT = "https://api.electricitymaps.com/v4"
START = datetime(2017, 1, 1, tzinfo=timezone.utc)
END_EXCLUSIVE = datetime(2026, 9, 2, 16, tzinfo=timezone.utc)
CHUNK_DAYS = 9


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _windows() -> list[tuple[datetime, datetime]]:
    result = []
    current = START
    while current < END_EXCLUSIVE:
        following = min(current + timedelta(days=CHUNK_DAYS), END_EXCLUSIVE)
        result.append((current, following))
        current = following
    return result


def _fetch_window(
    token: str,
    zone: str,
    endpoint: str,
    start: datetime,
    end: datetime,
    retries: int = 6,
) -> list[dict]:
    params = {
        "zone": zone,
        "start": _iso(start),
        "end": _iso(end),
        "temporalGranularity": "hourly",
    }
    delay = 1.0
    last_error = "unknown error"
    for _ in range(retries):
        try:
            response = requests.get(
                f"{API_ROOT}/{endpoint}/past-range",
                params=params,
                headers={"auth-token": token},
                timeout=45,
            )
            if response.status_code == 200:
                return response.json().get("data", [])
            last_error = f"HTTP {response.status_code}: {response.text[:160]}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(delay)
        delay = min(delay * 2, 20)
    raise RuntimeError(f"{zone} {endpoint} {start}->{end}: {last_error}")


def fetch_signal(token: str, zone: str, endpoint: str, workers: int) -> pd.DataFrame:
    windows = _windows()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_window, token, zone, endpoint, start, end): (start, end)
            for start, end in windows
        }
        for number, future in enumerate(as_completed(futures), 1):
            records.extend(future.result())
            if number % 75 == 0 or number == len(windows):
                print(f"{zone} {endpoint}: {number}/{len(windows)} windows")

    frame = pd.DataFrame(records)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.drop_duplicates("datetime").sort_values("datetime")
    expected = pd.date_range(START, END_EXCLUSIVE, freq="h", inclusive="left")
    observed = pd.DatetimeIndex(frame["datetime"])
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(frame) != len(expected) or len(missing) or len(extra):
        raise RuntimeError(
            f"{zone} {endpoint} is not complete: rows={len(frame)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", nargs="+", default=["IN-WE", "IN-SO"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    token = os.getenv("ELECTRICITYMAPS_API_KEY")
    if not token:
        raise SystemExit("Set ELECTRICITYMAPS_API_KEY before running this script.")

    for zone in args.zones:
        folder = Path(zone.replace("-", "_"))
        folder.mkdir(parents=True, exist_ok=True)
        outputs = {
            "carbon-intensity": folder / f"{zone}_carbon_intensity.csv",
            "renewable-energy": folder / f"{zone}_renewable_percentage.csv",
        }
        for endpoint, output in outputs.items():
            if output.exists() and not args.overwrite:
                print(f"Keeping existing {output}; use --overwrite to replace it")
                continue
            data = fetch_signal(token, zone, endpoint, args.workers)
            data.to_csv(output, index=False)
            print(f"Saved {len(data):,} rows to {output}")


if __name__ == "__main__":
    main()
