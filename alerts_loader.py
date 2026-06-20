"""
alerts_loader.py

Dynamic loader + cleaner for the official Ukrainian air raid alert dataset
maintained at:
https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset

raw.githubusercontent.com always serves the latest committed version of a
file, so calling load_official_alerts() picks up that day's update with no
manual download step. A small local CSV cache avoids re-pulling the ~27MB
file on every run within the same session.

Known data quality issues in the upstream source (checked 2026-06-20):
  - ~42% of rows are exact duplicates -- history up to ~2026-01-25 appears
    twice, almost certainly from a one-time upstream merge bug. Handled by
    dropping exact duplicate rows.
  - A small number of alerts (~0.4%) run longer than 24h. Most are genuine:
    frontline raions (e.g. Nikopolskyi in Dnipropetrovska, several Donetsk
    oblast raions) can stay "under alert" for days/weeks under continuous
    shelling threat -- kept as-is, just flagged. Two specific records (604
    and 438 days) are implausible outliers, almost certainly unclosed/stale
    records from the same upstream glitch -- flagged separately and
    excluded from duration-based aggregates (the event itself is kept).
"""

from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd

RAW_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
)


def _download(url: str, dest: Path, timeout: int = 30, chunk_size: int = 1 << 20) -> None:
    """Stream-download url to dest with progress output and a hard timeout.

    `timeout` here is per-chunk (connect + read-until-next-chunk), not for
    the whole transfer -- so a slow-but-alive connection won't time out,
    only a genuinely stalled one will, and within `timeout` seconds rather
    than hanging indefinitely.
    """
    print(f"downloading {url} ...")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        written = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                print(f"  {written / 1e6:6.1f} MB downloaded", end="\r")
        tmp.replace(dest)
    print(f"\nsaved {written / 1e6:.1f} MB to {dest}")


def _fetch_raw(cache_path: Path, max_cache_age_hours: float, force_refresh: bool) -> pd.DataFrame:
    needs_refresh = force_refresh or not cache_path.exists()

    if not needs_refresh:
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        needs_refresh = age > timedelta(hours=max_cache_age_hours)

    if needs_refresh:
        _download(RAW_URL, cache_path)

    df = pd.read_csv(cache_path, parse_dates=["started_at", "finished_at"])

    return df


def clean_alerts(
    df: pd.DataFrame,
    long_duration_hours: float = 24.0,
    error_duration_days: float = 30.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Deduplicate and flag known data quality issues (see module docstring).
    Does not mutate the input; returns a new, sorted DataFrame with:
      - duration_min            raw duration in minutes
      - is_long_duration        > long_duration_hours (default 24h)
      - is_likely_data_error    > error_duration_days (default 30d)
      - duration_min_clean      duration_min, but NaN where is_likely_data_error
                                 (use this column for any sum/mean-of-duration
                                 aggregation so the two broken records can't
                                 inject weeks of phantom "alert time")
    """
    n_before = len(df)
    df = df.drop_duplicates().sort_values("started_at").reset_index(drop=True)
    n_dupes = n_before - len(df)

    df["duration_min"] = (df["finished_at"] -
                          df["started_at"]).dt.total_seconds() / 60
    df["is_long_duration"] = df["duration_min"] > long_duration_hours * 60
    df["is_likely_data_error"] = df["duration_min"] > error_duration_days * 1440
    df["duration_min_clean"] = df["duration_min"].where(
        ~df["is_likely_data_error"])

    if verbose:
        n_long = int(df["is_long_duration"].sum())
        n_err = int(df["is_likely_data_error"].sum())
        print(
            f"dedup: dropped {n_dupes:,}/{n_before:,} exact duplicate rows ({n_dupes / n_before:.1%})")
        print(f"flagged {n_long:,} long-duration alerts (>{long_duration_hours:.0f}h) "
              f"-- kept as-is, mostly frontline raions under continuous threat")
        print(f"flagged {n_err:,} likely data-error rows (>{error_duration_days:.0f}d) "
              f"-- duration excluded via duration_min_clean, event itself kept")

    return df


def load_official_alerts(
    cache_path: str = "data/official_data_en.csv",
    max_cache_age_hours: float = 6.0,
    force_refresh: bool = False,
    local_tz: str = "Europe/Kyiv",
    clean: bool = True,
) -> pd.DataFrame:
    """
    Load the official air raid alert dataset, refreshing from GitHub
    whenever the local cache is missing or older than max_cache_age_hours.

    Each row is one alert event (oblast / raion / hromada level) with a
    start and end timestamp, plus local Kyiv-time columns. By default also
    deduplicates and flags known data issues -- pass clean=False to get the
    raw table instead.
    """
    df = _fetch_raw(Path(cache_path), max_cache_age_hours, force_refresh)
    df = clean_alerts(df) if clean else df.sort_values(
        "started_at").reset_index(drop=True)

    df["started_at_local"] = df["started_at"].dt.tz_convert(local_tz)
    df["finished_at_local"] = df["finished_at"].dt.tz_convert(local_tz)

    return df


if __name__ == "__main__":
    alerts = load_official_alerts(force_refresh=True)
    print()
    print(f"{len(alerts):,} clean alert events | "
          f"{alerts['started_at'].min().date()} -> {alerts['started_at'].max().date()}")
    cols = ["oblast", "raion", "level", "started_at_local", "duration_min",
            "is_long_duration", "is_likely_data_error"]
    print(alerts[cols].tail())
