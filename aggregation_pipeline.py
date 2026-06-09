"""
aggregation_pipeline.py
-----------------------
Stream-processes logon and device/USB logs directly from the raw zip archive
and writes a per-user-per-day feature matrix to disk.

Output columns:
    day, user, total_logins, after_hours_logins, usb_file_copies
"""

import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [LOGON-DEVICE-AGG] - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGON_FILE  = "r4.2/logon.csv"
DEVICE_FILE = "r4.2/device.csv"
CHUNK_SIZE  = 150_000
DATE_FMT    = "%m/%d/%Y %H:%M:%S"
AFTER_HOURS = lambda h: (h >= 18) | (h <= 6)   # noqa: E731


def _stream_logon(z: zipfile.ZipFile) -> pd.DataFrame:
    """Read logon.csv in chunks, return daily per-user aggregates."""
    chunks = []
    logging.info("Streaming logon logs …")
    with z.open(LOGON_FILE) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE):
            chunk["date"] = pd.to_datetime(chunk["date"], format=DATE_FMT, errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk["hour"] = chunk["date"].dt.hour
            chunk["day"]  = chunk["date"].dt.date
            chunk["user"] = chunk["user"].astype(str).str.strip()
            chunk["is_after_hours"] = AFTER_HOURS(chunk["hour"]).astype(int)
            agg = chunk.groupby(["day", "user"]).agg(
                total_logins     =("activity",       lambda x: (x == "Logon").sum()),
                after_hours_logins=("is_after_hours", "sum"),
            ).reset_index()
            chunks.append(agg)

    daily = pd.concat(chunks).groupby(["day", "user"]).sum().reset_index()
    logging.info(f"Logon matrix: {len(daily):,} user-day rows.")
    return daily


def _stream_device(z: zipfile.ZipFile) -> pd.DataFrame:
    """Read device.csv in chunks, return daily USB connect counts per user."""
    chunks = []
    logging.info("Streaming device / USB logs …")
    with z.open(DEVICE_FILE) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE):
            chunk["date"] = pd.to_datetime(chunk["date"], format=DATE_FMT, errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk["day"]  = chunk["date"].dt.date
            chunk["user"] = chunk["user"].astype(str).str.strip()
            agg = chunk.groupby(["day", "user"]).agg(
                usb_file_copies=("activity", lambda x: (x == "Connect").sum())
            ).reset_index()
            chunks.append(agg)

    daily = pd.concat(chunks).groupby(["day", "user"]).sum().reset_index()
    logging.info(f"Device matrix: {len(daily):,} user-day rows.")
    return daily


def stream_and_aggregate_zip(
    zip_path: str  = "data/raw/dataset.zip",
    output_path: str = "data/processed/cert_processed_matrix.csv",
) -> bool:
    """
    Entry point.  Streams logon + device CSVs from zip_path and writes
    a merged daily feature matrix to output_path.

    Returns True on success, False on failure.
    """
    if not os.path.exists(zip_path):
        logging.critical(f"Archive not found: {zip_path}")
        return False

    logging.info(f"Opening archive: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            logon_daily  = _stream_logon(z)
            device_daily = _stream_device(z)
    except Exception as exc:
        logging.critical(f"Streaming failed: {exc}")
        return False

    logging.info("Merging logon + device into base feature matrix …")
    merged = (
        pd.merge(logon_daily, device_daily, on=["day", "user"], how="outer")
        .fillna(0)
    )
    count_cols = ["total_logins", "after_hours_logins", "usb_file_copies"]
    merged[count_cols] = merged[count_cols].astype(int)
    merged["day"] = merged["day"].astype(str)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    logging.info(f"Base feature matrix written: {output_path}  ({len(merged):,} rows)")
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_ARCHIVE      = "data/raw/dataset.zip"
    PROCESSED_OUTPUT = "data/processed/cert_processed_matrix.csv"

    ok = stream_and_aggregate_zip(ZIP_ARCHIVE, PROCESSED_OUTPUT)
    if not ok:
        raise SystemExit(1)
