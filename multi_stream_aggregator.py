"""
multi_stream_aggregator.py
--------------------------
Streams HTTP proxy and file-access logs from the raw zip archive,
aggregates them to daily per-user counts, then merges them onto the
labeled base matrix to produce the full expanded feature matrix used
by the ML engine.

Input:
    data/processed/cert_labeled_matrix.csv   (from label_injector.py)
    data/raw/dataset.zip                     (contains r4.2/http.csv, r4.2/file.csv)

Output:
    data/processed/cert_expanded_matrix.csv
"""

import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [MULTI-STREAM] - %(levelname)s - %(message)s",
)

HTTP_FILE   = "r4.2/http.csv"
FILE_FILE   = "r4.2/file.csv"
CHUNK_SIZE  = 250_000
DATE_FMT    = "%m/%d/%Y %H:%M:%S"


def _stream_http(z: zipfile.ZipFile) -> pd.DataFrame:
    """Aggregate HTTP proxy log into daily web_clicks per user."""
    chunks = []
    logging.info(f"Streaming HTTP proxy logs from archive …")
    with z.open(HTTP_FILE) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, usecols=["date", "user"]):
            chunk["date"] = pd.to_datetime(chunk["date"], format=DATE_FMT, errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk["day"]  = chunk["date"].dt.date.astype(str)
            chunk["user"] = chunk["user"].astype(str).str.strip()
            agg = chunk.groupby(["day", "user"]).size().reset_index(name="web_clicks")
            chunks.append(agg)

    logging.info("Consolidating HTTP chunks …")
    daily = pd.concat(chunks).groupby(["day", "user"])["web_clicks"].sum().reset_index()
    return daily


def _stream_file(z: zipfile.ZipFile) -> pd.DataFrame:
    """Aggregate file-access log into daily file_touches per user."""
    chunks = []
    logging.info("Streaming file-access logs from archive …")
    with z.open(FILE_FILE) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, usecols=["date", "user"]):
            chunk["date"] = pd.to_datetime(chunk["date"], format=DATE_FMT, errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk["day"]  = chunk["date"].dt.date.astype(str)
            chunk["user"] = chunk["user"].astype(str).str.strip()
            agg = chunk.groupby(["day", "user"]).size().reset_index(name="file_touches")
            chunks.append(agg)

    logging.info("Consolidating file-access chunks …")
    daily = pd.concat(chunks).groupby(["day", "user"])["file_touches"].sum().reset_index()
    return daily


def run_multi_domain_pipeline(
    zip_path: str           = "data/raw/dataset.zip",
    labeled_matrix_path: str = "data/processed/cert_labeled_matrix.csv",
    final_output_path: str  = "data/processed/cert_expanded_matrix.csv",
) -> bool:
    """
    Loads the labeled base matrix, streams HTTP + file telemetry from the
    archive, merges everything, and writes the expanded matrix.

    Returns True on success, False on failure.
    """
    if not os.path.exists(zip_path):
        logging.error(f"Archive not found: {zip_path}")
        return False
    if not os.path.exists(labeled_matrix_path):
        logging.error(f"Labeled matrix not found: {labeled_matrix_path}")
        return False

    # ------------------------------------------------------------------
    # 1. Load labeled base matrix
    # ------------------------------------------------------------------
    logging.info(f"Loading labeled base matrix: {labeled_matrix_path}")
    base_df = pd.read_csv(labeled_matrix_path)
    base_df["user"] = base_df["user"].astype(str).str.strip()
    base_df["day"]  = base_df["day"].astype(str)

    # Drop any stale web/file columns to prevent _x/_y duplicates on re-runs
    stale = [c for c in base_df.columns if c in ("web_clicks", "file_touches")]
    if stale:
        logging.warning(f"Dropping stale columns to prevent merge duplication: {stale}")
        base_df = base_df.drop(columns=stale)

    assert "is_attacker" in base_df.columns, (
        "CRITICAL: is_attacker missing from labeled matrix!"
    )
    logging.info(
        f"Label check: {int(base_df['is_attacker'].sum())} attack rows loaded."
    )

    # ------------------------------------------------------------------
    # 2. Stream HTTP + file telemetry
    # ------------------------------------------------------------------
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            web_daily  = _stream_http(z)
            file_daily = _stream_file(z)
    except Exception as exc:
        logging.critical(f"Telemetry streaming failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # 3. Merge onto base matrix
    # ------------------------------------------------------------------
    logging.info("Merging multi-modal telemetry onto base matrix …")
    merged = (
        base_df
        .merge(web_daily,  on=["day", "user"], how="left")
        .merge(file_daily, on=["day", "user"], how="left")
        .fillna(0)
    )
    merged["web_clicks"]   = merged["web_clicks"].astype(int)
    merged["file_touches"] = merged["file_touches"].astype(int)

    assert "is_attacker" in merged.columns, (
        "CRITICAL: is_attacker lost during merge!"
    )
    logging.info(
        f"Final label check: {int(merged['is_attacker'].sum())} attack rows in expanded matrix."
    )

    # ------------------------------------------------------------------
    # 4. Write expanded matrix
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    merged.to_csv(final_output_path, index=False)
    logging.info(f"Expanded matrix written: {final_output_path}  ({len(merged):,} rows)")
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ok = run_multi_domain_pipeline()
    if not ok:
        raise SystemExit(1)
