"""
label_injector.py
-----------------
Aligns ground-truth insider threat windows from answers/insiders.csv
(inside the raw zip) onto the processed feature matrix, writing
a labeled matrix with an `is_attacker` column.

Requires:
    data/processed/cert_processed_matrix.csv   (from aggregation_pipeline.py)
    data/raw/dataset.zip                       (contains answers/insiders.csv)

Output:
    data/processed/cert_labeled_matrix.csv
"""

import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EXACT-LABELER] - %(levelname)s - %(message)s",
)

INSIDERS_FILE = "answers/insiders.csv"


def inject_precise_time_labels(
    zip_path: str             = "data/raw/dataset.zip",
    processed_matrix_path: str = "data/processed/cert_processed_matrix.csv",
    final_output_path: str    = "data/processed/cert_labeled_matrix.csv",
) -> bool:
    """
    Loads the feature matrix, resets all labels to 0, then flips
    is_attacker = 1 for every user-day that falls inside a confirmed
    insider threat window.

    Returns True on success, False on failure.
    """
    # ------------------------------------------------------------------
    # 1. Load feature matrix
    # ------------------------------------------------------------------
    if not os.path.exists(processed_matrix_path):
        logging.critical(f"Feature matrix not found: {processed_matrix_path}")
        return False

    logging.info(f"Loading feature matrix: {processed_matrix_path}")
    df = pd.read_csv(processed_matrix_path)
    df["user"]   = df["user"].astype(str).str.strip()
    df["day_dt"] = pd.to_datetime(df["day"], errors="coerce").dt.date

    bad_days = df["day_dt"].isna().sum()
    if bad_days:
        logging.warning(f"Dropping {bad_days} rows with unparseable day values.")
        df = df.dropna(subset=["day_dt"])

    # Baseline — everyone is innocent
    df["is_attacker"] = 0

    # ------------------------------------------------------------------
    # 2. Load ground-truth windows
    # ------------------------------------------------------------------
    if not os.path.exists(zip_path):
        logging.critical(f"Archive not found: {zip_path}")
        return False

    logging.info(f"Loading insider ground-truth windows from {zip_path} …")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open(INSIDERS_FILE) as f:
                insiders = pd.read_csv(f)
    except Exception as exc:
        logging.critical(f"Failed to read insider answer file: {exc}")
        return False

    insiders["user"]     = insiders["user"].astype(str).str.strip()
    logging.info(f"Answer file columns: {list(insiders.columns)}")

    insiders["start_dt"] = pd.to_datetime(insiders["start"], errors="coerce").dt.date
    insiders["end_dt"]   = pd.to_datetime(insiders["end"],   errors="coerce").dt.date

    before = len(insiders)
    insiders = insiders.dropna(subset=["start_dt", "end_dt"])
    dropped  = before - len(insiders)
    if dropped:
        logging.warning(f"Dropped {dropped} insider records with unparseable date ranges.")
    logging.info(f"Retained {len(insiders)} verified attack windows.")

    # ------------------------------------------------------------------
    # 3. Flip labels inside confirmed attack windows
    # ------------------------------------------------------------------
    for _, row in insiders.iterrows():
        mask = (
            (df["user"]   == row["user"]) &
            (df["day_dt"] >= row["start_dt"]) &
            (df["day_dt"] <= row["end_dt"])
        )
        df.loc[mask, "is_attacker"] = 1

    # ------------------------------------------------------------------
    # 4. Clean up and write
    # ------------------------------------------------------------------
    df = df.drop(columns=["day_dt"])

    total_rows   = len(df)
    total_attack = int(df["is_attacker"].sum())
    total_normal = total_rows - total_attack
    contam_rate  = total_attack / total_rows

    print("\n" + "=" * 60)
    print("          PRECISION TIME-ALIGNMENT COMPLETE")
    print("=" * 60)
    print(f" -> Total matrix volume:      {total_rows:,} rows")
    print(f" -> True attack vector days:  {total_attack:,} rows")
    print(f" -> Controlled normal days:   {total_normal:,} rows")
    print(f" -> True contamination rate:  {contam_rate:.5f} ({contam_rate * 100:.3f}%)")
    print("=" * 60 + "\n")

    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    df.to_csv(final_output_path, index=False)
    logging.info(f"Labeled matrix written: {final_output_path}")
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ZIP_ARCHIVE      = "data/raw/dataset.zip"
    PROCESSED_MATRIX = "data/processed/cert_processed_matrix.csv"
    LABELED_OUTPUT   = "data/processed/cert_labeled_matrix.csv"

    ok = inject_precise_time_labels(ZIP_ARCHIVE, PROCESSED_MATRIX, LABELED_OUTPUT)
    if not ok:
        raise SystemExit(1)
