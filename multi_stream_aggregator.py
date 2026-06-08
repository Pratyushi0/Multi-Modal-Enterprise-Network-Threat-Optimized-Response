import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MULTI-STREAM] - %(levelname)s - %(message)s')


def run_multi_domain_pipeline(zip_path="data/raw/dataset.zip",
                               labeled_matrix_path="data/processed/cert_labeled_matrix.csv",
                               final_output_path="data/processed/cert_expanded_matrix.csv"):

    if not os.path.exists(zip_path) or not os.path.exists(labeled_matrix_path):
        logging.error("Prerequisites missing for multi-stream aggregation. Skipping step.")
        return False

    logging.info(f"Opening core data matrix from {labeled_matrix_path}...")
    base_df = pd.read_csv(labeled_matrix_path)
    base_df['user'] = base_df['user'].astype(str).str.strip()
    base_df['day'] = base_df['day'].astype(str)

    # FIX: Drop any pre-existing web_clicks or file_touches columns to prevent
    # duplicate _x/_y columns on re-runs of this script
    cols_to_drop = [c for c in base_df.columns if c in ['web_clicks', 'file_touches']]
    if cols_to_drop:
        logging.warning(f"Dropping pre-existing columns to prevent merge duplication: {cols_to_drop}")
        base_df = base_df.drop(columns=cols_to_drop)

    # Verify is_attacker survived the load
    assert 'is_attacker' in base_df.columns, "CRITICAL: is_attacker column missing from labeled matrix!"
    logging.info(f"Label integrity check: {int(base_df['is_attacker'].sum())} attack rows loaded.")

    # --- STREAM 1: HTTP PROXY TELEMETRY ---
    web_chunks = []
    logging.info(f"Streaming compressed HTTP logs directly from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open("r4.2/http.csv") as f:
                for chunk in pd.read_csv(f, chunksize=250000, usecols=['date', 'user']):
                    chunk['date'] = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    chunk = chunk.dropna(subset=['date'])
                    chunk['day'] = chunk['date'].dt.date.astype(str)
                    chunk['user'] = chunk['user'].astype(str).str.strip()
                    agg_chunk = chunk.groupby(['day', 'user']).size().reset_index(name='web_clicks')
                    web_chunks.append(agg_chunk)

        logging.info("Consolidating memory-buffered web chunks...")
        web_daily = pd.concat(web_chunks).groupby(['day', 'user'])['web_clicks'].sum().reset_index()
        web_daily['day'] = web_daily['day'].astype(str)

    except Exception as e:
        logging.critical(f"HTTP streaming engine collapsed: {str(e)}")
        return False

    # --- STREAM 2: FILE INTERACTION TELEMETRY ---
    file_chunks = []
    logging.info(f"Streaming compressed FILE logs directly from {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open("r4.2/file.csv") as f:
                for chunk in pd.read_csv(f, chunksize=250000, usecols=['date', 'user']):
                    chunk['date'] = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    chunk = chunk.dropna(subset=['date'])
                    chunk['day'] = chunk['date'].dt.date.astype(str)
                    chunk['user'] = chunk['user'].astype(str).str.strip()
                    agg_chunk = chunk.groupby(['day', 'user']).size().reset_index(name='file_touches')
                    file_chunks.append(agg_chunk)

        logging.info("Consolidating memory-buffered file chunks...")
        file_daily = pd.concat(file_chunks).groupby(['day', 'user'])['file_touches'].sum().reset_index()
        file_daily['day'] = file_daily['day'].astype(str)

    except Exception as e:
        logging.critical(f"File streaming engine collapsed: {str(e)}")
        return False

    # --- UNIFIED MERGE ---
    logging.info("Merging multi-modal telemetry vectors onto master analytical tensor...")
    merged_df = pd.merge(base_df, web_daily, on=['day', 'user'], how='left').fillna(0)
    merged_df = pd.merge(merged_df, file_daily, on=['day', 'user'], how='left').fillna(0)

    merged_df['web_clicks'] = merged_df['web_clicks'].astype(int)
    merged_df['file_touches'] = merged_df['file_touches'].astype(int)

    # FIX: Final label integrity assertion before writing to disk
    assert 'is_attacker' in merged_df.columns, "CRITICAL: is_attacker column lost during merge!"
    logging.info(f"Final label integrity check: {int(merged_df['is_attacker'].sum())} attack rows in expanded matrix.")

    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    merged_df.to_csv(final_output_path, index=False)
    logging.info(f"Successfully compiled multi-domain matrix: {final_output_path}")
    return True


if __name__ == "__main__":
    run_multi_domain_pipeline()
