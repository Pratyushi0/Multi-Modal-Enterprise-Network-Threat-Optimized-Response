import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [STREAM-PREP] - %(levelname)s - %(message)s')

def stream_and_aggregate_zip(zip_path, output_path):
    logging.info(f"Opening secure streaming handle directly into {zip_path}...")
    
    logon_chunks = []
    device_chunks = []
    
    # Open the zip archive as a readable system stream
    with zipfile.ZipFile(zip_path, 'r') as z:
        
        # ---------------------------------------------------------------------
        # PART 1: STREAMING LOGON PROCESSING
        # ---------------------------------------------------------------------
        logging.info("Streaming compressed Logon binaries straight to memory...")
        try:
            # Open a direct text stream to the file inside the zip archive
            with z.open("r4.2/logon.csv") as f:
                for chunk in pd.read_csv(f, chunksize=150000):
                    chunk['date'] = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    chunk = chunk.dropna(subset=['date'])
                    
                    chunk['hour'] = chunk['date'].dt.hour
                    chunk['day'] = chunk['date'].dt.date
                    chunk['is_after_hours'] = ((chunk['hour'] >= 18) | (chunk['hour'] <= 6)).astype(int)
                    
                    agg_chunk = chunk.groupby(['day', 'user']).agg(
                        total_logins=('activity', lambda x: (x == 'Logon').sum()),
                        after_hours_logins=('is_after_hours', 'sum')
                    ).reset_index()
                    logon_chunks.append(agg_chunk)
                    
            logon_daily = pd.concat(logon_chunks).groupby(['day', 'user']).sum().reset_index()
            logging.info(f"Logon matrix successfully summarized in-memory: {len(logon_daily)} rows.")
        except Exception as e:
            logging.critical(f"Logon stream failed: {e}")
            return

        # ---------------------------------------------------------------------
        # PART 2: STREAMING DEVICE PROCESSING
        # ---------------------------------------------------------------------
        logging.info("Streaming compressed Device/USB binaries straight to memory...")
        try:
            with z.open("r4.2/device.csv") as f:
                for chunk in pd.read_csv(f, chunksize=150000):
                    chunk['date'] = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
                    chunk = chunk.dropna(subset=['date'])
                    chunk['day'] = chunk['date'].dt.date
                    
                    agg_chunk = chunk.groupby(['day', 'user']).agg(
                        usb_file_copies=('activity', lambda x: (x == 'Connect').sum())
                    ).reset_index()
                    device_chunks.append(agg_chunk)
                    
            device_daily = pd.concat(device_chunks).groupby(['day', 'user']).sum().reset_index()
            logging.info(f"Device matrix successfully summarized in-memory: {len(device_daily)} rows.")
        except Exception as e:
            logging.critical(f"Device stream failed: {e}")
            return

    # ---------------------------------------------------------------------
    # PART 3: VECTOR MATRIX COALESCE
    # ---------------------------------------------------------------------
    logging.info("Coalescing multi-modal matrices into final analytical tensor...")
    merged_matrix = pd.merge(logon_daily, device_daily, on=['day', 'user'], how='outer').fillna(0)
    
    count_cols = ['total_logins', 'after_hours_logins', 'usb_file_copies']
    merged_matrix[count_cols] = merged_matrix[count_cols].astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_matrix.to_csv(output_path, index=False)
    logging.info(f"Pillar-1 matrix securely preserved to disk at: {output_path}")

if __name__ == "__main__":
    # Point directly to the untouched zip file path
    ZIP_ARCHIVE = "data/raw/dataset.zip"
    PROCESSED_OUTPUT = "data/processed/cert_processed_matrix.csv"
    
    if os.path.exists(ZIP_ARCHIVE):
        stream_and_aggregate_zip(ZIP_ARCHIVE, PROCESSED_OUTPUT)
    else:
        logging.critical(f"Could not find dataset.zip at {ZIP_ARCHIVE}")