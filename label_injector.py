import pandas as pd
import logging
import zipfile
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EXACT-LABELER] - %(levelname)s - %(message)s')

def inject_precise_time_labels(zip_path, processed_matrix_path, final_output_path):
    logging.info(f"Loading feature matrix from {processed_matrix_path}...")
    df = pd.read_csv(processed_matrix_path)
    
    # Force clean column name and enforce clean strings
    df['user'] = df['user'].astype(str).str.strip()
    df['day_dt'] = pd.to_datetime(df['day']).dt.date
    
    # Reset target labels completely to baseline innocent (0)
    df['is_attacker'] = 0
    
    logging.info(f"Streaming threat answers file from archive: {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open("answers/insiders.csv") as f:
                insiders_df = pd.read_csv(f)
        
        insiders_df['user'] = insiders_df['user'].astype(str).str.strip()
        
        logging.info(f"Answer file structural fields: {list(insiders_df.columns)}")
        logging.info("Executing error-coerced timeline range extraction...")
        
        # Coerce malformed date strings to NaT (Not a Time) instead of crashing the script
        insiders_df['start_dt'] = pd.to_datetime(insiders_df['start'], errors='coerce').dt.date
        insiders_df['end_dt'] = pd.to_datetime(insiders_df['end'], errors='coerce').dt.date
        
        # Drop any insider metadata rows where start/end date parsing completely failed
        insiders_df = insiders_df.dropna(subset=['start_dt', 'end_dt'])
        logging.info(f"Successfully cleaned and retained {len(insiders_df)} ground-truth timeline maps.")
        
        # ---------------------------------------------------------------------
        # VECTORIZED DATE RANGE OVERLAY
        # ---------------------------------------------------------------------
        # Iterate over verified attack windows and flip target matrix flag to 1
        for _, row in insiders_df.iterrows():
            user_mask = df['user'] == row['user']
            date_mask = (df['day_dt'] >= row['start_dt']) & (df['day_dt'] <= row['end_dt'])
            df.loc[user_mask & date_mask, 'is_attacker'] = 1
            
        # Clean up temporary date-processing indices before disk preservation
        if 'day_dt' in df.columns:
            df = df.drop(columns=['day_dt'])
            
        total_rows = len(df)
        total_attacks = df['is_attacker'].sum()
        
        print("\n" + "="*60)
        print("          PRECISION TIME-ALIGNMENT COMPLETE")
        print("="*60)
        print(f" -> Total Matrix Volume:      {total_rows} rows")
        print(f" -> True Attack Vector Days:  {total_attacks} rows")
        print(f" -> Controlled Normal Days:   {total_rows - total_attacks} rows")
        print("="*60 + "\n")
        
        df.to_csv(final_output_path, index=False)
        logging.info(f"Precisely labeled target tensor written to disk: {final_output_path}")
        
    except Exception as e:
        logging.critical(f"Label processing pipeline failed: {str(e)}")

if __name__ == "__main__":
    ZIP_ARCHIVE = "data/raw/dataset.zip"
    PROCESSED_MATRIX = "data/processed/cert_processed_matrix.csv"
    LABELED_OUTPUT = "data/processed/cert_labeled_matrix.csv"
    
    inject_precise_time_labels(ZIP_ARCHIVE, PROCESSED_MATRIX, LABELED_OUTPUT)