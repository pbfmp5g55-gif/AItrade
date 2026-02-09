import boto3
import pandas as pd
import os
import glob
from tqdm import tqdm
import argparse

BUCKET = "test-atena-glue-quicksight"
PREFIX = "ticks/symbol=USDJPY"
LOCAL_DIR = "full_data_2023_2026"
OUTPUT_FILE = "full_data_consolidated_2023_2026.parquet"

def download_data(years):
    s3 = boto3.client('s3')
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)
        
    for year in years:
        print(f"Checking S3 for year={year}...")
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET, Prefix=f"{PREFIX}/year={year}/")
        
        for page in pages:
            if 'Contents' not in page: continue
            for obj in tqdm(page['Contents'], desc=f"Downloading {year}"):
                key = obj['Key']
                if not key.endswith('.parquet'): continue
                
                filename = os.path.basename(key)
                local_path = os.path.join(LOCAL_DIR, filename)
                
                if not os.path.exists(local_path):
                    s3.download_file(BUCKET, key, local_path)

def consolidate():
    files = sorted(glob.glob(os.path.join(LOCAL_DIR, "*.parquet")))
    if not files:
        print("No files found!")
        return
        
    print(f"Consolidating {len(files)} files...")
    
    # Peek at columns
    sample = pd.read_parquet(files[0])
    print(f"Sample columns: {list(sample.columns)}")
    print(f"Sample index type: {type(sample.index)}")
    
    dfs = []
    for f in tqdm(files):
        df = pd.read_parquet(f)
        
        # Robust Time Detection
        ts_col = None
        for col in ['timestamp', 'time', 'datetime', 'DateTime', 'date']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try converting existing index
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
                
        # Ensure UTC/Local consistency safely
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        
        # We only need bid/ask
        dfs.append(df[['ask', 'bid']] if ('ask' in df.columns and 'bid' in df.columns) else df[['ask']])
        
    print(f"Concatenating {len(dfs)} dataframes...")
    full_df = pd.concat(dfs).sort_index()
    if full_df.index.duplicated().any():
        print("Dropping duplicates...")
        full_df = full_df[~full_df.index.duplicated(keep='first')]
    
    print(f"Final Data Range: {full_df.index.min()} to {full_df.index.max()}")
    print(f"Saving to {OUTPUT_FILE} ({len(full_df)} rows)...")
    full_df.to_parquet(OUTPUT_FILE, index=True)
    print("Consolidation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', default=['2023', '2024', '2025', '2026'])
    args = parser.parse_args()
    
    download_data(args.years)
    consolidate()
