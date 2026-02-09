import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os
import glob

# --- CONFIG ---
ROLE = "arn:aws:iam::760347630952:role/service-role/AmazonSageMaker-ExecutionRole-20250912T151676" 
BUCKET = "sagemaker-us-east-1-760347630952"
PREFIX = "scalping-system"
LOCAL_DATA = "full_data_cache.parquet"

def upload_data():
    sess = sagemaker.Session()
    print(f"Uploading {LOCAL_DATA} to s3://{BUCKET}/{PREFIX}/data...")
    upload_path = sess.upload_data(path=LOCAL_DATA, bucket=BUCKET, key_prefix=f"{PREFIX}/data")
    print(f"Data uploaded to: {upload_path}")
    return upload_path

def main():
    print("Initializing SageMaker Session...")
    try:
        sess = sagemaker.Session()
        role = ROLE
    except Exception as e:
        print(f"Error initializing session: {e}")
        print("Please ensure credentials are set.")
        return

    # Upload Data
    try:
        s3_data = upload_data()
    except Exception as e:
        print(f"Upload failed: {e}")
        # Fallback manual command
        print("\n--- MANUAL UPLOAD COMMAND ---")
        print(f"aws s3 sync {LOCAL_DATA} s3://{BUCKET}/{PREFIX}/data")
        return

    print("Defining Estimator...")
    sklearn_estimator = SKLearn(
        entry_point='train_rf_sagemaker_entry.py',
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='1.2-1',
        py_version='py3',
        dependencies=['requirements.txt'], 
    )
    
    print("Launching Job...")
    sklearn_estimator.fit({'train': s3_data})
    
    print("\nJob launched!")
    print(f"Model will be saved to: {sklearn_estimator.model_data}")

if __name__ == "__main__":
    main()
