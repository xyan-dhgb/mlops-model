import os
import boto3
import s3fs
from dotenv import load_dotenv

def download_isic_data():
    # Load environment variables from a .env file if it exists
    load_dotenv(dotenv_path="load.env")

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN") # Often None if not using temporary credentials

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("Warning: AWS credentials not found. Please set them in your environment or a load.env file.")
    else:
        print("AWS credentials found.")

    S3_INPUT_BUCKET  = 'kltn-isic-2024-challenge'
    S3_INPUT_PREFIX  = 'isic-2024-challenge'
    
    LOCAL_METADATA = 'train-metadata.csv'
    LOCAL_HDF5     = 'train-image.hdf5'

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
    )

    print(f'Input S3 bucket : s3://{S3_INPUT_BUCKET}/{S3_INPUT_PREFIX}/')

    # Download Metadata CSV
    print('\\nĐang tải train-metadata.csv từ S3...')
    s3_client.download_file(
        S3_INPUT_BUCKET,
        f'{S3_INPUT_PREFIX}/train-metadata.csv',
        LOCAL_METADATA
    )
    print(f'   Đã lưu -> {LOCAL_METADATA}')

    # Download HDF5 Image Data
    print('\\nĐang tải train-image.hdf5 từ S3 (file lớn, có thể mất vài phút)...')
    s3_client.download_file(
        S3_INPUT_BUCKET,
        f'{S3_INPUT_PREFIX}/train-image.hdf5',
        LOCAL_HDF5
    )
    print(f'   Đã lưu -> {LOCAL_HDF5}')

if __name__ == "__main__":
    download_isic_data()