#!/usr/bin/env python3
"""
scripts/verify_pipeline_outputs.py
Xác nhận các file output của pipeline tồn tại trên MinIO/S3
Dùng trong CI integration test để verify kết quả
"""

import sys
import os
import argparse
import boto3
from botocore.exceptions import ClientError


def verify_keys(bucket: str, expected_keys: list[str]) -> bool:
    """
    Kiểm tra các keys có tồn tại trên MinIO
    Returns True nếu tất cả keys tồn tại, False nếu thiếu
    """
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD", "minioadmin123"),
        region_name="us-east-1"
    )

    all_ok = True
    print(f"\nVerifying {len(expected_keys)} keys in bucket '{bucket}':")
    print("─" * 60)

    for key in expected_keys:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"  ✅ {key}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"  ❌ MISSING: {key}")
                all_ok = False
            else:
                print(f"  ⚠️  ERROR: {key} — {e}")
                all_ok = False

    print("─" * 60)
    if all_ok:
        print("✅ Tất cả outputs verified thành công!\n")
    else:
        print("❌ Một số outputs bị thiếu!\n")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",        required=True)
    parser.add_argument("--expected-keys", nargs="+", required=True)
    args = parser.parse_args()

    ok = verify_keys(args.bucket, args.expected_keys)
    sys.exit(0 if ok else 1)
