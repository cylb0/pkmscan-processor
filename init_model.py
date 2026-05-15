from aws_shared.aws_clients import aws_client
import os
import sys


def download_model():
    s3_key = os.getenv("YOLO_SEG_MODEL_S3KEY")
    local_path = os.getenv("YOLO_SEG_MODEL_LOCALPATH")

    if not s3_key or not local_path:
        print(
            "Error: Missing YOLO_S3_SEG_MODEL_KEY or YOLO_SEG_MODEL_LOCAL_PATH environment variables"
        )
        sys.exit(1)

    print("Downloading model from s3...")
    try:
        aws_client.download_file(s3_key, local_path)
        print("Model downloaded successfully to {local_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
