# --- AWS ---
AWS_PROFILE=pkmscan-processor
AWS_REGION_NAME=eu-west-3
AWS_STORAGE_BUCKET=pkmscan-storage
AWS_RAW_IMG_QUEUE_URL=https://sqs.eu-west-3.amazonaws.com/xxx/queue

# --- YOLO ---
YOLO_SEG_MODEL_S3KEY=models/latest_seg.pt
YOLO_SEG_MODEL_LOCALPATH_=/tmp/seg.pt