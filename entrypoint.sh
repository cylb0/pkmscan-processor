#!/bin/bash
set -e

mkdir -p "/tmp"

python /app/init_model.py

exec "$@"
