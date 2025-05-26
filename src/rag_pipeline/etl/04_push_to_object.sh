#!/usr/bin/env bash
# (if you prefer a shell wrapper instead of inline in Compose)
rclone copy /mnt/block/faiss_base \
  $RCLONE_REMOTE:$RCLONE_CONTAINER/faiss_base --progress