#!/usr/bin/env sh
set -eux

LOCAL=/data/processed
REMOTE="chi_tacc:object-persist-project39/processed"

echo "▶ Local processed files:"
ls -lh "${LOCAL}"

echo "▶ Uploading only new files to ${REMOTE} …"
rclone copy --ignore-existing --checksum --progress "${LOCAL}" "${REMOTE}"

echo "▶ Remote now contains:"
rclone ls "${REMOTE}"
