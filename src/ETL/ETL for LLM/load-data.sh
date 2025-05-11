#!/usr/bin/env sh
set -e # Exit immediately if a command exits with a non-zero status.
# set -eux # Uncomment for very verbose debugging (shows every command executed)

echo "--- load-data.sh script starting ---"

LOCAL_SOURCE_DIR="/data/processed"
# RCLONE_CONTAINER is an environment variable, e.g., object-persist-project39
REMOTE_BASE_DEST="chi_tacc:${RCLONE_CONTAINER}"
REMOTE_FULL_DEST="${REMOTE_BASE_DEST}/processed"

echo "Local source directory: ${LOCAL_SOURCE_DIR}"
echo "Remote destination: ${REMOTE_FULL_DEST}"
echo

echo "--- Contents of local source directory (${LOCAL_SOURCE_DIR}) ---"
if [ -d "${LOCAL_SOURCE_DIR}" ]; then
  ls -lR "${LOCAL_SOURCE_DIR}"
else
  echo "Error: Local source directory ${LOCAL_SOURCE_DIR} does not exist."
  exit 1
fi
echo

# Verify the specific file we expect to upload
EXPECTED_FILE="${LOCAL_SOURCE_DIR}/all_meetings.jsonl"
if [ -f "${EXPECTED_FILE}" ]; then
  echo "Local file ${EXPECTED_FILE} found. Size: $(stat -c %s "${EXPECTED_FILE}") bytes."
else
  echo "Warning: Expected local file ${EXPECTED_FILE} not found in ${LOCAL_SOURCE_DIR}."
fi
echo

echo "--- Listing remote destination BEFORE copy (${REMOTE_FULL_DEST}/) ---"
# Use rclone ls to see files. If the directory doesn't exist, this might show an error or nothing.
rclone ls "${REMOTE_FULL_DEST}/" || echo "Note: Remote path ${REMOTE_FULL_DEST}/ may not exist yet or is empty."
echo

echo "Attempting to upload from ${LOCAL_SOURCE_DIR} to ${REMOTE_FULL_DEST} ..."
# -vv for very verbose output from rclone
# --checksum forces rclone to check files based on their checksums (more reliable than modtime)
# --progress shows overall progress
rclone copy -vv --checksum --progress \
  "${LOCAL_SOURCE_DIR}" \
  "${REMOTE_FULL_DEST}"
echo

echo "--- Listing remote destination AFTER copy (${REMOTE_FULL_DEST}/) ---"
rclone ls "${REMOTE_FULL_DEST}/" || echo "Note: Remote path ${REMOTE_FULL_DEST}/ may not exist or is empty after copy attempt."
echo

echo "--- Listing top-level of remote container (${REMOTE_BASE_DEST}) ---"
rclone lsd "${REMOTE_BASE_DEST}"
echo

echo "--- load-data.sh script finished ---"