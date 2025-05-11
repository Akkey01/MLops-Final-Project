cc@node-persist-am15111-nyu-edu:~/my-etl$ cat docker-compose.yml
services:

  extract-data:
    image: python:3.11-slim
    working_dir: /data
    volumes:
      - /mnt/block:/data
      - ./extract-data.sh:/data/extract-data.sh:ro
      - ./wget.txt:/data/wget.txt:ro
    entrypoint: ["bash","/data/extract-data.sh"]


  transform-data:
    image: python:3.11-slim
    working_dir: /data
    volumes:
      - /mnt/block:/data
      - ./transform-data.sh:/data/transform-data.sh:ro
    entrypoint: ["bash","/data/transform-data.sh"]

  load-data:
    image: rclone/rclone:latest
    working_dir: /data
    environment:
      RCLONE_CONTAINER: object-persist-project39
    volumes:
      - /mnt/block:/data
      - ~/.config/rclone/rclone.conf:/root/.config/rclone/rclone.conf:ro
      - ./load-data.sh:/data/load-data.sh:ro
    entrypoint: ["sh","/data/load-data.sh"]
cc@node-persist-am15111-nyu-edu:~/my-etl$ cat extract-data.sh
#!/usr/bin/env bash
set -eux

# base working directory inside the container
WORKDIR="/data"
RAW_ROOT="${WORKDIR}/raw/newami"
WGET_TXT="${WORKDIR}/wget.txt"

cd "${WORKDIR}"

# 1) Ensure the raw directories exist
mkdir -p \
  "${RAW_ROOT}/amicorpus" \
  "${RAW_ROOT}/manual_annotations" \
  "${RAW_ROOT}/automatic_annotations"

# 2) AMI corpus itself: skip if already present
if [ -d "${RAW_ROOT}/amicorpus" ] && [ "$(ls -A "${RAW_ROOT}/amicorpus")" ]; then
  echo "▶ AMI corpus already present – skipping download/extract."

# 2a) If wget.txt is provided, run it to grab everything
elif [ -f "${WGET_TXT}" ]; then
  echo "▶ Running wget commands from ${WGET_TXT} to populate AMI…"
  cd "${RAW_ROOT}/amicorpus"
  bash "${WGET_TXT}"
  cd "${WORKDIR}"

# 2b) Otherwise, if you’ve jiggled down a .tgz, extract it
elif [ -f "${RAW_ROOT}/amicorpus.tgz" ]; then
  echo "▶ Extracting ${RAW_ROOT}/amicorpus.tgz…"
  tar -xzf "${RAW_ROOT}/amicorpus.tgz" -C "${RAW_ROOT}"

else
  echo "✖ Error: no AMI source found (wget.txt or amicorpus.tgz or existing folder)." >&2
  exit 1
fi

# 3) Manual annotations
if [ -d "${RAW_ROOT}/manual_annotations" ] && [ "$(ls -A "${RAW_ROOT}/manual_annotations")" ]; then
  echo "▶ Manual annotations already present – skipping."
else
  echo "▶ Downloading & extracting manual_annotations v1.6.2…"
  wget -q -O "${RAW_ROOT}/ami_public_manual_1.6.2.zip" \
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
  unzip -q "${RAW_ROOT}/ami_public_manual_1.6.2.zip" \
    -d "${RAW_ROOT}/manual_annotations"
fi

# 4) Automatic annotations
if [ -d "${RAW_ROOT}/automatic_annotations" ] && [ "$(ls -A "${RAW_ROOT}/automatic_annotations")" ]; then
  echo "▶ Automatic annotations already present – skipping."
else
  echo "▶ Downloading & extracting automatic_annotations v1.5.1…"
  wget -q -O "${RAW_ROOT}/ami_public_auto_1.5.1.zip" \
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_auto_1.5.1.zip"
  unzip -q "${RAW_ROOT}/ami_public_auto_1.5.1.zip" \
    -d "${RAW_ROOT}/automatic_annotations"
fi

# 5) Final summary
echo "▶ Final sizes:"
du -sh \
  "${RAW_ROOT}/amicorpus" \
  "${RAW_ROOT}/manual_annotations" \
  "${RAW_ROOT}/automatic_annotations"

echo "=== extract-data.sh complete ==="