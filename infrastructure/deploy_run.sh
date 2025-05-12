#!/usr/bin/env bash
set -euo pipefail

REPO=MLops-Final-Project
SRC_DIR=src
IMAGE_TAG=a79f7185a34e
SSH_ID=~/.ssh/id_rsa_chameleon

# 1) Clone or update
if [ -d "$REPO" ]; then
  echo "↻  '$REPO' exists, pulling latest changes…"
  cd "$REPO"
  git pull
else
  echo " Cloning '$REPO'…"
  git clone https://github.com/Akkey01/MLops-Final-Project.git "$REPO"
  cd "$REPO"
fi

# 2) Quick listings
echo
echo " top-level files:"
ls

echo
echo " running containers:"
docker ps

# 3) Build
echo
echo " Building image from ./$SRC_DIR…"
cd "$SRC_DIR"
docker build .

# 4) Show run
echo
echo " Running container on :8000…"
docker run -p 8000:8000 "$IMAGE_TAG"