#!/usr/bin/env bash
set -eux

# where we live:
cd /data

# 1) make sure our dirs exist
mkdir -p raw/ami raw/meetingbank

# 2) AMI corpus (required)
if [ -d raw/ami/amicorpus ] && [ "$(ls -A raw/ami/amicorpus)" ]; then
  echo "AMI corpus already extracted, skipping."
else
  if [ -f raw/ami/amicorpus.tgz ]; then
    echo "Extracting AMI corpus archive…"
    tar -xzf raw/ami/amicorpus.tgz -C raw/ami
  else
    echo "Error: amicorpus.tgz not found in raw/ami. Please download it there." >&2
    exit 1
  fi
fi

# 3) MeetingBank (optional)
if [ -d raw/meetingbank/MeetingBank ] && [ "$(ls -A raw/meetingbank/MeetingBank)" ]; then
  echo "MeetingBank already extracted, skipping."
else
  if [ -f raw/meetingbank/MeetingBank.zip ]; then
    echo "Extracting MeetingBank archive…"
    unzip -q raw/meetingbank/MeetingBank.zip -d raw/meetingbank
  else
    echo "Warning: MeetingBank.zip not found in raw/meetingbank — skipping MeetingBank entirely." >&2
  fi
fi

# 4) show us what we got
du -sh raw/ami raw/meetingbank