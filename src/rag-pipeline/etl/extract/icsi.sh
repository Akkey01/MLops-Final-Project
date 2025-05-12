#!/usr/bin/env bash
set -e

# Dest dir under block storage
DEST="/data/raw/icsi/Signals"

# List of meeting IDs to pull
for MID in Bdb001 Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010; do
  mkdir -p "${DEST}/${MID}"
  wget -q -P "${DEST}/${MID}" \
    "https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/${MID}.interaction.wav"
done

# Download manifest and license
wget -q -O /data/raw/icsi/manifest.txt \
  "https://groups.inf.ed.ac.uk/ami/download/temp/icsiBuild-15735-Sun-May-11-2025.manifest.txt"
wget -q -O /data/raw/icsi/CCBY4.0.txt \
  "https://groups.inf.ed.ac.uk/ami/download/temp/CCBY4.0.txt"

echo "✅ ICSI signals, manifest, and license downloaded."
