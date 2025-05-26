#!/usr/bin/env bash
set -e
DEST=/data/raw/icsi/audio
mkdir -p "$DEST"
for MID in Bdb001 Bed002 Bed003 Bed004 Bed005 Bed006; do
  mkdir -p "${DEST}/${MID}"
  wget -q -P "${DEST}/${MID}" \
    "https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/${MID}.interaction.wav"
done
# manifest + license
wget -q -O /data/raw/icsi/manifest.txt \
  "https://groups.inf.ed.ac.uk/ami/download/temp/icsiBuild-15735-Sun-May-11-2025.manifest.txt"
wget -q -O /data/raw/icsi/CCBY4.0.txt \
  "https://groups.inf.ed.ac.uk/ami/download/temp/CCBY4.0.txt"
echo "✅ ICSI signals downloaded."