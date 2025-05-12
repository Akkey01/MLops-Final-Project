#!/usr/bin/env bash
set -e

# Downloads ICSI wav signals into /data/raw/icsi/Signals/<meeting_id>/
# Make sure this file is executable (chmod +x)

DEST="/data/raw/icsi/Signals"

# List of meetings to pull
for MID in Bdb001 Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010; do
  mkdir -p "${DEST}/${MID}"
  wget -q -P "${DEST}/${MID}" \
    "https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/${MID}.interaction.wav"
done

# download Manifest / license i
wget -q -O /data/raw/icsi/manifest.txt \
  "https://groups.inf.ed.ac.uk/ami/download/temp/icsiBuild-143922-Sun-May-11-2025.manifest.txt"

