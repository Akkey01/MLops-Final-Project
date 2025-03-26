#!/bin/bash

BASE_URL="https://groups.inf.ed.ac.uk/ami//ICSIsignals"
SIGNALS=(
  "Bdb001:NXT/Bdb001.interaction.wav SPH/Bdb001/{chan1,chan2,chan3,chan4,chan6,chan7,chan8,chanB,chanC,chanD,chanE,chanF}.sph"
  "Bed002:NXT/Bed002.interaction.wav SPH/Bed002/{chan0,chan1,chan2,chan3,chan4,chan6,chan7,chan8,chanB,chanC,chanD,chanE,chanF}.sph"
  "Bed003:NXT/Bed003.interaction.wav SPH/Bed003/{chan1,chan2,chan3,chan4,chan6,chan7,chanC,chanD,chanE,chanF}.sph"
)

for signal in "${SIGNALS[@]}"; do
  IFS=":" read -r dir paths <<< "$signal"
  mkdir -p "Signals/$dir"
  for path in $(eval echo $paths); do
    wget -P "Signals/$dir" "$BASE_URL/$path"
  done
done

wget "https://groups.inf.ed.ac.uk/ami//download/temp/icsiBuild-204230-Wed-Mar-26-2025.manifest.txt"
wget "https://groups.inf.ed.ac.uk/ami//download/temp/../CCBY4.0.txt"
