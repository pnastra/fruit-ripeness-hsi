#!/usr/bin/env bash
# Download DeepHS Fruit 2023: annotations + one fruit (default Mango). Resumable and idempotent.
set -euo pipefail

FRUIT="${FRUIT:-Mango}"
BASE="https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets"
RAW="data/raw"
mkdir -p "$RAW/_zips"

fetch() {  # fetch <file>: -C - resumes a partial download, -f fails on HTTP errors
    curl -fL -s -S -C - --retry 5 -o "$RAW/_zips/$1" "$BASE/$1"
}

fetch "annotations-upd-2024-01-09.zip"
fetch "$FRUIT.zip"

# The zips already contain a top-level folder (Mango/, annotations/), so extract into data/raw.
[ -d "$RAW/annotations" ] || unzip -q -o "$RAW/_zips/annotations-upd-2024-01-09.zip" -d "$RAW"
[ -d "$RAW/$FRUIT" ] || unzip -q -o "$RAW/_zips/$FRUIT.zip" -d "$RAW"
echo "done: $RAW/$FRUIT and $RAW/annotations"
