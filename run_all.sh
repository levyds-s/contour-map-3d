#!/bin/bash

# Configuration
STEP=10
BASE=2.0
COLOR="terrain"
INPUT_DIR="exemplos"
OUTPUT_DIR="output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

for img in "$INPUT_DIR"/*.[jJ][pP][eE]*[gG]; do
    [ -e "$img" ] || { echo "No images found in $INPUT_DIR"; exit 1; }

    filename=$(basename -- "$img")
    basename="${filename%.*}"

    echo "Processing $img -> $OUTPUT_DIR/$basename.stl"

    python main.py "$img" \
        --output "$OUTPUT_DIR/$basename.stl" \
        --step $STEP \
        --base $BASE \
        --smooth \
        --color $COLOR
done

echo "-----------------------------------"
echo "Done. Models saved to $OUTPUT_DIR/"
