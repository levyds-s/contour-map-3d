#!/bin/bash

# Configuration
STEP=10
STRETCH=5.0
BASE=2.0
INPUT_DIR="exemplos"
OUTPUT_DIR="output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all .jpeg and .jpg files
for img in "$INPUT_DIR"/*.[jJ][pP][eE]*[gG]; do
    [ -e "$img" ] || continue
    
    filename=$(basename -- "$img")
    basename="${filename%.*}"
    
    echo "Processing $filename (Stretch: ${STRETCH}x)..."
    
    python main.py "$img" \
        --output "$OUTPUT_DIR/$basename.stl" \
        --smooth
done

echo "---------------------------------------"
echo "Done. Models are in the '$OUTPUT_DIR' folder."
