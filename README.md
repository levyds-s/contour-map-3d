# 3D Contour to STL

Converts topographic contour images into 3D surface models. 
Supports dashed lines for depressions and nested logic.

### 1. Setup
pip install -r requirements.txt

### 2. Batch Process (All files in /exemplos)
chmod +x run_all.sh
./run_all.sh

### 3. Single File Execution
python main.py exemplos/map.jpg --output output/model.stl --step 10 --stretch 5.0 --smooth

### CLI Parameters
- `--step`: Height difference between lines.
- `--stretch`: Vertical exaggeration (increase if model looks flat).
- `--base`: Minimum thickness of the model.
- `--smooth`: Rounds off "steps" for a natural look.
