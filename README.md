# 3D Contour to STL Setup

### 1. Install Dependencies
pip install -r requirements.txt

### 2. Run Single Image
python main.py input.jpg --output model.stl --step 10 --smooth --color terrain

### 3. Run Batch Script
chmod +x run_all.sh
./run_all.sh

### CLI Options
- `--step`: Height per contour (default: 5.0)
- `--base`: Floor thickness (default: 2.0)
- `--valley`: Invert heights (inner rings become holes)
- `--smooth`: Apply Gaussian blur to soften edges
- `--color`: Apply visual colormap (e.g., 'viridis', 'terrain', 'magma')
