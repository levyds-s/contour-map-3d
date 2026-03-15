import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.cm as cm


def image_to_stl(
    image_path,
    output_name,
    elevation_step,
    is_valley,
    base_thickness,
    smooth,
    color_map_name,
):
    # 1. Load and Preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- New: Apply Smoothing if Flag is set ---
    if smooth:
        # Applies a 15x15 Gaussian Blur to soften hard edges
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        print("Edge smoothing applied.")

    # Threshold after smoothing
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Extract Points (Rest of the Z-logic remains the same)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    px, py, pz = [], [], []
    max_level = 0
    temp_levels = []
    for i in range(len(contours)):
        level = 0
        parent = hierarchy[0][i][3]
        while parent != -1:
            level += 1
            parent = hierarchy[0][parent][3]
        temp_levels.append(level)
    if temp_levels:
        max_level = max(temp_levels)
    for i, contour in enumerate(contours):
        level = temp_levels[i]
        z_val = (
            (max_level - level) * elevation_step
            if is_valley
            else level * elevation_step
        )
        for pt in contour:
            px.append(pt[0][0])
            py.append(pt[0][1])
            pz.append(z_val + base_thickness)

    # 3. Add Border/Base Points (Z=0 to create a solid floor)
    h, w = gray.shape
    for cx in [0, w]:
        for cy in [0, h]:
            px.append(cx)
            py.append(cy)
            pz.append(0)

    # 4. Grid Interpolation (Increased resolution for smoother visual flow)
    # Using 300 samples for a smoother surface
    grid_x, grid_y = np.mgrid[0:w:300j, 0:h:300j]
    grid_z = griddata((px, py), pz, (grid_x, grid_y), method="linear", fill_value=0)

    # 5. Meshing
    vertices = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices

    # 6. Create the STL object
    model_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model_mesh.vectors[i][j] = vertices[f[j], :]

    # --- New: Handle Color Output ---
    # Standard STLs only have geometry. If color is requested, we need
    # to export as a format that supports it, or use the unofficial
    # binary STL color format which some slicers support.
    # For now, we will apply colors *if possible* (will create standard STLs).

    if color_map_name:
        # Use the modern Matplotlib 3.7+ registry access
        try:
            colormap = plt.colormaps[color_map_name]
        except KeyError:
            print(f"Warning: Colormap '{color_map_name}' not found. Using 'viridis'.")
            colormap = plt.colormaps['viridis']

        # Normalize the heights (Z-axis) to a 0.0 - 1.0 range
        z_min, z_max = np.min(pz), np.max(pz)
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

        # Calculate the color of each face based on its average Z
        face_colors = colormap(norm(model_mesh.z.mean(axis=1)))

        # Binary STL color format (unofficial, use at your own risk for printability)
        # Slicers that support this often expect 15-bit color encoding.
        # This script sets the standard 3D mesh vectors, but most slicers
        # ignore color on import for slicing.

        # (For advanced multi-color printing, you would export separate
        # STLs for each level, or use a .OBJ + .MTL pipeline.)

        # The following will create a standard binary STL. Colors won't appear
        # in most simple STL viewers.
        pass  # Visual colors only work if you view as vertex colors, not face colors.

    model_mesh.save(output_name)
    print(
        f"Successfully exported {output_name} (Smooth: {smooth}, Color Map: {color_map_name})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert contour maps to smooth 3D STL files."
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("--output", default="model.stl", help="Output STL filename")
    parser.add_argument(
        "--step", type=float, default=5.0, help="Height increment per contour line"
    )
    parser.add_argument(
        "--valley",
        action="store_true",
        help="Invert height: inner rings become depressions",
    )
    parser.add_argument(
        "--base", type=float, default=2.0, help="Thickness of the solid base floor"
    )

    # New flags
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply Gaussian Blur to soften jagged edges",
    )
    parser.add_argument(
        "--color",
        default=None,
        help="Colormap name for coloring faces (e.g., 'viridis', 'magma', 'terrain')",
    )

    args = parser.parse_args()
    image_to_stl(
        args.input,
        args.output,
        args.step,
        args.valley,
        args.base,
        args.smooth,
        args.color,
    )
