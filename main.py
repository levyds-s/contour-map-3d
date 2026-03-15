import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stl import mesh

def is_line_dashed(contour, binary_original):
    mask = np.zeros(binary_original.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, 1)
    points = np.where(mask == 255)
    line_pixels = binary_original[points]
    if len(line_pixels) == 0: return False
    return (np.count_nonzero(line_pixels) / len(line_pixels)) < 0.7

def image_to_stl(image_path, output_name, elevation_step, base_thickness, smooth, stretch, preview):
    img = cv2.imread(image_path)
    if img is None: return
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary_closed = cv2.dilate(binary, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(binary_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or hierarchy is None: return

    heightmap = np.zeros((h, w), dtype=np.float32)
    img_area = h * w

    dashed_status = [is_line_dashed(cnt, binary) for cnt in contours]
    
    frame_idx = -1
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > (img_area * 0.90):
            frame_idx = i
            break

    contour_indices = sorted(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]), reverse=True)

    for i in contour_indices:
        if i == frame_idx: continue
        
        z_val = 0
        curr = i

        while curr != -1:
            if curr != frame_idx:
                z_val += elevation_step if dashed_status[curr] else -elevation_step
            curr = hierarchy[0][curr][3]
        
        cv2.drawContours(heightmap, [contours[i]], -1, float(z_val), thickness=cv2.FILLED)

    min_z = np.min(heightmap)
    heightmap = (heightmap - min_z) * stretch + base_thickness
    
    if smooth:
        heightmap = cv2.GaussianBlur(heightmap, (21, 21), 0)

    rows, cols = heightmap.shape
    grid_step = 2
    faces = []

    for r in range(0, rows - grid_step, grid_step):
        for c in range(0, cols - grid_step, grid_step):
            p1 = [c, r, heightmap[r, c]]
            p2 = [c + grid_step, r, heightmap[r, c + grid_step]]
            p3 = [c, r + grid_step, heightmap[r + grid_step, c]]
            p4 = [c + grid_step, r + grid_step, heightmap[r + grid_step, c + grid_step]]

            faces.append([p1, p3, p2])
            faces.append([p2, p3, p4])

    model_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        model_mesh.vectors[i] = f

    model_mesh.save(output_name)

    if preview:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        s = 10
        ax.plot_surface(X[::s,::s], Y[::s,::s], heightmap[::s,::s], cmap='terrain')
        ax.set_ylim(h, 0)
        plt.title("Mountains high, valleys low")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="output.stl")
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--base", type=float, default=2.0)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--stretch", type=float, default=1.0)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    image_to_stl(
        args.input,
        args.output,
        args.step,
        args.base,
        args.smooth,
        args.stretch,
        args.preview
    )
