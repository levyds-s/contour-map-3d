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

def image_to_stl(image_path, output_name, elevation_step, base_thickness, smooth, stretch, preview, valley_stretch, simplify):

    img = cv2.imread(image_path)
    if img is None: return

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
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
        if i == frame_idx: 
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contours[i]], -1, 255, thickness=cv2.FILLED)

        # Your working elevation logic
        delta = (elevation_step * valley_stretch) if dashed_status[i] else -elevation_step

        heightmap[mask == 255] += delta

    min_z = np.min(heightmap)
    heightmap = (heightmap - min_z) * stretch + base_thickness

    if smooth:
        heightmap = cv2.GaussianBlur(heightmap, (21,21), 0)

    rows, cols = heightmap.shape
    # Now driven by the --simplify parameter
    grid_step = simplify
    faces = []

    for r in range(0, rows-grid_step, grid_step):
        for c in range(0, cols-grid_step, grid_step):

            p1 = [c, r, heightmap[r,c]]
            p2 = [c+grid_step, r, heightmap[r,c+grid_step]]
            p3 = [c, r+grid_step, heightmap[r+grid_step,c]]
            p4 = [c+grid_step, r+grid_step, heightmap[r+grid_step,c+grid_step]]

            faces.append([p1,p2,p3])
            faces.append([p2,p4,p3])

    bottom_z = 0

    # Side walls and bottom generation, stepping by grid_step
    for c in range(0, cols-grid_step, grid_step):
        z1 = heightmap[0,c]
        z2 = heightmap[0,c+grid_step]

        t1=[c,0,z1]
        t2=[c+grid_step,0,z2]
        b1=[c,0,bottom_z]
        b2=[c+grid_step,0,bottom_z]

        faces.append([t1,b1,t2])
        faces.append([t2,b1,b2])

    for c in range(0, cols-grid_step, grid_step):
        z1 = heightmap[rows-1,c]
        z2 = heightmap[rows-1,c+grid_step]

        t1=[c,rows-1,z1]
        t2=[c+grid_step,rows-1,z2]
        b1=[c,rows-1,bottom_z]
        b2=[c+grid_step,rows-1,bottom_z]

        faces.append([t1,t2,b1])
        faces.append([t2,b2,b1])

    for r in range(0, rows-grid_step, grid_step):
        z1 = heightmap[r,0]
        z2 = heightmap[r+grid_step,0]

        t1=[0,r,z1]
        t2=[0,r+grid_step,z2]
        b1=[0,r,bottom_z]
        b2=[0,r+grid_step,bottom_z]

        faces.append([t1,t2,b1])
        faces.append([t2,b2,b1])

    for r in range(0, rows-grid_step, grid_step):
        z1 = heightmap[r,cols-1]
        z2 = heightmap[r+grid_step,cols-1]

        t1=[cols-1,r,z1]
        t2=[cols-1,r+grid_step,z2]
        b1=[cols-1,r,bottom_z]
        b2=[cols-1,r+grid_step,bottom_z]

        faces.append([t1,b1,t2])
        faces.append([t2,b1,b2])

    for r in range(0, rows-grid_step, grid_step):
        for c in range(0, cols-grid_step, grid_step):

            p1=[c,r,bottom_z]
            p2=[c+grid_step,r,bottom_z]
            p3=[c,r+grid_step,bottom_z]
            p4=[c+grid_step,r+grid_step,bottom_z]

            faces.append([p1,p3,p2])
            faces.append([p2,p3,p4])

    model_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))

    for i,f in enumerate(faces):
        model_mesh.vectors[i] = f

    model_mesh.update_normals()
    model_mesh.save(output_name)
    print(f"Saved {output_name} with simplify={simplify}")

    if preview:
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        X,Y = np.meshgrid(np.arange(w), np.arange(h))
        s=10
        ax.plot_surface(X[::s,::s],Y[::s,::s],heightmap[::s,::s],cmap='terrain')
        ax.set_ylim(h,0)
        plt.title("Closed terrain mesh")
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
    parser.add_argument("--valley-stretch", type=float, default=3.0)
    
    # New argument to control file size. Default is 2 (75% smaller than step=1).
    parser.add_argument("--simplify", type=int, default=2, help="Higher number = smaller file size (try 2, 3, or 4)")

    args = parser.parse_args()

    image_to_stl(
        args.input,
        args.output,
        args.step,
        args.base,
        args.smooth,
        args.stretch,
        args.preview,
        args.valley_stretch,
        args.simplify
    )
