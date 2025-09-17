import pycolmap
import open3d as o3d
import numpy as np

# Load the reconstruction (COLMAP binary model folder)
recon = pycolmap.Reconstruction("featureout/ETs/colmap_rec_aliked/")
# The folder must contain images.bin, points3D.bin, cameras.bin
recon.export_PLY("out.ply")  # optional: export to PLY file

def view_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        print("No points found in:", ply_path)
        return
    o3d.visualization.draw_geometries([pcd], window_name="COLMAP dense.ply", width=1280, height=720)

view_ply("out.ply")
