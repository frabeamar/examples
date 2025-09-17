from pathlib import Path
import pycolmap
import open3d as o3d
import numpy as np

# Load the reconstruction (COLMAP binary model folder)
# The folder must contain images.bin, points3D.bin, cameras.bin
def view_ply(ply_path):
    
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        print("No points found in:", ply_path)
        return
    o3d.visualization.draw_geometries([pcd], window_name="COLMAP dense.ply", width=1280, height=720)


for dataset in Path("featureout").iterdir():
    colmap = dataset/"colmap_rec_aliked"
    recon = pycolmap.Reconstruction(colmap)
    print(dataset, recon.summary())

    # for image_id, image in recon.images.items():
    #     print(image_id, image)

    # for point3D_id, point3D in recon.points3D.items():
    #     print(point3D_id, point3D)

    # for camera_id, camera in recon.cameras.items():
    #     print(camera_id, camera)

    recon.export_PLY(dataset / "dense.ply")  # optional: export to PLY file
    view_ply(dataset / "dense.ply")


