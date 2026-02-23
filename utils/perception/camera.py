import pyrealsense2 as rs
import numpy as np
import open3d as o3d

realsense_serial_numbers = {
    0: "123456789012", # TODO: update
    1: "234567890123"
}

class multi_cam:
    def __init__(self):
        self.pipelines = []
        self.configs = []
        self.align = rs.align(rs.stream.color)
        
        # Initialize all available cameras
        ctx = rs.context()
        for serial in realsense_serial_numbers.values():
            pipe = rs.pipeline(ctx)
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipe.start(cfg)
            self.pipelines.append(pipe)
            self.configs.append(cfg)

    def take_bgrd(self, visualize=False):
        """
        Captures and processes data from all configured cameras.
        Returns:
            bgr_images: Dict mapping serial to BGR numpy array
            pcd_merged: open3d.geometry.PointCloud (combined from all cams)
            raw_points: List of 3D point arrays for each camera
            None: Placeholder for the 4th return value in your template
        """
        # Create an instance if not already managed globally
        # In a real setup, you'd likely keep the instance alive
        cam_system = multi_cam()
        
        bgr_images = {}
        raw_points = []
        pcd_merged = o3d.geometry.PointCloud()
        
        # Warm up sensors to allow auto-exposure to settle
        for _ in range(30):
            for pipe in cam_system.pipelines:
                pipe.wait_for_frames()

        for i, (pipe, serial) in enumerate(zip(cam_system.pipelines, realsense_serial_numbers.values())):
            frames = pipe.wait_for_frames()
            aligned_frames = cam_system.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            color_image = np.asanyarray(color_frame.get_data())
            bgr_images[serial] = color_image
            
            # Generate Point Cloud for this camera
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            v = points.get_vertices()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            
            # Store raw points (reshaped later in perception_utils)
            raw_points.append(verts)
            
            # Create Open3D cloud for merging
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(verts)
            
            # Extract colors for the point cloud
            color_arr = color_image.reshape(-1, 3) / 255.0
            temp_pcd.colors = o3d.utility.Vector3dVector(color_arr[:, ::-1]) # BGR to RGB
            
            pcd_merged += temp_pcd

        # Clean up pipelines
        for pipe in cam_system.pipelines:
            pipe.stop()

        if visualize:
            o3d.visualization.draw_geometries([pcd_merged])

        return bgr_images, pcd_merged, raw_points, None