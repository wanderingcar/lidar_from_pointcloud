import open3d as o3d
import numpy as np


def read_pose_log(filename):
    """
    Reads a camera pose log file and returns a list of transformation matrices.
    :param filename: path to the camera pose log file
    :return: homogeneous transformation matrices
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    matrices = []
    i = 0
    while i < len(lines):
        # Skip the first line of each block (the three integers)
        i += 1
        # Read the next 4 lines as a matrix
        matrix = []
        for _ in range(4):
            row = np.array(list(map(float, lines[i].split())))
            matrix.append(row)
            i += 1
        matrices.append(np.array(matrix))

    return np.array(matrices)


def sample_points_like_lidar(pcd, h_fov=np.radians(360), v_fov=np.radians(30),
                             h_resolution=0.16, num_channels=16, max_range=100):
    """
    Samples points from a point cloud to simulate a lidar scan.
    :param pcd: point cloud
    :param h_fov: horizontal field of view of the lidar
    :param v_fov: vertical field of view of the lidar
    :param h_resolution: angular resolution of the lidar
    :param num_channels: number of channels of the lidar
    :param max_range: maximum range of the lidar
    :return: sampled point cloud
    """
    # Convert Open3D.o3d.geometry.PointCloud to numpy array
    points = np.asarray(pcd.points)

    # Get the distances and angles of all points
    distances = np.linalg.norm(points, axis=1)
    angles_h = np.arctan2(points[:, 1], points[:, 0])
    angles_v = np.arcsin(points[:, 2] / distances)

    # Create a mask of points within the FOV and range of the Lidar
    mask = (np.abs(angles_h) <= h_fov / 2) & (np.abs(angles_v) <= v_fov / 2) & (distances <= max_range)

    # Create a discretized grid of horizontal and vertical angles
    angles_h_grid = np.linspace(-h_fov / 2, h_fov / 2, int(h_fov / h_resolution), endpoint=False)
    angles_v_grid = np.linspace(-v_fov / 2, v_fov / 2, num_channels, endpoint=False)

    # Discretize the angles of the points within the FOV and range
    discretized_angles_h = np.digitize(angles_h[mask], angles_h_grid)
    discretized_angles_v = np.digitize(angles_v[mask], angles_v_grid)

    # Initialize the Lidar-like depth map
    depth_map = np.full((len(angles_v_grid), len(angles_h_grid)), np.inf)

    # For each point within the FOV and range,
    # if it's closer than the current point in the depth map at the same discretized angle, replace it
    for angle_v, angle_h, distance in zip(discretized_angles_v, discretized_angles_h, distances[mask]):
        if distance < depth_map[angle_v, angle_h]:
            depth_map[angle_v, angle_h] = distance

    # Replace infinities with the maximum range
    depth_map[depth_map == np.inf] = max_range

    return depth_map


def depth_map_to_point_cloud(depth_map, angles_h_grid, angles_v_grid):
    """
    Convert a depth map back into a point cloud.
    :param depth_map: the Lidar-like depth map
    :param angles_h_grid: discretized grid of horizontal angles
    :param angles_v_grid: discretized grid of vertical angles
    :return: point cloud
    """

    # Initialize the point cloud
    points = []

    # For each point in the depth map, if it's not infinity, add it to the point cloud
    for i, row in enumerate(depth_map):
        for j, depth in enumerate(row):
            if depth != np.inf:
                # Convert from spherical coordinates to Cartesian coordinates
                angle_h = angles_h_grid[j]
                angle_v = angles_v_grid[i]
                x = depth * np.cos(angle_v) * np.cos(angle_h)
                y = depth * np.cos(angle_v) * np.sin(angle_h)
                z = depth * np.sin(angle_v)
                points.append([x, y, z])

    return np.array(points)



if __name__ == '__main__':
    # Paths
    ply_path = "/home/cadit/data/redwood/apartment/full_ply/apt.ply"
    log_path = "/home/cadit/data/redwood/apartment/pose/apartment.log"
    output_path = "/home/cadit/data/redwood/apartment/sampled_ply/"

    # Load PLY file
    pcd = o3d.io.read_point_cloud(ply_path)

    # Load Camera Poses - 4x4 transformation matrices (tcw: redwood dataset format)
    camera_poses = read_pose_log(log_path)

    for i, T in enumerate(camera_poses):
        # Transform pointcloud into camera frame
        T_inv = np.linalg.inv(T)
        pcd_transformed = pcd.transform(T_inv)

        # Sample Points as if they were from a LiDAR
        depth_map = sample_points_like_lidar(pcd_transformed)

        # Convert depth map to point cloud
        points = depth_map_to_point_cloud(depth_map, angles_h_grid, angles_v_grid)
        pcd_sampled = o3d.geometry.PointCloud()
        pcd_sampled.points = o3d.utility.Vector3dVector(points)

        # Save the sampled point cloud
        output_path_ply = output_path + f"apt_{i}.ply"
        o3d.io.write_point_cloud(output_path_ply, pcd_sampled)
        print(f"Saved {output_path_ply}")
