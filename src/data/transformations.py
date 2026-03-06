import torch
import pytorch3d.transforms as pt3d
"""
This module contains various transformations that can be applied to the point cloud data.


Possible transformations:
- isolate external points (done)
- isolate internal points (done)
- isolate hemisphere (done)
- front, middle, back parcellation (done)
- random subsampling with a certain number of points (done)
- choose an hub (a certain point in space) and choose k points close to it (done)
- Compute singular values (done)
- Compute singular vectors (done)

Every transformation should be a function that takes in a Data object and returns a Data object
"""

"""
Hemisphere can be 'left' or 'right'
"""
def split_by_hemisphere(data, hemisphere):
    point_cloud = data.x
    if hemisphere == 'left':
        data.x = point_cloud[:100,:]
    elif hemisphere == 'right':
        data.x = point_cloud[100:,:]
    else:
        raise ValueError("Invalid hemisphere. Choose from 'left' or 'right'.")
    return data

def center(data):
    point_cloud = data.x
    centroid = point_cloud.mean(dim=0, keepdim=True)
    data.x = point_cloud - centroid
    return data

"""
partition_type: 'front', 'middle', 'back'
"""
def divide_point_cloud(data, partition_type):
    point_cloud = data.x
    n_points = point_cloud.shape[0]
    sorted_indices = torch.argsort(point_cloud[:, 0])
    n_points_per_part = n_points // 3
    if partition_type == 'front':
        data_indices = sorted_indices[:n_points_per_part]
    elif partition_type == 'middle':
        data_indices = sorted_indices[n_points_per_part:n_points_per_part*2]
    elif partition_type == 'back':
        data_indices = sorted_indices[n_points_per_part*2:]
    else:
        raise ValueError("Invalid partition type. Choose from 'front', 'middle', 'back'.")
    data.x = point_cloud[data_indices]
    return data
"""
Choose num_points randomly from the point cloud
"""
def random_subsample(data, num_points):
    point_cloud = data.x
    n_points = point_cloud.shape[0]
    if num_points > n_points:
        raise ValueError("num_points exceeds the number of points in the point cloud.")
    selected_indices = torch.randperm(n_points)[:num_points]
    data.x = point_cloud[selected_indices]
    return data

"""
Isolate all the points that are inside a certain threshold distance from the origin
"""
def internal_points(data, threshold=0.5):
    point_cloud = data.x
    n_points = point_cloud.shape[0]
    center = data.x.mean(dim=0, keepdim=True)
    distances = torch.norm(point_cloud - center, dim=1)
    closest_indices = torch.argsort(distances)[:n_points * threshold]
    data.x = point_cloud[closest_indices]
    return data

"""
Isolate all the points that are outside a certain threshold distance from the origin
"""
def external_points(data, threshold=0.5):
    internal_data = internal_points(data, 1-threshold)
    point_cloud = data.x
    n_points = point_cloud.shape[0]
    center = data.x.mean(dim=0, keepdim=True)
    distances = torch.norm(point_cloud - center, dim=1)
    closest_indices = torch.argsort(distances)[n_points * threshold:]
    data.x = point_cloud[closest_indices]
    return data

def isolate_hub_points(data, hub_point, k):
    point_cloud = data.x 
    if isinstance(hub_point, str):
        if hub_point == 'center':
            hub_point = point_cloud.mean(dim=0, keepdim=True)
        elif hub_point == 'front':
            # choose the point with the maximum x value
            hub_point = torch.argmax(point_cloud[:,0])
        elif hub_point == 'back':
            # choose the point with the minimum x value
            hub_point = torch.argmin(point_cloud[:,0])
        else:
            raise ValueError("Invalid hub_point. Currently only 'center' is supported as a string.")
    
    distances = torch.norm(point_cloud - hub_point, dim=1)
    data.x = point_cloud[torch.argsort(distances)[:k]]
    return data

"""
Compute the top num_vectors singular vectors of the point cloud and store them in data.singular_vectors
"""
def singular_vectors(data, num_vectors):
    point_cloud = data.x
    # Center the point cloud
    centered_pc = point_cloud - point_cloud.mean(dim=0, keepdim=True)
    # Compute SVD
    U, S, Vt = torch.svd(centered_pc)
    # Select the top num_vectors singular vectors
    data.singular_vectors = Vt[:, :num_vectors].T
    return data

"""
Compute the top num_values singular values of the point cloud and store them in data.singular_values
"""

def singular_values(data, num_values):
    point_cloud = data.x
    # Center the point cloud
    centered_pc = point_cloud - point_cloud.mean(dim=0, keepdim=True)
    # Compute SVD
    U, S, Vt = torch.svd(centered_pc)
    # Select the top num_values singular values
    data.singular_values = S[:num_values]
    return data

"""
Reshape the point cloud to a specified shape
"""

def reshape(data, shape):
    point_cloud = data.x
    data.x = point_cloud.view(shape)
    return data

"""
Apply a random rotation to the point cloud
"""

def random_rotation(data):
    point_cloud = data.x
    R = pt3d.random_rotation(dtype=point_cloud.dtype, device=point_cloud.device)  # Random (3, 3) rotation matrix
    data.x = point_cloud @ R.T  # Rotate the point cloud
    return data

"""
Add random jitter to the point cloud
"""

def add_jitter(data, scale=1):
    point_cloud = data.x
    noise = torch.randn_like(point_cloud) * scale
    data.x = point_cloud + noise
    return data

"""
Change orientation of the point cloud to a specified orientation
"""

def change_orientation(data, target_orientation='RL'):
    """
    Rotate data.pc by 180 degrees horizontally around centroid if data.pc_orientation != target_orientation.
    
    Args:
        data: Object with 'pc' ([200,3] tensor) and 'pc_orientation' (str) attributes.
        target_orientation: Target string (e.g., 'Left', 'Right').
    
    Returns:
        data with potentially rotated pc.
    """
    if data.pc_orientation != target_orientation:
        pc = data.pc  # [200, 3]
        centroid = pc.mean(dim=0)  # [3]
        centered = pc - centroid  # [200, 3]
        
        # 180 deg horizontal rotation around Y-axis (left-right flip)
        theta = torch.tensor(180.0 * torch.pi / 180, dtype=pc.dtype, device=pc.device)
        R_y = pt3d.RotateY(theta).get_matrix()  # [3,3]
        
        rotated_centered = (centered @ R_y.T)  # [200,3]
        data.pc = rotated_centered + centroid
    
        #Update orientation after flip
        data.pc_orientation = target_orientation
    
    return data

"""
Registry of available transformations
"""

_TRANSFORMATIONS = {
    'split_by_hemisphere': split_by_hemisphere,
    'center': center,
    'divide_point_cloud': divide_point_cloud,
    'random_subsample': random_subsample,
    'internal_points': internal_points,
    'external_points': external_points,
    'isolate_hub_points': isolate_hub_points,
    'singular_vectors': singular_vectors,
    'singular_values': singular_values,
    'reshape': reshape,
    'random_rotation': random_rotation,
    'change_orientation': change_orientation,
}

def get_transformation(transformation_names, params):
    """
    Compose transformations by name with their parameters.
    
    Args:
        transformation_names: List of function names to compose
        params: Dict mapping function names to their arguments
        
    Example:
        transform_fn = get_transformation(
            ['center', 'random_subsample', 'singular_values'],
            {
                'center': {},
                'random_subsample': {'num_points': 100},
                'singular_values': {'num_values': 3}
            }
        )
    """
    def composed_transform(data):
        for name in transformation_names:
            if name not in _TRANSFORMATIONS:
                raise ValueError(f"Unknown transformation: {name}")
            fn = _TRANSFORMATIONS[name]
            fn_params = params.get(name, {})
            data = fn(data, **fn_params)
        return data
    
    return composed_transform