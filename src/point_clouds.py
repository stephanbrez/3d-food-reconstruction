"""
═════ 3D Point Cloud Processing for Food Scene Reconstruction ═════

This module provides comprehensive tools for converting 2D segmentation masks and depth maps
into 3D point clouds, fitting geometric primitives (planes), and computing real-world
measurements of food items and tableware.

Key Features:
- Convert segmentation masks to 3D point clouds using depth information
- Fit planes to point cloud data for table surface estimation
- Project objects onto fitted planes to compute real-world dimensions
- Visualize results with interactive 3D plotting and image overlays
- Handle outliers and provide robust geometric computations

Typical Workflow:
1. Input: Segmentation masks (pixel coordinates) + depth map + camera parameters
2. Generate 3D point clouds from masks using camera projection
3. Fit plane to reference object (e.g., plate) to establish coordinate system
4. Project all objects onto the fitted plane
5. Compute real-world dimensions (width, height, diagonal) in meters
6. Visualize results for validation

Dependencies:
- torch: GPU-accelerated tensor operations for geometric computations
- numpy: Array operations and mathematical functions
- opencv-python (cv2): Image processing and geometric operations
- matplotlib: 3D plotting and visualization
- plotly (optional): Interactive 3D visualization
- typing: Type annotations for better code documentation

Example Usage:
```python
# Project masks to 3D point clouds
clouds = project_masks_to_pointclouds(depth_map, masks, f_px, "plate", cx, cy)

# Compute real-world dimensions using plate as reference
dimensions = compute_true_XY(clouds, plane_key="plate", known_plate_diameter=0.25)

# Visualize the fitted plane
visualize_plane_fit(clouds["plate"], dimensions["plate"]["diag"], f_px, image, cx, cy)
```

Authors:
License: MIT
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt

def remove_mask_outliers(mask: np.ndarray, distance_threshold: float = 30) -> np.ndarray:
    """
    Remove outlier points from a contour mask by identifying discontinuities
    in the contour sequence.

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates forming a contour
    distance_threshold : float, default=30
        Maximum distance between consecutive points to be considered part of the same contour

    Returns
    -------
    np.ndarray
        Cleaned contour mask with outliers removed
    """
    if mask.shape[0] < 10:
        return mask  # Too few points to process

    # Convert to the format required by OpenCV
    points = mask.astype(np.float32)

    # If the mask contains too many points, OpenCV's contour operations might be slow
    # In that case, we'll resample the contour
    if points.shape[0] > 1000:
        # Calculate the perimeter of the contour
        perimeter = cv2.arcLength(points.reshape(-1, 1, 2), closed=True)
        # Approximate the contour with fewer points
        points = cv2.approxPolyDP(points.reshape(-1, 1, 2), 0.01 * perimeter, closed=True).reshape(-1, 2)

    # Find the convex hull of the points
    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)

    # Calculate the area of the convex hull
    hull_area = cv2.contourArea(hull.reshape(-1, 1, 2))

    # If the area is too small, return the original mask
    if hull_area < 100:
        return mask

    # The convex hull removes concavities but preserves the overall shape
    # This is useful for utensils which generally have simple shapes

    # For more complex shapes where we want to preserve significant concavities,
    # we can use alpha shapes or concave hulls, but we'll use convex hull for simplicity

    # The convex hull gives us the main shape without outliers
    if hull.shape[0] < mask.shape[0] * 0.5:
        # If we've lost more than half the points, the shape might be too complex for a convex hull
        # In this case, revert to the original mask
        return mask

    if hull.shape[0] < mask.shape[0]:
        print(f"Removed {mask.shape[0] - hull.shape[0]} outlier points from contour mask")

    return hull


def get_median_depth(obj_mask: Optional[np.ndarray], depth_map: np.ndarray) -> Optional[float]:
    """
    Calculate median depth for an object mask.

    Parameters
    ----------
    obj_mask : np.ndarray or None
        (N, 2) array of pixel coordinates for the object mask, where each row
        contains [x, y] coordinates. Can be None.
    depth_map : np.ndarray
        (H, W) array representing the depth map with depth values per pixel.

    Returns
    -------
    float or None
        The median depth value for the object mask, or None if the mask is
        invalid or has fewer than 5 points.
    """
    if obj_mask is None or obj_mask.shape[0] < 5:
        return None

    h, w = depth_map.shape
    xs = np.clip(obj_mask[:, 0].astype(int), 0, w - 1)
    ys = np.clip(obj_mask[:, 1].astype(int), 0, h - 1)
    depths = depth_map[ys, xs]

    # Filter out zero or negative depths
    valid_depths = depths[depths > 0]
    if len(valid_depths) < 5:
        return None

    return float(np.median(valid_depths))


def fit_ellipse_to_mask(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Fit an ellipse to a mask and return the ellipse parameters.

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates for the mask

    Returns
    -------
    Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]
        ((center_x, center_y), (major_radius, minor_radius), angle)
        or None if fitting fails
    """
    if mask.shape[0] < 5:  # Need at least 5 points for ellipse fitting
        return None

    # Fit an ellipse to the mask points
    points = mask.astype(np.float32)

    try:
        # OpenCV's ellipse fitting requires at least 5 points
        ellipse = cv2.fitEllipse(points)

        # Extract center, axes, and angle from the fitted ellipse
        (center_x, center_y), (axis_1, axis_2), angle = ellipse

        # Ensure we have major and minor axes correctly ordered
        major_radius = max(axis_1, axis_2) / 2
        minor_radius = min(axis_1, axis_2) / 2

        return ((center_x, center_y), (major_radius, minor_radius), angle)

    except Exception as e:
        print(f"Error fitting ellipse: {e}")
        print("Falling back to bounding box method")

        # Fallback to bounding box method
        min_x, min_y = np.min(mask, axis=0)
        max_x, max_y = np.max(mask, axis=0)

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        major_radius = max((max_x - min_x), (max_y - min_y)) / 2
        minor_radius = min((max_x - min_x), (max_y - min_y)) / 2

        # Construct equivalent return value
        return ((center_x, center_y), (major_radius, minor_radius), 0.0)


def fit_plane_to_points(points: torch.Tensor) -> torch.Tensor:
    """
    Fit a plane to a set of 3D points using least squares.

    ═════ Fix: Correct plane equation fitting ═════
    We fit z = ax + by + c, then convert to ax + by - z + c = 0 format.

    Parameters
    ----------
    points : torch.Tensor
        (N, 3) tensor - The 3D points (x, y, z) to fit the plane to.

    Returns
    -------
    torch.Tensor
        (4,) tensor - The plane parameters [a, b, c, d] of ax + by + cz + d = 0.
    """
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    # Build the design matrix to solve z = a*x + b*y + c
    # A @ [a, b, c] = z
    A = torch.cat([
        points[:, 0:2],  # x, y columns
        torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    ], dim=1)  # shape (N, 3)

    z = points[:, 2]  # shape (N,)

    # Solve least squares for [a, b, c] where z = ax + by + c
    try:
        sol = torch.linalg.lstsq(A, z.unsqueeze(1)).solution.squeeze()
    except RuntimeError:
        # Fallback to pseudoinverse if lstsq fails
        sol = torch.pinverse(A) @ z.unsqueeze(1)
        sol = sol.squeeze()

    a, b, c = sol

    # Convert z = ax + by + c to standard form ax + by - z + c = 0
    # So plane_params = [a, b, -1, c]
    plane_params = torch.tensor([a, b, -1.0, c], device=points.device, dtype=points.dtype)

    return plane_params


def plane_axes_and_centroid(plane_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given plane_params = [a, b, c, d] for ax + by + cz + d = 0,
    returns the centroid point and three orthonormal vectors in camera coords:
      - centroid: the point on the plane closest to the origin
      - n_unit: the plane normal [a, b, c] normalized
      - e1: one in‐plane axis (≈ "plane length")
      - e2: the other in‐plane axis (≈ "plane width")

    These let you build a rotation matrix between table‐frame and camera‐frame.

    Parameters
    ----------
    plane_params : torch.Tensor
        Tensor of shape (4,) containing plane parameters [a, b, c, d]
        for the equation ax + by + cz + d = 0.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - centroid: The centroid point on the plane closest to origin (3,)
        - n_unit: The normalized plane normal vector (3,)
        - e1: First in-plane orthonormal axis vector (3,)
        - e2: Second in-plane orthonormal axis vector (3,)
    """
    a, b, c, d = plane_params

    # ─── Calculate plane normal ───
    n = torch.tensor([a, b, c], device=plane_params.device, dtype=plane_params.dtype)
    n_norm = torch.norm(n)

    # Handle degenerate case
    if n_norm < 1e-8:
        raise ValueError("Degenerate plane normal vector")

    n_unit = n / n_norm

    # ─── Calculate centroid (point on plane closest to origin) ───
    # For plane ax + by + cz + d = 0, the closest point to origin is:
    # centroid = -d * [a, b, c] / (a² + b² + c²)
    norm_squared = torch.sum(n * n)  # a² + b² + c²
    centroid = -d * n / norm_squared

    # ─── Calculate in-plane orthonormal axes ───
    # Pick a reference direction and cross to get e1
    z_glob = torch.tensor([0.0, 0.0, 1.0], device=plane_params.device, dtype=plane_params.dtype)
    e1 = torch.cross(n_unit, z_glob)

    if torch.norm(e1) < 1e-6:
        # If plane is nearly vertical (normal ≈ Z), use X instead
        x_glob = torch.tensor([1.0, 0.0, 0.0], device=plane_params.device, dtype=plane_params.dtype)
        e1 = torch.cross(n_unit, x_glob)

    e1 = e1 / torch.norm(e1)

    # The second in‐plane axis orthogonal to both
    e2 = torch.cross(n_unit, e1)
    e2 = e2 / torch.norm(e2)

    return centroid, n_unit, e1, e2


def compute_depth_consistency(plates: List[str],
                             utensils: List[str],
                             masks: Dict[str, np.ndarray],
                             depth_map: np.ndarray,
                             debug_info: bool = False) -> Dict[str, Any]:
    """
    Calculate depth consistency between plates and utensils.

    This function analyzes the depth values of plates and utensils to determine
    how consistent their depth measurements are relative to each other. A high
    consistency score indicates that objects at similar depths in the scene
    have similar depth values, while a low score suggests depth measurement
    inconsistencies that may affect focal length estimation accuracy.

    Parameters
    ----------
    plates : List[str]
        List of plate object keys corresponding to entries in the masks dictionary.
    utensils : List[str]
        List of utensil object keys corresponding to entries in the masks dictionary.
    masks : Dict[str, np.ndarray]
        Dictionary mapping object keys to their corresponding mask arrays, where
        each mask is an (N, 2) array of pixel coordinates.
    depth_map : np.ndarray
        2D array representing the depth map with depth values per pixel.
    debug_info : bool, default=False
        Whether to print debug information during processing.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing depth consistency analysis results with the following keys:
        - 'overall_consistency' : float
            Overall consistency score between 0.0 and 1.0, where 1.0 indicates
            perfect consistency and values approaching 0.0 indicate inconsistency.
        - 'comparisons' : List[Dict[str, Any]]
            List of dictionaries, each containing comparison data for a plate-utensil
            pair with keys: 'objects', 'depths', 'consistency', 'penalized_consistency'.
        - 'plate_depths' : Dict[str, float]
            Dictionary mapping plate keys to their median depth values.
        - 'utensil_depths' : Dict[str, float]
            Dictionary mapping utensil keys to their median depth values.
    """
    if not plates or not utensils:
        return {"overall_consistency": 1.0, "comparisons": [], "plate_depths": {}, "utensil_depths": {}}

    # ═════ Calculate median depths for all objects ═════
    plate_depths = {}
    for plate_key in plates:
        if plate_key in masks:
            median_depth = get_median_depth(masks[plate_key], depth_map)
            if median_depth is not None:
                plate_depths[plate_key] = median_depth

    utensil_depths = {}
    for utensil_key in utensils:
        if utensil_key in masks:
            median_depth = get_median_depth(masks[utensil_key], depth_map)
            if median_depth is not None:
                utensil_depths[utensil_key] = median_depth

    # If we couldn't get depths for any plates or utensils, return perfect consistency
    if not plate_depths or not utensil_depths:
        return {
            "overall_consistency": 1.0,
            "comparisons": [],
            "plate_depths": plate_depths,
            "utensil_depths": utensil_depths
        }

    # ═════ Compare all plates with all utensils ═════
    comparisons = []
    consistency_scores = []

    for plate_key, plate_depth in plate_depths.items():
        for utensil_key, utensil_depth in utensil_depths.items():
            # Calculate consistency score (1.0 = perfect consistency, approaches 0 for inconsistent)
            ratio = min(plate_depth, utensil_depth) / max(plate_depth, utensil_depth)

            # Apply additional penalty for extreme inconsistency
            penalized_ratio = ratio ** 2  # Square to more aggressively penalize

            comparisons.append({
                "objects": (plate_key, utensil_key),
                "depths": (plate_depth, utensil_depth),
                "consistency": ratio,
                "penalized_consistency": penalized_ratio
            })
            consistency_scores.append(ratio)

    # Calculate overall consistency as weighted average
    # Weight by the depth of the objects (deeper objects have more impact)
    if consistency_scores:
        overall_consistency = min(consistency_scores)  # Most conservative approach: use worst consistency
    else:
        overall_consistency = 1.0

    return {
        "overall_consistency": overall_consistency,
        "comparisons": comparisons,
        "plate_depths": plate_depths,
        "utensil_depths": utensil_depths
    }


def project_points_to_plane(points: torch.Tensor, plane_params: torch.Tensor) -> torch.Tensor:
    """
    Projects 3D points onto a plane defined by plane parameters.

    Parameters
    ----------
    points : torch.Tensor
        Tensor of shape (N, 3) representing N 3D points.
    plane_params : torch.Tensor
        Tensor of shape (4,) representing plane parameters [a, b, c, d]
        for ax + by + cz + d = 0.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) containing the projected points on the plane.
    """
    a, b, c, d = plane_params
    normal = torch.tensor([a, b, c], device=points.device, dtype=points.dtype)
    normal = normal / torch.norm(normal)  # Normalize

    # Distance from each point to the plane
    distances = (torch.sum(points * normal, dim=1) + d) / torch.norm(normal)

    # Project points onto the plane
    projected_points = points - distances.unsqueeze(1) * normal.unsqueeze(0)

    return projected_points


def compute_parametric_circle_uv(
    c: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    radius: float,
    f_px: float,
    cx: float,
    cy: float,
    num_points: int = 360
) -> Tuple[np.ndarray, float, float]:
    """
    Generate the 3D parametric circle in the fitted plane, project to (u,v),
    and compute pixel extents.

    Parameters
    ----------
    c : np.ndarray
        (3,) array - Plane centroid in camera coords.
    e1, e2 : np.ndarray
        (3,) arrays - Orthonormal basis vectors spanning the plane.
    radius : float
        Circle radius in meters (D/2).
    f_px : float
        Focal length in pixels.
    cx, cy : float
        Principal point offsets in pixels.
    num_points : int, default=360
        Number of samples around the circle.

    Returns
    -------
    Tuple[np.ndarray, float, float]
        - uv_curve: (num_points, 2) int array - Pixel coordinates of the reprojected circle.
        - du: float - Width of the curve in pixels (max(u) − min(u)).
        - dv: float - Height of the curve in pixels (max(v) − min(v)).
    """
    # 1) sample angles
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    # 2) build 3D circle points in camera coords
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    C3d = (
        c[np.newaxis, :] +
        radius * (cos_thetas[:, None] * e1[np.newaxis, :] +
                  sin_thetas[:, None] * e2[np.newaxis, :])
    )

    # 3) project each point to pixel coords
    X, Y, Z = C3d[:, 0], C3d[:, 1], C3d[:, 2]

    # Safety check for division by zero
    Z_safe = np.where(np.abs(Z) < 1e-8, 1e-8, Z)

    u = f_px * X / Z_safe + cx
    v = f_px * Y / Z_safe + cy
    uv_curve = np.stack([u, v], axis=1).astype(np.int32)

    # 4) compute extents
    du = float(uv_curve[:, 0].max() - uv_curve[:, 0].min())
    dv = float(uv_curve[:, 1].max() - uv_curve[:, 1].min())

    return uv_curve, du, dv


def compute_true_XY(clouds: Dict[str, np.ndarray],
                   plane_key: str = "plate",
                   known_plate_diameter: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """
    Given a dict of point‐clouds (in camera coords), and the orthonormal vectors for a plane,
    project *all* clouds into that plane to get local X, Y, and diagonal measurements.

    Parameters
    ----------
    clouds : Dict[str, np.ndarray]
        Dictionary of point clouds
    plane_key : str, default="plate"
        Key for the reference plane in the clouds dictionary
    known_plate_diameter : float, optional
        Known diameter of the plate in meters, used for validation

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary mapping each object key to its measurements:
        - 'x': The true width (in meters) along the first plane axis
        - 'y': The true width (in meters) along the second plane axis
        - 'diag': The true diagonal length (in meters) in the plane
    """
    results = {}

    # ═════ Fit plane to reference object and project all clouds ═════
    if plane_key not in clouds:
        raise ValueError(f"❗ ERROR: Plane key '{plane_key}' not found in processed clouds")

    # 1) Grab plate points and fit its plane
    plate_pc = clouds[plane_key]  # (N, 3) numpy

    # Convert to tensor for processing
    if isinstance(plate_pc, np.ndarray):
        plate_tensor = torch.tensor(plate_pc, dtype=torch.float32)
    else:
        plate_tensor = plate_pc

    plane_params = fit_plane_to_points(plate_tensor)

    # 2) Get the plane normal and orthonormal axes
    centroid, plane_normal, e1, e2 = plane_axes_and_centroid(plane_params)

    # Ensure all tensors are on the same device
    device = centroid.device

    for key, points in clouds.items():
        # Convert numpy to torch if needed
        if isinstance(points, np.ndarray):
            points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        else:
            points_tensor = points.to(device)

        # ─── Project onto orthonormal basis vectors ───
        # Compute local coordinates (u, v)
        u = torch.sum(points_tensor * e1, dim=1)  # (N,)
        v = torch.sum(points_tensor * e2, dim=1)  # (N,)

        # ─── Compute true extents ───
        x_true = float(torch.max(u) - torch.min(u))  # max(u) - min(u)
        y_true = float(torch.max(v) - torch.min(v))  # max(v) - min(v)
        diag = float(torch.sqrt(torch.tensor(x_true**2 + y_true**2, device=device)))

        # Build results dictionary
        results[key] = {'x': x_true, 'y': y_true, 'diag': diag}

        print(f"{key:>10s} → True X: {x_true:.3f} m, "
              f"True Y: {y_true:.3f} m,  Diag: {diag:.3f} m")

    # ═════ Add validation against known plate diameter if available ═════
    if plane_key in results and known_plate_diameter is not None:
        measured_diameter = max(results[plane_key]['x'], results[plane_key]['y'])
        accuracy = (measured_diameter / known_plate_diameter) * 100
        print(f"\nValidation: Measured plate diameter: {measured_diameter:.3f}m vs "
              f"Known diameter: {known_plate_diameter:.3f}m")
        print(f"Measurement accuracy: {accuracy:.1f}% of expected diameter")

        # Flag significant discrepancies
        if abs(100 - accuracy) > 10:
            print(f"⚠️ WARNING: Measured plate diameter differs from expected by {abs(100-accuracy):.1f}%")

    return results


def compute_tilt_angle(normal: torch.Tensor) -> torch.Tensor:
    """
    Computes the tilt angle between the plane's normal and the z-axis.

    Parameters
    ----------
    normal : torch.Tensor
        Tensor of shape (3,) representing the plane normal vector.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the tilt angle in radians.
    """
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
    cos_theta = torch.dot(normal, z_axis) / torch.norm(normal)
    # Clamp to avoid numerical issues with acos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    tilt_angle = torch.acos(cos_theta)
    return tilt_angle


def project_masks_to_pointclouds(depth: np.ndarray,
                                masks: Dict[str, np.ndarray],
                                f_px: float,
                                plane_key: str,
                                cx: float,
                                cy: float,
                                device: str = 'cuda') -> Dict[str, np.ndarray]:
    """
    Projects each mask into an N×3 point cloud in meters based on a reference plane.

    ═════ Efficiency improvements: Vectorized operations ═════

    Parameters
    ----------
    depth : np.ndarray
        (H, W) array of depth values in meters
    masks : Dict[str, np.ndarray]
        Dictionary where each mask is an (N, 2) array of pixel coords [[x0,y0], [x1,y1], …]
    f_px : float
        Focal length in pixels
    plane_key : str
        Key of the plane in the masks dictionary
    cx : float
        Principal point x-coordinate in pixels
    cy : float
        Principal point y-coordinate in pixels
    device : str, default='cuda'
        The device ('cuda' for GPU or 'cpu')

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping object keys to metric point clouds for each object
    """
    H, W = depth.shape
    projected_clouds = {}

    # ═════ Check depth scale - warn if it seems out of reasonable range for meters ═════
    depth_median = float(np.median(depth[depth > 0]))
    if depth_median > 10.0:
        print(f"⚠️ WARNING: Median depth value ({depth_median:.2f}) seems large for metric units")
    elif depth_median < 0.1:
        print(f"⚠️ WARNING: Median depth value ({depth_median:.2f}) seems small for metric units")
    else:
        print(f"Depth values appear to be in reasonable metric range (median: {depth_median:.2f}m)")

    # Move the depth map to the specified device
    depth_t = torch.tensor(depth, device=device, dtype=torch.float32)

    # ─── Process all objects ───
    for key, m in masks.items():
        # Skip processing for items with keys containing 'fork' or 'knife'
        if 'knife' in key.lower() or 'fork' in key.lower():
            continue

        # Clean the mask to remove outlier points
        original_count = m.shape[0]
        if original_count < 10:  # Skip if mask has too few points
            continue

        cleaned_mask = remove_mask_outliers(m, distance_threshold=15)
        if cleaned_mask.shape[0] < 10:  # Skip if cleaned mask has too few points
            print(f"Skipping {key}: too few points after outlier removal")
            continue

        if original_count != cleaned_mask.shape[0]:
            print(f"Cleaned {key} mask: {original_count} → {cleaned_mask.shape[0]} points")

        m = cleaned_mask  # Use the cleaned mask for projection

        # ─── Convert pixel coordinates to 3D points ───
        # 1) Index vectors (int) and projection vectors (float)
        xs_i = np.clip(m[:, 0].astype(int), 0, W - 1)
        ys_i = np.clip(m[:, 1].astype(int), 0, H - 1)
        ix = torch.tensor(xs_i, device=device)
        iy = torch.tensor(ys_i, device=device)

        Z = depth_t[iy, ix]  # (N,)
        X = (ix.float() - cx) * Z / f_px
        Y = (iy.float() - cy) * Z / f_px
        mask_pts = torch.stack([X, Y, Z], dim=1)  # Shape (N, 3)

        projected_clouds[key] = mask_pts.cpu().numpy()

    return projected_clouds


def plot_point_clouds_view(clouds: Dict[str, np.ndarray],
                          elev: float = 30,
                          azim: float = 45,
                          point_size: float = 1.0,
                          interactive: bool = True) -> None:
    """
    Plot multiple point clouds in a single 3D axes with +Z up and +Y away from the viewer.

    Parameters
    ----------
    clouds : Dict[str, np.ndarray]
        Mapping from label -> (N_i, 3) arrays, where each array is [X, Y, Z]
        in camera coords (X right, Y down, Z into scene).
    elev : float, default=30
        Elevation angle in the z plane for initial view (degrees).
    azim : float, default=45
        Azimuth angle in the x,y plane for initial view (degrees).
    point_size : float, default=1.0
        Marker size for the scatter points.
    interactive : bool, default=True
        If True, enable interactive 3D rotation in Jupyter notebooks.
    """
    # Enable interactive plotting for Jupyter notebooks
    if interactive:
        try:
            # Try to use widget backend for interactive plots
            import matplotlib
            current_backend = matplotlib.get_backend()
            if 'widget' not in current_backend.lower():
                print("💡 For interactive 3D plotting, run: %matplotlib widget")
                print("   Then restart this cell for full interactivity")
        except ImportError:
            print("⚠️ For interactive plots, install: pip install ipympl")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Pick a qualitative colormap
    cmap = plt.get_cmap('tab10')
    labels = list(clouds.keys())
    num = len(labels)

    for i, label in enumerate(labels):
        pts = np.asarray(clouds[label])
        # Transform to view coords:
        #   X stays right,
        #   Y_view = Z_cam (so into scene → away from viewer),
        #   Z_view = -Y_cam (so down → up)
        Xv = pts[:, 0]
        Yv = pts[:, 2]  # Z becomes Y (positive = away from viewer)
        Zv = -pts[:, 1]  # Y becomes Z

        color = cmap(i % cmap.N)
        ax.scatter(
            Xv, Yv, Zv,
            s=point_size,
            c=[color],
            label=label,
            depthshade=False,
            alpha=0.8
        )

    ax.set_xlabel('X (m, right)')
    ax.set_ylabel('Y (m, away from viewer)')
    ax.set_zlabel('Z (m, up)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.view_init(elev=elev, azim=azim)

    # Add interactive instructions if enabled
    if interactive:
        ax.text2D(0.02, 0.98, "🖱️ Click and drag to rotate\n📏 Scroll to zoom",
                  transform=ax.transAxes, fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.show()


def plot_point_clouds_interactive_plotly(clouds: Dict[str, np.ndarray],
                                        point_size: float = 2.0,
                                        title: str = "3D Point Clouds") -> Optional[Any]:
    """
    Create an interactive 3D scatter plot using Plotly for superior interactivity.

    Parameters
    ----------
    clouds : Dict[str, np.ndarray]
        Mapping from label -> (N_i, 3) arrays, where each array is [X, Y, Z]
        in camera coords (X right, Y down, Z into scene).
    point_size : float, default=2.0
        Marker size for the scatter points.
    title : str, default="3D Point Clouds"
        Title for the plot.

    Returns
    -------
    Optional[Any]
        Interactive Plotly figure that can be displayed in Jupyter, or None if Plotly unavailable
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        print("⚠️ Plotly not installed. Install with: pip install plotly")
        print("   Falling back to matplotlib version...")
        plot_point_clouds_view(clouds, interactive=True)
        return None

    fig = go.Figure()

    # Get a color palette
    colors = px.colors.qualitative.Set1

    for i, (label, pts) in enumerate(clouds.items()):
        pts = np.asarray(pts)

        # Transform to view coords (same as matplotlib version)
        Xv = pts[:, 0]  # X stays the same
        Yv = pts[:, 2]  # Z becomes Y (positive = away from viewer)
        Zv = -pts[:, 1]  # -Y becomes Z (up)

        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter3d(
            x=Xv,
            y=Yv,
            z=Zv,
            mode='markers',
            marker=dict(
                size=point_size,
                color=color,
                opacity=0.8
            ),
            name=label,
            text=[f"{label}<br>X: {x:.3f}<br>Y: {y:.3f}<br>Z: {z:.3f}"
                  for x, y, z in zip(Xv, Yv, Zv)],
            hovertemplate="%{text}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m, right)',
            yaxis_title='Y (m, away from viewer)',
            zaxis_title='Z (m, up)',
            aspectmode='data'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
        width=800
    )

    # Add instructions as annotation
    fig.add_annotation(
        text="🖱️ Drag to rotate • 🖱️ Shift+drag to pan • 🔍 Scroll to zoom • 📏 Hover for coordinates",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="lightblue",
        bordercolor="blue",
        borderwidth=1,
        font=dict(size=10)
    )

    return fig

def plot_plane_fit(plane_points, plane_params, grid_steps=50):
    """
    Visualize a 3D point cloud (plane_points) and its fitted plane,
    with equal axis scaling so proportions are preserved.

    Parameters
    ----------
    plane_points : (N,3) array or torch.Tensor
        The 3D points in camera/world coords.
    plane_params : array-like of length 4
        [a, b, c, d] for the plane ax + by + cz + d = 0.
    grid_steps : int
        Resolution of the plane mesh.
    """
    # 1) To NumPy
    if hasattr(plane_points, 'cpu'):
        pts = plane_points.cpu().numpy()
    else:
        pts = np.asarray(plane_points)

    # 2) Unpack plane
    a, b, c, d = plane_params

    # 3) Build XY grid over data extents
    x_min, x_max = pts[:,0].min(), pts[:,0].max()
    y_min, y_max = pts[:,1].min(), pts[:,1].max()
    xs = np.linspace(x_min, x_max, grid_steps)
    ys = np.linspace(y_min, y_max, grid_steps)
    X, Y = np.meshgrid(xs, ys)

    # 4) Compute Z on plane: aX + bY + cZ + d = 0 → Z = -(aX + bY + d)/c
    Z = -(a*X + b*Y + d) / c

    # 5) Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # point cloud
    ax.scatter(pts[:,0], pts[:,1], pts[:,2],
               c='r', marker='o', s=5, label='Point Cloud')

    # plane surface
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')

    # labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud & Fitted Plane')
    ax.legend()

    # 6) Equal aspect ratio
    # compute maximum range across axes
    max_range = np.array([
        pts[:,0].ptp(),
        pts[:,1].ptp(),
        pts[:,2].ptp()
    ]).max() / 2.0

    # compute midpoints
    mid_x = 0.5 * (x_max + x_min)
    mid_y = 0.5 * (y_max + y_min)
    mid_z = pts[:,2].mean()

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

def visualize_circle_plane_fit(point_cloud: np.ndarray,
                       diagonal_size: float,
                       f_px: float,
                       input_image: Any,
                       cx: Optional[float] = None,
                       cy: Optional[float] = None) -> None:
    """
    Visualize the fitted plane by overlaying a reprojected circle onto the input image.

    This function takes a 3D point cloud, fits a plane to it, and projects a circle
    of the specified diagonal size back onto the image plane for visualization.

    Parameters
    ----------
    point_cloud : np.ndarray
        (N, 3) array of 3D points representing the object to fit a plane to.
    diagonal_size : float
        The diagonal size/diameter of the circle to project back onto the image plane (in meters).
    f_px : float
        Focal length in pixels.
    input_image : Any
        Input image (PIL Image or numpy array) to overlay the visualization on.
    cx : Optional[float], default=None
        Principal point x-coordinate in pixels. If None, uses image center.
    cy : Optional[float], default=None
        Principal point y-coordinate in pixels. If None, uses image center.

    Returns
    -------
    None
        Displays the visualization using OpenCV.
    """
    # Convert PIL image to numpy array if needed
    if hasattr(input_image, 'size'):  # PIL Image
        visualization = np.array(input_image)
        height, width = visualization.shape[:2]
    else:  # numpy array
        visualization = input_image.copy()
        height, width = visualization.shape[:2]

    # Set default principal point if not provided
    if cx is None:
        cx = float(width // 2)
    if cy is None:
        cy = float(height // 2)

    # 1) Grab plate points and fit its plane
    plane_params = fit_plane_to_points(torch.tensor(point_cloud))

    # 2) Get the plane normal and orthonormal axes
    centroid, plane_normal, e1, e2 = plane_axes_and_centroid(plane_params)

    # 3) Convert diagonal size to radius and compute parametric circle
    radius = diagonal_size / 2.0
    uv_curve, width_px, height_px = compute_parametric_circle_uv(
        centroid.cpu().numpy(),
        e1.cpu().numpy(),
        e2.cpu().numpy(),
        radius,
        f_px,
        cx,
        cy
    )

    # Draw the reprojected circle curve in red
    # Note: ensure uv_curve is in the right shape for polylines
    cv2.polylines(visualization, [uv_curve], isClosed=True, color=(0, 0, 255), thickness=2)

    # Display the visualization
    cv2.imshow("Plane Fit Visualization", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═════ Example Workflow ═════
# Example workflow to project segmentations to point clouds and get real world dimensions.
# Verify the results visually.
#
# Usage:
# plane_key = "plate"
# clouds_np = project_masks_to_pointclouds(depth, masks, f_px, plane_key, cx, cy)
# true_dims = compute_true_XY(clouds_np, plane_key, known_plate_diameter=0.25)
# visualize_circle_plane_fit(clouds_np[plane_key], true_dims[plane_key]['diag'], f_px, image, cx, cy)
