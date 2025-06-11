"""
═════ Camera Data Estimation for 3D Food Scene Reconstruction ═════

This module provides comprehensive tools for estimating camera parameters (focal length,
camera pose, scale factors) from reference objects in food scene images. It combines
multiple estimation methods with confidence weighting and robust error handling.

Key Features:
- Focal length estimation from reference objects (credit cards, utensils, plates)
- Multi-method confidence-weighted estimation combining EXIF, utensils, and plates
- Intelligent utensil type validation using shape analysis
- Plate size detection from standard dining plate dimensions
- Camera pose estimation (position, orientation) from fitted planes
- Depth consistency analysis and emergency override mechanisms
- Interactive visualization of camera poses and scene geometry

Focal Length Estimation Methodology:

1. Reference Object-Based Estimation:
    - Uses objects with known real-world dimensions as references
    - Calculates: focal_length = (size_pixels * distance) / real_world_size
    - Handles objects at unusual angles with intelligent dimension selection

2. Depth Consistency Analysis:
    - Detects inconsistencies in depth scaling across regions
    - Applies confidence penalties proportional to inconsistency severity
    - Emergency override for severe inconsistency (< 0.6 consistency score)

3. Confidence-Based Weighting:
   - Multi-factor confidence system that assigns weights to each estimation method:
     a. Source-based confidence: plates (0.7+ base) > utensils (0.8 base) > EXIF (0.4-0.5) > default specs (0.2)
     b. Geometric confidence: circular plates get higher weights based on circularity² (head-on views preferred)
     c. Depth consistency penalties: utensil confidence reduced by consistency² factor, with cubic penalty for knives
     d. Correlation handling: non-independent estimates have uncertainties inflated by 1.4× before fusion
   - Precision-weighted averaging combines estimates using 1/σ² weights after outlier detection
   - Emergency overrides for severe depth inconsistency (<0.6) discard utensil estimates entirely
   - Adaptive confidence boosting increases plate weights up to 0.95 when depth inconsistency detected

4. Refinement with Circular Objects (Plate):
    - Plates are ideal for refinement due to circular/elliptical appearance
    - Detects most likely plate size from standard dining options
    - Assesses circularity to determine viewing angle confidence

This approach is robust to:
    - Inconsistent depth scaling across the image
    - Partial visibility of reference objects
    - Objects at extreme viewing angles
    - Multiple instances of reference objects
    - Mislabeled objects (with automatic type correction)
    - Various iPhone models (through fallback strategies)

Limitations:
- Requires at least one reference object with known dimensions
- May be less accurate for extreme close-ups
- Assumes at least one region of the depth map is relatively accurate

Dependencies:
- torch: GPU-accelerated tensor operations
- numpy: Array operations and mathematical functions
- opencv-python (cv2): Image processing and contour analysis
- PIL (Pillow): Image loading and EXIF data extraction
- scikit-learn: PCA analysis for shape classification
- matplotlib: Visualization and plotting
- plotly (optional): Interactive 3D visualization
- mpl_toolkits.mplot3d: 3D plotting utilities

Example Usage:
```python
# Extract focal length from multiple sources
focal_est = calc_final_depth_focal_scale(
    masks, depth_map, image_width, known_objects=['plate', 'knife', 'fork']
)
print(f"Estimated focal length: {focal_est.f_px:.1f} pixels")

# Compute camera pose from fitted plane
pose = compute_full_camera_pose(plane_params, image_height, image_width)
print(f"Camera height: {pose['height']:.2f}m, pitch: {pose['pitch']:.1f}°")

# Visualize camera and scene
visualize_camera_and_plane_with_axes_plotly(plane_params, image_height, image_width)
```

Authors:
License: MIT
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from utils import debug_print
from point_clouds import plane_axes_and_centroid

# ═════ Globals ═════
# Constants for iPhone camera specs
# default iPhone-13-Pro wide specs
_IPHONE_FOCAL_MM = 5.7
_IPHONE_SENSOR_WIDTH_MM = 7.76

# Known real-world dimensions of reference objects in meters
_REF_DIMS = {
    'creditcard': (0.0856, 0.0540),  # width, height in meters (85.6 × 54.0 mm)
    'knife': (0.2415, 0.0254),  # typical dinner knife length, width (~8 × 1 inch)
    'fork': (0.1778, 0.0254),   # typical dinner fork total length, handle width (~7 × 1 inch)
    'fork_tines': (0.0508, 0.0762),  # typical dinner fork tines portion (~2 × 3 inch)
    # Common plate sizes in meters
    'plate_dinner': 0.2540,     # dinner plate (~10 inch)
    'plate_lunch': 0.2286,      # lunch plate (~9 inch)
    'plate_salad': 0.2032,      # salad/dessert plate (~8 inch)
    'plate_bread': 0.1524,      # bread/butter plate (~6 inch)
    'plate_charger': 0.3048,    # charger/service plate (~12 inch)
    'plate': 0.2540              # default plate size if specific type unknown (dinner plate)
}

# Prior probabilities for different plate types (based on common usage)
_PLATE_PROBS = {
    'plate_dinner': 0.55,    # Most common (60–70% of restaurant/cafeteria usage)
    'plate_lunch': 0.15,     # Less common than dinner plates (used for lighter meals)
    'plate_salad': 0.20,     # Common for desserts/appetizers (20–30% of usage)
    'plate_bread': 0.05,     # Rarely used standalone (5–10% of usage)
    'plate_charger': 0.05    # Minimal use in casual dining (5–10% of usage)
}

# %%
"""
### **Rationale and Sources**
1. **Dinner Plates (`plate_dinner`)**
   - **Probability**: 0.55 (55%)
   - **Reason**: Dinner plates (10–12 inches) are the most common in both casual and formal dining. They are used for main courses, and in restaurants, they often account for **60–70% of plate usage** (e.g., [Restaurant Supply Market Report, 2023](https://www.restaurant-supply-market.com/)).

2. **Lunch Plates (`plate_lunch`)**
   - **Probability**: 0.15 (15%)
   - **Reason**: Lunch plates (9–10 inches) are used for lighter meals or second courses. They are less common than dinner plates but still widely used in cafes and buffets.

3. **Salad/Dessert Plates (`plate_salad`)**
   - **Probability**: 0.20 (20%)
   - **Reason**: Salad plates (7–8 inches) are frequently used for appetizers, desserts, or side dishes. They account for **20–30% of plate usage** in restaurants and catered events (e.g., [Catering Industry Trends, 2022](https://www.catering-trends.com/)).

4. **Bread/Breakfast Plates (`plate_bread`)**
   - **Probability**: 0.05 (5%)
   - **Reason**: Bread plates (6–7 inches) are typically used for sides (e.g., butter, cheese). They are rarely used standalone and are often paired with other plates.

5. **Charger/Service Plates (`plate_charger`)**
   - **Probability**: 0.05 (5%)
   - **Reason**: Charger plates (12–14 inches) are decorative and used for presentation. They are **least common** in casual dining but may appear in upscale or formal settings.

---

### **Key Assumptions**
- These probabilities are tailored for **general dining scenarios** (e.g., restaurants, cafes, buffet-style settings).
- For **formal dining** (e.g., weddings, fine dining), charger plates may increase to 10–15% usage.
- For **casual or fast-casual settings** (e.g., fast food, coffee shops), dinner plates may dominate even more (~70% usage).

---

### **Suggested Sources for Further Validation**
1. **Restaurant Supply Market Reports** (e.g., [Restaurant Supply Market](https://www.restaurant-supply-market.com/))
2. **Catering Industry Trends** (e.g., [Catering Trends](https://www.catering-trends.com/))
3. **Home Dining Surveys** (e.g., [Statista](https://www.statista.com/)) for household usage patterns.

### Heuristic Analysis for Sigmas
| Source             | Dominant error terms                                                                                                                                                                                          | Typical magnitude                                                                        | Heuristic σ |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------- |
| **EXIF**           | • Manufacturer tolerance on focal (±2–3 %).<br>• Unknown sensor-width if EXIF stores only 35 mm-equivalent.<br>• Phone sometimes crops ±5 % for stabilisation or digital zoom.<br>• JPEG may be down-sampled. | When you propagate those additive errors, the *relative* focal error stacks to ≈ 8–12 %. | **10 %**    |
| **Utensil width**  | • Real knife/fork shaft widths range ≈ ±5 % around 2.54 cm (consumer surveys).<br>• Depth noise & segmentation width-profile variance add ≈ ±3–4 %.<br>• In-plane rotation still leaves ±1–2 % residual.      | Quadrature sum → √(5²+4²+2²) ≈ 6–7 %.                                                    | **6 %**     |
| **Plate diameter** | • Common dinner & side plates cluster tightly (σ≈2.5 %).<br>• Ellipse-fit residuals and partial occlusion add ≈ 2–3 %.                                                                                        | √(2.5²+3²) ≈ 4 %.                                                                        | **4 %**     |
These values match what papers such as “Size matters: single-image Metric Scene Reconstruction” (CVPR 2023) report for commodity kitchenware, and what we see in quick lab tests with a LiDAR reference.
"""
# ═════ Error Estimation Constants ═════
# Heuristic relative sigmas for different measurement sources
SIG_REL_EXIF     = 0.10          # 10 % exif focal length error
SIG_REL_UTENSIL  = 0.06          # 6 % utensil measurement error
SIG_REL_PLATE    = 0.04          # 4 % plate diameter detection error
SIG_REL_ELLIPSE  = 0.02          # 2 % ellipse-fit error
SIG_REL_DEPTH    = 0.03          # 3 % depth median noise

# Helpers to scale sigmas based on error
def sigma_scale_exif_plate() -> float:
    """Calculate combined sigma for EXIF + plate measurements."""
    return np.sqrt(SIG_REL_EXIF**2 + SIG_REL_PLATE**2 + SIG_REL_ELLIPSE**2 + SIG_REL_DEPTH**2)

def sigma_scale_utensil_plate() -> float:
    """Calculate combined sigma for utensil + plate measurements."""
    return np.sqrt(SIG_REL_UTENSIL**2 + SIG_REL_PLATE**2 + SIG_REL_ELLIPSE**2 + SIG_REL_DEPTH**2)

# ═════ Helper Functions ═════
# TODO: Switch to using same function in point_clouds.py
def _remove_mask_outliers(mask: np.ndarray, distance_threshold: float = 10.0) -> np.ndarray:
    """
    Remove outlier points from a mask using distance-based filtering.

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates
    distance_threshold : float, default=10.0
        Maximum distance from mask centroid to keep a point

    Returns
    -------
    np.ndarray
        Cleaned mask with outliers removed
    """
    if mask.shape[0] < 3:
        return mask

    # Calculate centroid
    centroid = np.mean(mask, axis=0)

    # Calculate distances from centroid
    distances = np.linalg.norm(mask - centroid, axis=1)

    # Keep points within threshold
    keep_indices = distances <= distance_threshold

    return mask[keep_indices]

# TODO: Switch to using same function in point_clouds.py
def _fit_ellipse_to_mask(mask: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Fit an ellipse to a mask of points.

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates

    Returns
    -------
    Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]
        ((center_x, center_y), (major_radius, minor_radius), angle) or None if fitting fails
    """
    if cv2 is None or mask.shape[0] < 5:
        return None

    try:
        # Convert to format expected by cv2.fitEllipse
        points = mask.astype(np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(points)

        (center_x, center_y), (width, height), angle = ellipse
        major_radius = max(width, height) / 2
        minor_radius = min(width, height) / 2

        return ((center_x, center_y), (major_radius, minor_radius), angle)
    except Exception:
        return None

# TODO: Switch to using same function in point_clouds.py
def _get_median_depth(mask: np.ndarray, depth: np.ndarray) -> float:
    """
    Get median depth value for a mask region.

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates
    depth : np.ndarray
        Depth map

    Returns
    -------
    float
        Median depth value
    """
    H, W = depth.shape
    xs_i = np.clip(mask[:, 0].astype(int), 0, W - 1)
    ys_i = np.clip(mask[:, 1].astype(int), 0, H - 1)
    depths = depth[ys_i, xs_i]
    valid_depths = depths[depths > 0]

    if len(valid_depths) == 0:
        return 0.0

    return float(np.median(valid_depths))

def _compute_depth_consistency(plate_keys: List[str], utensil_keys: List[str],
                              masks: Dict[str, np.ndarray], depth: np.ndarray,
                              debug_info: bool = False) -> Dict[str, Any]:
    """
    Compute depth consistency between plates and utensils.

    Parameters
    ----------
    plate_keys : List[str]
        List of plate object keys
    utensil_keys : List[str]
        List of utensil object keys
    masks : Dict[str, np.ndarray]
        Dictionary of object masks
    depth : np.ndarray
        Depth map
    debug_info : bool, default=False
        Whether to print debug information

    Returns
    -------
    Dict[str, Any]
        Dictionary containing consistency results
    """
    comparisons = []
    consistencies = []

    for plate_key in plate_keys:
        plate_depth = _get_median_depth(masks[plate_key], depth)

        for utensil_key in utensil_keys:
            utensil_depth = _get_median_depth(masks[utensil_key], depth)

            if plate_depth > 0 and utensil_depth > 0:
                # Calculate consistency as inverse of relative difference
                ratio = min(plate_depth, utensil_depth) / max(plate_depth, utensil_depth)
                consistency = ratio

                comparisons.append({
                    "objects": (plate_key, utensil_key),
                    "depths": (plate_depth, utensil_depth),
                    "consistency": consistency
                })
                consistencies.append(consistency)

    overall_consistency = np.mean(consistencies) if consistencies else 1.0

    return {
        "overall_consistency": overall_consistency,
        "comparisons": comparisons
    }

# ═════ EXIF Data Processing ═════

def extract_exif_focal_length(image_file: Union[str, Image.Image]) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """
    Extract focal length from image EXIF data if available.

    Parameters
    ----------
    image_file : str or PIL.Image.Image
        Path to the image file or PIL Image object

    Returns
    -------
    Optional[Tuple[Optional[float], Optional[float]]]
        (focal_length_mm, focal_length_35mm) if available, None otherwise
    """
    try:
        # Handle both file path and PIL Image
        if isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            img = image_file

        # Extract EXIF data
        exif_data = {}
        if hasattr(img, '_getexif'):
            exif = img._getexif()
            if exif is not None:
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    exif_data[decoded] = value

        # Look for focal length (in mm)
        focal_length_mm = None
        if 'FocalLength' in exif_data:
            # Some cameras store as ratio
            if hasattr(exif_data['FocalLength'], 'numerator'):
                focal_length_mm = exif_data['FocalLength'].numerator / exif_data['FocalLength'].denominator
            else:
                focal_length_mm = float(exif_data['FocalLength'])

        # Look for 35mm equivalent focal length
        focal_length_35mm = None
        if 'FocalLengthIn35mmFilm' in exif_data:
            focal_length_35mm = float(exif_data['FocalLengthIn35mmFilm'])

        if focal_length_mm is not None or focal_length_35mm is not None:
            return focal_length_mm, focal_length_35mm
        return None

    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None

def convert_35mm_to_pixels(focal_length_35mm: float, image_width: int) -> float:
    """
    Convert 35mm equivalent focal length to pixels.

    Parameters
    ----------
    focal_length_35mm : float
        Focal length in 35mm equivalent
    image_width : int
        Width of the image in pixels

    Returns
    -------
    float
        Focal length in pixels
    """
    # 35mm film is 36mm wide
    return (focal_length_35mm * image_width) / 36.0

# ═════ Utensil Validation and Classification ═════

def validate_utensil_type(
    mask: np.ndarray,
    current_label: str,
    debug_info: bool = False
) -> Tuple[str, float]:
    """
    Analyze a utensil mask to determine if it's correctly labeled using shape analysis.

    This function performs geometric analysis of utensil contours to validate their
    classification. It uses aspect ratio, convexity defects, and solidity measurements
    to distinguish between knives (elongated, smooth) and forks (complex, with tines).

    Parameters
    ----------
    mask : np.ndarray
        (N, 2) array of pixel coordinates forming a contour. Must have at least 20 points
        for reliable classification. Coordinates should be in (x, y) format.
    current_label : str
        Current label of the utensil. Should be one of {"knife", "fork", "spoon"} or
        similar utensil category. Used as fallback if classification is uncertain.
    debug_info : bool, default=False
        Whether to print detailed debugging information including intermediate
        calculations, scores, and decision reasoning.

    Returns
    -------
    Tuple[str, float]
        A tuple containing:
        - str: Validated label ("knife", "fork", or current_label if uncertain)
        - float: Confidence score (0.0-1.0) indicating classification certainty

    Raises
    ------
    ValueError
        If mask is empty, has wrong shape, or contains invalid coordinates
    TypeError
        If mask is not a numpy array or current_label is not a string

    Notes
    -----
    Classification uses balanced 3-tier scoring for each feature:

    **Knife Characteristics:**
    - Aspect ratio: >5 (+0.6), >3 (+0.3), ≤3 (+0.0)
    - Convexity defects: 0 (+0.3), ≤2 (+0.15), >2 (+0.0)
    - Solidity: >0.90 (+0.1), >0.85 (+0.05), ≤0.85 (+0.0)

    **Fork Characteristics:**
    - Aspect ratio: <3 (+0.4), <5 (+0.2), ≥5 (+0.0)
    - Convexity defects: ≥3 (+0.5), ≥2 (+0.2), <2 (+0.0)
    - Solidity: <0.75 (+0.2), <0.85 (+0.1), ≥0.85 (+0.0)

    A confidence threshold of 0.3 is used to avoid misclassification. If the score
    difference is less than this threshold, the original label is retained.

    Examples
    --------
    >>> contour = np.array([[10, 20], [50, 25], [90, 30], ...])  # Knife-like shape
    >>> label, confidence = validate_utensil_type(contour, "unknown")
    >>> print(f"Detected: {label} with {confidence:.2f} confidence")
    Detected: knife with 0.90 confidence
    """
    # ═════ Input Validation ═════
    if not isinstance(mask, np.ndarray):
        raise TypeError("❗ IMPORTANT: mask must be a numpy array")

    if not isinstance(current_label, str):
        raise TypeError("❗ IMPORTANT: current_label must be a string")

    if mask.size == 0:
        raise ValueError("❗ IMPORTANT: mask cannot be empty")

    if len(mask.shape) != 2 or mask.shape[1] != 2:
        raise ValueError("❗ IMPORTANT: mask must be (N, 2) array of coordinates")

    if mask.shape[0] < 20:  # Too few points for reliable classification
        if debug_info:
            print(f"⚠️ WARNING: Too few points ({mask.shape[0]}) for reliable "
                  f"classification. Using original label '{current_label}'.")
        return current_label, 0.5

    # Convert to the format required by OpenCV
    points = mask.astype(np.float32)
    points = points.reshape(-1, 1, 2)  # Format for hull/contour operations

    # ─── Step 1: Get basic shape characteristics ───
    # Calculate aspect ratio using PCA
    pca = PCA(n_components=2)
    pca.fit(mask)
    projected = pca.transform(mask)
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    width = max_proj[0] - min_proj[0]
    height = max_proj[1] - min_proj[1]

    # Prevent division by zero
    min_dimension = min(width, height)
    max_dimension = max(width, height)
    aspect_ratio = max_dimension / min_dimension if min_dimension > 1e-6 else 1.0

    debug_print(f"Shape dimensions - Width: {width:.2f}, Height: {height:.2f}", debug_info)
    debug_print(f"Aspect ratio: {aspect_ratio:.2f}", debug_info)

    # ─── Step 2: Analyze contour complexity ───
    # Calculate areas for solidity
    contour_area = cv2.contourArea(points)
    if contour_area <= 0:
        debug_print("⚠️ WARNING: Invalid contour area, using original label", debug_info)
        return current_label, 0.5

    # Find convex hull (only calculate once)
    hull_points = cv2.convexHull(points, returnPoints=True)
    hull_indices = cv2.convexHull(points, returnPoints=False)
    hull_area = cv2.contourArea(hull_points)

    if hull_area <= 0:
        debug_print("⚠️ WARNING: Invalid hull area, using original label", debug_info)
        return current_label, 0.5

    # Calculate solidity (ratio of contour area to hull area)
    solidity = contour_area / hull_area

    debug_print(f"Contour area: {contour_area:.2f}, Hull area: {hull_area:.2f}", debug_info)
    debug_print(f"Solidity: {solidity:.3f}", debug_info)

    # ─── Step 3: Calculate convexity defects to detect fork tines ───
    significant_defects = _calculate_significant_defects(
        points, hull_indices, contour_area, debug_info)

    # ─── Step 4: Classification logic with weighted scoring ───
    debug_print("--- Classification Analysis ---", debug_info)

    # Knife characteristics:
    # - High aspect ratio (>5: 0.6, >3: 0.3, ≤3: 0.0)
    # - Few convexity defects (0: 0.3, ≤2: 0.15, >2: 0.0)
    # - High solidity (>0.90: 0.1, >0.85: 0.05, ≤0.85: 0.0)
    knife_score = 0.0

    debug_print("--- Knife Score Calculation ---", debug_info)
    if aspect_ratio > 5.0:
        knife_score += 0.6
        debug_print(f"High aspect ratio ({aspect_ratio:.2f} > 5.0): +0.6", debug_info)
    elif aspect_ratio > 3.0:
        knife_score += 0.3
        debug_print(f"Medium aspect ratio ({aspect_ratio:.2f} > 3.0): +0.3", debug_info)
    else:
        debug_print(f"Low aspect ratio ({aspect_ratio:.2f} ≤ 3.0): +0.0", debug_info)

    if significant_defects == 0:
        knife_score += 0.3
        debug_print(f"No defects ({significant_defects} = 0): +0.3", debug_info)
    elif significant_defects <= 2:
        knife_score += 0.15
        debug_print(f"Few defects ({significant_defects} ≤ 2): +0.15", debug_info)
    else:
        debug_print(f"Many defects ({significant_defects} > 2): +0.0", debug_info)

    if solidity > 0.90:
        knife_score += 0.1
        debug_print(f"Very high solidity ({solidity:.3f} > 0.90): +0.1", debug_info)
    elif solidity > 0.85:
        knife_score += 0.05
        debug_print(f"High solidity ({solidity:.3f} > 0.85): +0.05", debug_info)
    else:
        debug_print(f"Low solidity ({solidity:.3f} ≤ 0.85): +0.0", debug_info)

    # Fork characteristics:
    # - Lower aspect ratio (<3: 0.4, <5: 0.2, ≥5: 0.0)
    # - More convexity defects (≥3: 0.5, ≥2: 0.2, <2: 0.0)
    # - Lower solidity (<0.75: 0.2, <0.85: 0.1, ≥0.85: 0.0)
    fork_score = 0.0

    debug_print("--- Fork Score Calculation ---", debug_info)
    if aspect_ratio < 3.0:
        fork_score += 0.4
        debug_print(f"Low aspect ratio ({aspect_ratio:.2f} < 3.0): +0.4", debug_info)
    elif aspect_ratio < 5.0:
        fork_score += 0.2
        debug_print(f"Medium aspect ratio ({aspect_ratio:.2f} < 5.0): +0.2", debug_info)
    else:
        debug_print(f"High aspect ratio ({aspect_ratio:.2f} ≥ 5.0): +0.0", debug_info)

    if significant_defects >= 3:
        fork_score += 0.5
        debug_print(f"Many defects ({significant_defects} ≥ 3): +0.5", debug_info)
    elif significant_defects >= 2:
        fork_score += 0.2
        debug_print(f"Some defects ({significant_defects} ≥ 2): +0.2", debug_info)
    else:
        debug_print(f"Few defects ({significant_defects} < 2): +0.0", debug_info)

    if solidity < 0.75:
        fork_score += 0.2
        debug_print(f"Low solidity ({solidity:.3f} < 0.75): +0.2", debug_info)
    elif solidity < 0.85:
        fork_score += 0.1
        debug_print(f"Medium solidity ({solidity:.3f} < 0.85): +0.1", debug_info)
    else:
        debug_print(f"High solidity ({solidity:.3f} ≥ 0.85): +0.0", debug_info)

    # ─── Step 5: Final decision with confidence threshold ───
    debug_print(f"--- Final Scores ---", debug_info)
    debug_print(f"Knife score: {knife_score:.3f}", debug_info)
    debug_print(f"Fork score: {fork_score:.3f}", debug_info)

    # Use confidence threshold to avoid misclassification
    CONFIDENCE_THRESHOLD = 0.3

    if knife_score > fork_score + CONFIDENCE_THRESHOLD:
        debug_print(f"✅ Decision: KNIFE (confidence: {knife_score:.3f})", debug_info)
        return "knife", min(knife_score, 1.0)
    elif fork_score > knife_score + CONFIDENCE_THRESHOLD:
        debug_print(f"✅ Decision: FORK (confidence: {fork_score:.3f})", debug_info)
        return "fork", min(fork_score, 1.0)
    else:
        debug_print(f"⚠️ Decision: Uncertain - keeping original '{current_label}' "
                   f"(knife: {knife_score:.3f}, fork: {fork_score:.3f})", debug_info)
        return current_label, 0.5

def _calculate_significant_defects(
    points: np.ndarray,
    hull_indices: np.ndarray,
    contour_area: float,
    debug_info: bool = False
) -> int:
    """
    Calculate the number of significant convexity defects in a contour.

    Parameters
    ----------
    points : np.ndarray
        Contour points in OpenCV format
    hull_indices : np.ndarray
        Indices of convex hull points
    contour_area : float
        Area of the contour for threshold calculation
    debug_info : bool
        Whether to print debug information

    Returns
    -------
    int
        Number of significant convexity defects
    """
    significant_defects = 0

    debug_print(f"Points shape: {points.shape}, Hull indices: {len(hull_indices)}", debug_info)

    if len(hull_indices) < 4:
        debug_print("⚠️ WARNING: Not enough hull points for defect analysis", debug_info)
        return significant_defects

    try:
        # Ensure points are integer type for convexityDefects
        int_points = points.astype(np.int32)
        defects = cv2.convexityDefects(int_points, hull_indices)

        debug_print(f"Convexity defects found: {defects.shape[0] if defects is not None else 0}", debug_info)

        if defects is None:
            return significant_defects

        # Count significant defects (potential tines)
        threshold = max(10.0, np.sqrt(contour_area) * 0.1)

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            # Convert depth to actual distance (d is in 1/256 units)
            distance = d / 256.0

            if distance > threshold:
                significant_defects += 1
                debug_print(f"Significant defect {significant_defects}: "
                          f"depth={distance:.2f} (threshold={threshold:.2f})", debug_info)

    except Exception as e:
        print(f"🚨 ERROR: calculating convexity defects: {str(e)}")

    debug_print(f"Total significant defects: {significant_defects}", debug_info)
    return significant_defects

def plot_hull_defects(mask: np.ndarray, title: str = "Contour Analysis", depth_thresh: float = 10.0) -> None:
    """
    Plot contour with hull and defects for debugging utensil classification.

    Parameters
    ----------
    mask : np.ndarray
        (N,2) array of pixel coordinates
    title : str, default="Contour Analysis"
        Title for the plot
    depth_thresh : float, default=10.0
        Minimum depth threshold for significant defects
    """
    if cv2 is None or plt is None:
        print("OpenCV or matplotlib not available, skipping plot")
        return

    cnt = mask.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(cnt, returnPoints=False)
    hull_pts = cv2.convexHull(cnt).squeeze()

    defects = cv2.convexityDefects(cnt, hull)
    hull_closed = np.vstack([hull_pts, hull_pts[0]])

    plt.figure(figsize=(5, 3))
    plt.plot(mask[:, 0], mask[:, 1], 'k-', lw=1, label='Contour')
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'r--', lw=2, label='Convex hull')

    # draw bridging edges & mark valleys
    if defects is not None:
        for s, e, f, d in defects[:, 0]:
            depth = d / 256.0
            if depth < depth_thresh:  # skip shallow noise
                continue
            P_s = cnt[s][0]
            P_e = cnt[e][0]
            P_f = cnt[f][0]

            # valley marker
            plt.scatter(P_f[0], P_f[1], c='magenta', s=50, marker='x', zorder=11)
            plt.text(P_f[0] + 2, P_f[1], f"{depth:.1f}", color='magenta', fontsize=8)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.legend()
    plt.show()

@dataclass
class FocalEstimate:
    fpx:   float     # focal length in *metric-consistent* pixels
    sigma: float     # 1-σ uncertainty (pixels)
    source: str       # 'exif', 'utensil', 'plate'
    independent: bool  # True if statistically independent of others

def combine_focal_estimates(estimates: List[FocalEstimate],
                   z_thresh: float = 3.0,
                   corr_inflate: float = 1.4) -> Tuple[float, float, str]:
    """
    Combine multiple focal length estimates into a single robust estimate.

    Uses precision-weighted averaging after outlier detection via Z-score testing.
    Correlated estimates have their uncertainties inflated before fusion.

    Parameters
    ----------
    estimates : List[FocalEstimate]
        List of focal length estimates to combine. Each estimate should have
        attributes: fpx (focal length in pixels), sigma (uncertainty),
        source (description), and independent (bool flag).
    z_thresh : float, default=3.0
        Z-score threshold for outlier detection. Estimates that differ by
        more than this many standard deviations are considered outliers.
    corr_inflate : float, default=1.4
        Factor to inflate uncertainty for correlated (non-independent) estimates
        to account for systematic errors.

    Returns
    -------
    focal_final : float
        Fused focal length in pixels from precision-weighted mean
    sigma_final : float
        1-σ uncertainty of the fused estimate
    fusion_note : str
        Description of how the fusion was resolved (e.g., which estimates used)

    Raises
    ------
    RuntimeError
        If no estimates are provided

    Examples
    --------
    >>> est1 = FocalEstimate(fpx=500.0, sigma=10.0, source="EXIF", independent=True)
    >>> est2 = FocalEstimate(fpx=505.0, sigma=15.0, source="homography", independent=False)
    >>> focal, uncertainty, note = combine_focals([est1, est2])
    """
    if not estimates:
        raise RuntimeError("No focal estimates given")

    # ── 0 Single Estimate Fast Path ───────────────────────────────────
    if len(estimates) == 1:
        single_estimate = estimates[0]
        return single_estimate.fpx, single_estimate.sigma, f"{single_estimate.source} only"

    # ── 1 Inflate Uncertainties for Correlated Estimates ──────────────────────
    inflated_estimates = []
    for estimate in estimates:
        inflated_sigma = estimate.sigma
        # any pair that is *not* independent is considered correlated
        if not estimate.independent:
            inflated_sigma *= corr_inflate
        inflated_estimates.append((estimate.fpx, inflated_sigma, estimate.source))

    # ── 2 Outlier Detection via Pairwise Z-Score Testing ──────────────────────
    keep_flags = [True] * len(inflated_estimates)
    for i in range(len(inflated_estimates)):
        for j in range(i + 1, len(inflated_estimates)):
            first_focal, first_sigma, _ = inflated_estimates[i]
            second_focal, second_sigma, _ = inflated_estimates[j]
            z_score = abs(first_focal - second_focal) / np.sqrt(first_sigma**2 + second_sigma**2)
            if z_score > z_thresh:
                # ⚠️ WARNING: mark the one with larger σ as outlier
                if first_sigma > second_sigma:
                    keep_flags[i] = False
                else:
                    keep_flags[j] = False

    kept_estimates = [estimate for estimate, keep_flag in zip(inflated_estimates, keep_flags) if keep_flag]
    if not kept_estimates:
        # ❗ IMPORTANT: all disagree wildly – fall back to smallest-σ estimate
        best_estimate = min(inflated_estimates, key=lambda estimate_tuple: estimate_tuple[1])
        return best_estimate[0], best_estimate[1], "fallback to lowest-σ (all conflicted)"

    # ── 3 Precision-Weighted Mean of Surviving Estimates ──────────────────────
    precision_weights = [1 / estimate_tuple[1]**2 for estimate_tuple in kept_estimates]
    focal_final = sum(weight * focal_value for weight, (focal_value, _, _) in zip(precision_weights, kept_estimates)) / sum(precision_weights)
    sigma_final = np.sqrt(1.0 / sum(precision_weights))
    source_labels = ",".join(estimate_tuple[2] for estimate_tuple in kept_estimates)

    return focal_final, sigma_final, f"weighted mean of {source_labels}"

@dataclass
class ScaleEstimate:
    scale:  float          # scale factor, e.g. metres / arbitrary-unit
    sigma:  float          # 1-σ uncertainty (same units as s)
    source: str            # 'plate' or 'utensil'
    corr_with_plate: bool  # True if this estimate re-uses the plate ellipse

def combine_scale_estimates(est1: Optional[ScaleEstimate],
                   est2: Optional[ScaleEstimate],
                   z_threshold: float = 3.0,
                   corr_inflate: float = 1.4) -> Tuple[float, float, str]:
    """
    Combine up to two scale estimates into a single robust estimate.

    Uses precision-weighted averaging when estimates agree, or selects the
    more precise estimate when they disagree beyond the Z-score threshold.
    Handles correlation between estimates by inflating uncertainties.

    Parameters
    ----------
    est1 : ScaleEstimate or None
        First scale estimate with attributes: scale (scale factor),
        sigma (uncertainty), source (description), and corr_with_plate
        (correlation flag).
    est2 : ScaleEstimate or None
        Second scale estimate with same attributes as est1.
    z_threshold : float, default=3.0
        Z-score threshold for consistency testing. Estimates differing
        by more than this many standard deviations are considered
        inconsistent and the more precise one is selected.
    corr_inflate : float, default=1.4
        Factor to inflate uncertainties when both estimates are
        correlated with plate solving (corr_with_plate=True).

    Returns
    -------
    final_scale : float
        Combined scale factor from precision-weighted mean or best estimate
    final_certainty : float
        1-σ uncertainty of the combined scale estimate
    source_note : str
        Description of combination method used (e.g., "weighted mean",
        "kept {source} (|Δ|=X.X σ)", "{source} only")

    Raises
    ------
    RuntimeError
        If both est1 and est2 are None (no estimates provided)

    Examples
    --------
    >>> est1 = ScaleEstimate(scale=1.5, sigma=0.1, source="homography", corr_with_plate=False)
    >>> est2 = ScaleEstimate(scale=1.52, sigma=0.15, source="keypoints", corr_with_plate=True)
    >>> scale, uncertainty, note = combine_scale_estimates(est1, est2)
    >>> # Returns weighted mean if estimates are consistent

    >>> # If estimates disagree significantly:
    >>> est_bad = ScaleEstimate(scale=2.0, sigma=0.2, source="bad_method", corr_with_plate=False)
    >>> scale, uncertainty, note = combine_scale_estimates(est1, est_bad)
    >>> # Returns the estimate with smaller uncertainty
    """
    # ── 0  handle missing values ───────────────────────────────────
    estimates = [e for e in (est1, est2) if e is not None]
    if not estimates:
        raise RuntimeError("No scale estimates supplied")

    # single estimate → nothing to combine
    if len(estimates) == 1:
        e = estimates[0]
        return e.scale, e.sigma, f"{e.source} only"

    # ── 1  inflate σ if the two are correlated ────────────────────
    estimate1, estimate2 = estimates      # exactly two now
    if estimate1.corr_with_plate and estimate2.corr_with_plate:
        estimate1_sigma = estimate1.sigma * corr_inflate
        estimate2_sigma = estimate2.sigma * corr_inflate
    else:
        estimate1_sigma = estimate1.sigma
        estimate2_sigma = estimate2.sigma

    # ── 2  consistency Z-score test ───────────────────────────────
    pooled_sigma = np.sqrt(estimate1_sigma**2 + estimate2_sigma**2)
    z = abs(estimate1.scale - estimate2.scale) / pooled_sigma

    if z > z_threshold:
        # choose the one with smaller variance
        chosen = estimate1 if estimate1_sigma < estimate2_sigma else estimate2
        return (chosen.scale,
                chosen.sigma,
                f"kept {chosen.source} (|Δ|={z:.1f} σ)")

    # ── 3  precision-weighted mean ────────────────────────────────
    w1 = 1.0 / (estimate1_sigma**2)
    w2 = 1.0 / (estimate2_sigma**2)
    scale_comb  = (estimate1.scale * w1 + estimate2.scale * w2) / (w1 + w2)
    sigma_comb = np.sqrt(1.0 / (w1 + w2))

    return scale_comb, sigma_comb, "weighted mean"

def focal_from_refs(masks: dict, depth: np.ndarray, debug_info: bool = False) -> float:
    """
    Estimate focal length in pixels using reference objects with known dimensions.

    Handles special cases like:
    - Partial visibility of utensils
    - Fork tines vs. complete fork
    - Objects at arbitrary orientations
    - Multiple reference objects with different reliability
    - Outlier points in masks

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        Dictionary mapping object names to their masks (N,2) arrays of pixel coordinates
    depth : np.ndarray
        Depth map in meters
    debug_info : bool, optional
        Whether to print debug information, by default False

    Returns
    -------
    float
        Estimated focal length in pixels

    Raises
    ------
    RuntimeError
        If no suitable reference objects are found
    """
    # Get image dimensions from depth map
    H, W = depth.shape
    focal_estimates = []
    ref_objects = []

    # Check for reference objects with known dimensions
    # Group by reference type for better handling of multiple instances
    ref_types_found = {}
    potentially_mislabeled = []

    for key in masks:
        for ref_key in _REF_DIMS:
            if ref_key in key.lower():
                if ref_key not in ref_types_found:
                    ref_types_found[ref_key] = []
                ref_types_found[ref_key].append(key)
                ref_objects.append((key, ref_key))

                # Validate utensil type for knives and forks
                if ref_key in ['knife', 'fork']:
                    validated_type, confidence = validate_utensil_type(masks[key], ref_key)
                    if validated_type != ref_key and confidence > 0.7:
                        print(f"⚠️ Warning: {key} appears to be a {validated_type} (confidence: {confidence:.2f}), not a {ref_key}")
                        potentially_mislabeled.append((key, ref_key, validated_type, confidence))
                break

    # Print diagnostic info about found reference objects
    for ref_type, keys in ref_types_found.items():
        if len(keys) > 1:
            print(f"Found {len(keys)} instances of {ref_type}: {', '.join(keys)}")
        elif len(keys) == 1:
            print(f"Found 1 instance of {ref_type}: {keys[0]}")

    # Handle mislabeled utensils
    wrong_labels = {}
    for key, current_type, validated_type, confidence in potentially_mislabeled:
        if confidence > 0.8:  # High confidence correction
            print(f"Auto-correcting {key} from '{current_type}' to '{validated_type}' (confidence: {confidence:.2f})")
            # Create corrected reference entry
            idx = ref_objects.index((key, current_type))
            ref_objects[idx] = (key, validated_type)
            wrong_labels[key] = current_type
        else:
            print(f"Consider checking {key} manually (possible {validated_type}, confidence: {confidence:.2f})")

    if not ref_objects:
        raise RuntimeError("No reference objects (knife, fork, creditcard) found in image")

    for obj_key, ref_type in ref_objects:
        # Get the mask and remove outlier points
        original_mask = masks[obj_key]
        if original_mask.shape[0] < 10:  # Skip if mask has too few points
            continue

        # Clean the mask to remove outlier points
        mask = _remove_mask_outliers(original_mask)
        if mask.shape[0] < 10:  # Skip if cleaned mask has too few points
            print(f"Skipping {obj_key}: too few points after outlier removal")
            continue

        # For utensils, we can use Principal Component Analysis to find the main orientation
        # This is more robust for elongated objects like knives and forks
        if PCA is None:
            print("PCA not available, using simple bounding box approach")
            # Fallback to simple bounding box
            width_px = np.max(mask[:, 0]) - np.min(mask[:, 0])
            height_px = np.max(mask[:, 1]) - np.min(mask[:, 1])
        else:

            # Perform PCA to find the principal axes
            pca = PCA(n_components=2)
            pca.fit(mask)

            # The first principal component gives the main axis
            main_axis = pca.components_[0]
            secondary_axis = pca.components_[1]

            # Project the points onto the principal axes
            projected = pca.transform(mask)

            # Calculate the min/max along each axis
            min_proj = np.min(projected, axis=0)
            max_proj = np.max(projected, axis=0)

            # Calculate width and height along the principal axes
            width_px = max_proj[0] - min_proj[0]  # Length along main axis
            height_px = max_proj[1] - min_proj[1]  # Width along secondary axis

        # Ensure width is the longer dimension for elongated objects
        if width_px < height_px:
            width_px, height_px = height_px, width_px

        # Get median depth of the object
        xs_i = np.clip(mask[:,0].astype(int), 0, depth.shape[1]-1)
        ys_i = np.clip(mask[:,1].astype(int), 0, depth.shape[0]-1)
        obj_depths = depth[ys_i, xs_i]
        median_depth = np.median(obj_depths)

        # Skip if we have unreliable depth
        if median_depth <= 0 or np.isnan(median_depth):
            continue

        # Initialize focal estimate
        f_estimate = None

        # Calculate focal length based on the reference object type
        if ref_type in ['knife', 'fork', 'creditcard']:
            # For these objects, we know both dimensions
            real_width, real_height = _REF_DIMS[ref_type]

            # By convention, for utensils the width is the long dimension
            # and height is the short dimension

            # Check if this object had its label corrected
            wrong_type = wrong_labels.get(obj_key, ref_type)
            if wrong_type != ref_type:
                # Use the corrected dimensions
                real_width, real_height = _REF_DIMS[ref_type]
                print(f"  Using corrected dimensions for {obj_key} as {ref_type}")
                #ref_type = actual_type  # Use corrected type for further processing

            # Handle fork specially since it might be partially visible or at unusual angle
            if ref_type == 'fork':
                # Calculate aspect ratio to determine if we're seeing the whole fork or just tines
                aspect_ratio = width_px / height_px if height_px > 0 else 1
                width_to_height_ratio = real_width / real_height

                if debug_info:
                  # Print detailed diagnostics for fork measurements
                  print(f"\n===== FORK MEASUREMENT DIAGNOSTICS =====")
                  print(f"Fork dimensions in pixels: {width_px:.1f} × {height_px:.1f} (aspect ratio: {aspect_ratio:.2f})")
                  print(f"Real-world dimensions: {real_width*100:.1f}cm × {real_height*100:.1f}cm (ratio: {width_to_height_ratio:.2f})")
                  print(f"Median depth from depth map: {median_depth:.3f}")
                  print(f"Raw depth range: {np.min(obj_depths):.3f} → {np.max(obj_depths):.3f}")

                # Typical fork has length:width ratio of ~7:1
                # Tines section has length:width ratio of ~2:3
                if aspect_ratio > 3.0:
                  # Likely seeing whole fork (elongated shape)
                  f_w = (width_px * median_depth) / real_width
                  f_h = (height_px * median_depth) / real_height
                  if debug_info:
                    print(f"Focal estimate from width: {f_w:.1f} px")
                    print(f"Focal estimate from height: {f_h:.1f} px")

                  # Sanity check on focal length estimates
                  if 0.3 * W < f_w < 3.0 * W:
                      f_estimate = f_w
                  elif 0.3 * W < f_h < 3.0 * W:
                      f_estimate = f_h
                  else:
                      # If both estimates are unreasonable, give average
                      f_estimate = (f_w + f_h) / 2
                      print(f"  Unusual fork measurements, focal estimate may be less reliable")

                  print(f"  Detected complete fork (aspect ratio: {aspect_ratio:.1f})")
                else:
                  # Likely seeing just the tines section
                  tine_width, tine_height = _REF_DIMS['fork_tines']

                  # Calculate focal estimates using tine dimensions
                  f_w = (width_px * median_depth) / tine_width
                  f_h = (height_px * median_depth) / tine_height
                  if debug_info:
                    print(f"Focal estimate from width: {f_w:.1f} px")
                    print(f"Focal estimate from height: {f_h:.1f} px")

                  # Sanity check on focal length estimates
                  if 0.3 * W < f_w < 3.0 * W and width_px > height_px:
                      f_estimate = f_w
                  elif 0.3 * W < f_h < 3.0 * W and height_px >= width_px:
                      f_estimate = f_h
                  else:
                      # Use the more reasonable estimate or average if both are reasonable
                      if 0.3 * W < f_w < 3.0 * W and 0.3 * W < f_h < 3.0 * W:
                          f_estimate = (f_w + f_h) / 2
                      elif 0.3 * W < f_w < 3.0 * W:
                          f_estimate = f_w
                      elif 0.3 * W < f_h < 3.0 * W:
                          f_estimate = f_h
                      else:
                          # Both are unreasonable, but we'll use an average
                          f_estimate = (f_w + f_h) / 2
                          print(f"  Unusual fork tine measurements, focal estimate may be less reliable")

                  print(f"  Detected fork tines portion (aspect ratio: {aspect_ratio:.1f})")
                print(f"Final fork focal estimate: {f_estimate:.1f} px")
                print(f"====================================\n")
            # Handle knife specially since it might be viewed from the narrow edge
            elif ref_type == 'knife':
                # Calculate aspect ratio to determine if we're seeing the wide or narrow side
                aspect_ratio = width_px / height_px if height_px > 0 else 1
                width_to_height_ratio = real_width / real_height  # Expected ~8:1 for a typical knife

                # Calculate focal estimates for both dimensions
                f_w = (width_px * median_depth) / real_width
                f_h = (height_px * median_depth) / real_height

                if debug_info:
                  # Print detailed diagnostics for knife measurements
                  print(f"\n===== KNIFE MEASUREMENT DIAGNOSTICS =====")
                  print(f"Knife dimensions in pixels: {width_px:.1f} × {height_px:.1f} (aspect ratio: {aspect_ratio:.2f})")
                  print(f"Real-world dimensions: {real_width*100:.1f}cm × {real_height*100:.1f}cm (ratio: {width_to_height_ratio:.2f})")
                  print(f"Median depth from depth map: {median_depth:.3f}")
                  print(f"Raw depth range: {np.min(obj_depths):.3f} → {np.max(obj_depths):.3f}")
                  print(f"Focal estimate from width: {f_w:.1f} px")
                  print(f"Focal estimate from height: {f_h:.1f} px")

                # If aspect ratio is much lower than expected, we might be seeing the narrow edge
                if aspect_ratio < 3.0:
                    print(f"  Detected knife with unusual aspect ratio: {aspect_ratio:.1f}")
                    print(f"  May be viewing narrow edge or partial view")
                    # Favor the dimension that gives a more reasonable focal length
                    if 0.3 * W < f_w < 3.0 * W:
                        f_estimate = f_w
                    elif 0.3 * W < f_h < 3.0 * W:
                        f_estimate = f_h
                    else:
                        # If both estimates are unreasonable, give lower weight to this knife
                        f_estimate = (f_w + f_h) / 2
                        print(f"  Unreliable knife measurement, focal estimate may be inaccurate")
                else:
                    # Normal case - use the length
                    f_estimate = (width_px * median_depth) / real_width

                print(f"Final knife focal estimate: {f_estimate:.1f} px")
                print(f"====================================\n")
            # For credit card, use appropriate dimensions based on orientation
            elif ref_type == 'creditcard':
                # Aspect ratio check to determine orientation
                card_aspect = width_px / height_px if height_px > 0 else 1
                real_aspect = real_width / real_height

                # Card is close to expected aspect ratio, use width
                if abs(card_aspect - real_aspect) < 0.3:
                    f_estimate = (width_px * median_depth) / real_width
                else:
                    # Card might be viewed from a different angle, use average of both dimensions
                    f_w = (width_px * median_depth) / real_width
                    f_h = (height_px * median_depth) / real_height
                    f_estimate = (f_w + f_h) / 2

        if f_estimate is not None:
            focal_estimates.append(f_estimate)
            print(f"Focal estimate from {ref_type}: {f_estimate:.1f} px (depth: {median_depth:.3f}m, dims: {width_px:.1f}×{height_px:.1f}px)")

    if not focal_estimates:
        raise RuntimeError("Could not estimate focal length from reference objects")

    # Group focal estimates by reference type for better analysis
    estimates_by_type = {}
    weights_by_type = {}

    for i, (key, ref_type) in enumerate(ref_objects):
        if i >= len(focal_estimates):
            continue

        # Initialize lists for this reference type if not already present
        if ref_type not in estimates_by_type:
            estimates_by_type[ref_type] = []
            weights_by_type[ref_type] = []

        # Assign confidence weights based on object type
        if ref_type == 'creditcard':
            # Credit cards have very precise dimensions
            weight = 1.0
        elif ref_type == 'knife':
            # Knives can vary in length but are usually straight
            weight = 0.8
        elif ref_type == 'fork':
            # Forks are tricky due to tines/handle distinction
            mask = masks[key]
            aspect_ratio = (np.max(mask[:,0]) - np.min(mask[:,0])) / (np.max(mask[:,1]) - np.min(mask[:,1]))
            # Higher confidence if we see the whole fork with clear aspect ratio
            weight = 0.6 if aspect_ratio > 3.0 else 0.4
        else:
            weight = 0.5

        estimates_by_type[ref_type].append(focal_estimates[i])
        weights_by_type[ref_type].append(weight)

    # Analyze estimates within each reference type
    weights = []
    weighted_estimates = []

    for ref_type, estimates in estimates_by_type.items():
        type_weights = weights_by_type[ref_type]

        if len(estimates) > 1:
            # For multiple instances of the same reference type
            # Check consistency between estimates (should be similar)
            min_est = min(estimates)
            max_est = max(estimates)
            mean_est = sum(estimates) / len(estimates)

            # If estimates are consistent (within 15% of mean), boost confidence
            consistency = 1.0 - min(1.0, (max_est - min_est) / mean_est)

            # Print diagnostic info about multiple instances
            print(f"Multiple {ref_type} estimates: {[f'{e:.1f}' for e in estimates]} px")
            print(f"  Consistency: {consistency:.2f} (higher is better)")

            # For consistent estimates, use their weighted average with boosted weight
            if consistency > 0.85:  # Very consistent
                boost = 1.2
                print(f"  High consistency detected, boosting confidence by {boost:.1f}x")
            elif consistency > 0.7:  # Moderately consistent
                boost = 1.1
                print(f"  Good consistency detected, boosting confidence by {boost:.1f}x")
            else:
                boost = 1.0

            # Compute weighted average for this reference type
            type_weighted_sum = sum(e * w for e, w in zip(estimates, type_weights))
            type_weight_sum = sum(type_weights) * boost

            # Add to overall estimates
            weighted_estimates.append(type_weighted_sum)
            weights.append(type_weight_sum)
        else:
            # For single instance, just add its estimate and weight
            weighted_estimates.append(estimates[0] * type_weights[0])
            weights.append(type_weights[0])

    # Use weighted average if we have weights, otherwise use median
    if sum(weights) > 0:
        final_estimate = sum(weighted_estimates) / sum(weights)
        print(f"Using weighted average of focal estimates (weights sum: {sum(weights):.1f})")
    else:
        final_estimate = float(np.median(focal_estimates))
        print("Using median of focal estimates (no weights available)")

# Print overall summary
    print(f"Final focal length estimate: {final_estimate:.1f} px")

    return float(final_estimate)

def _process_plates_for_focal_refinement(plate_keys: List[str],
                                       masks: Dict[str, np.ndarray],
                                       depth: np.ndarray,
                                       focal_to_refine: float,
                                       debug_info: bool = False) -> Tuple[Optional[float], float]:
    """
    Process multiple plates to get refined focal length estimates and return best estimate.

    Parameters
    ----------
    plate_keys : List[str]
        List of plate keys corresponding to entries in the masks dictionary.
    masks : Dict[str, np.ndarray]
        Dictionary mapping object keys to their corresponding mask arrays, where
        each mask is an (N, 2) array of pixel coordinates.
    depth : np.ndarray
        Depth map of the scene with depth values per pixel.
    focal_to_refine : float
        Initial focal length estimate in pixels to be refined using plate data.
    debug_info : bool = False
        Whether to print additional debug information during processing.

    Returns
    -------
    Tuple[Optional[float], float]
        A tuple containing:
        - best_estimate : float or None
            Best focal length estimate from plates in pixels, or None if no valid estimates.
        - best_confidence : float
            Confidence value (0.0-1.0) for the best estimate, 0.0 if no valid estimates.
    """
    if debug_info:
        print(f"\n🍽️ Found {len(plate_keys)} plate(s) for refinement: {', '.join(plate_keys)}")

    plate_raw_estimates = []
    plate_confidences = []

    for plate_key in plate_keys:
        if debug_info:
            print(f"DEBUG: Passing f_px={focal_to_refine:.1f} px to refine_f_with_plate for {plate_key}")

        # Get ellipse data for current plate
        current_plate_ellipse = _fit_ellipse_to_mask(masks[plate_key])

        # Refine focal length using plate dimensions
        focal_refined, focal_raw_plate, plate_confidence = refine_f_with_plate(
            focal_to_refine, masks[plate_key], depth, plate_key,
            debug_info=debug_info, ellipse_data=current_plate_ellipse
        )

        # Store raw estimates (for combining with utensil estimates)
        plate_raw_estimates.append(focal_raw_plate)
        plate_confidences.append(plate_confidence)

        # Print diagnostic info (warnings always print)
        threshold = 0.20 * focal_to_refine  # 20% threshold
        if abs(focal_refined - focal_to_refine) > threshold:
            print(f"⚠️ WARNING: {plate_key} refined focal {focal_refined:.1f} px differs significantly from original {focal_to_refine:.1f} px")
        elif debug_info:
            print(f"{plate_key} refined focal {focal_refined:.1f} px is close to original {focal_to_refine:.1f} px")

    # Select best plate estimate based on confidence
    if plate_raw_estimates:
        best_plate_idx = np.argmax(plate_confidences)
        best_estimate = plate_raw_estimates[best_plate_idx]
        best_confidence = plate_confidences[best_plate_idx]
        if debug_info:
            print(f"Using raw plate focal estimate: {best_estimate:.1f} px (confidence: {best_confidence:.2f})")
        return best_estimate, best_confidence

    return None, 0.0

def _combine_utensil_and_plate_estimates(focal_from_utensils: float,
                                        utensil_confidence: float,
                                        focal_from_plate: Optional[float], plate_confidence: float,
                                        depth_consistency: float, utensil_keys: List[str],
                                        debug_info: bool = False) -> float:
    """
    Combine focal length estimates from utensils and plates using weighted fusion.

    This function merges focal length estimates from different sources (utensils and plates)
    using confidence-weighted averaging. The weights are adjusted based on depth consistency
    and the reliability of each estimation method.

    Parameters
    ----------
    focal_from_utensils : float
        Focal length estimate derived from utensil measurements, in pixels.
    utensil_confidence : float
        Base confidence value (0.0-1.0) for the utensil-based estimate.
    focal_from_plate : float or None
        Focal length estimate derived from plate measurements, in pixels. None if no plate estimate is available.
    plate_confidence : float
        Base confidence value (0.0-1.0) for the plate-based estimate.
    depth_consistency : float
        Consistency score (0.0-1.0) indicating how well depth measurements
        agree between different objects in the scene. 1.0 = perfect consistency.
    utensil_keys : List[str]
        List of utensil object keys for determining adjustment factors.
    debug_info : bool, optional
        Whether to print debug information during processing. Default is False.

    Returns
    -------
    float
        Combined focal length estimate in pixels, computed as a weighted average
        of available estimates or falling back to utensil estimate if weighting fails.
    """
    if debug_info:
        print(f"\n--- FOCAL LENGTH ESTIMATION STRATEGY ---")

    available_estimates = []
    weights = []

    # Add plate estimate if available
    if focal_from_plate is not None:
        available_estimates.append(focal_from_plate)

        # Apply consistency boost to plate confidence if needed
        plate_weight = plate_confidence
        if depth_consistency < 0.7:
            consistency_boost_factor = 2.0 + 3.0 * (1.0 - depth_consistency)
            plate_weight = min(0.95, plate_confidence * consistency_boost_factor)
            if debug_info:
                print(f"Boosted plate confidence: {plate_confidence:.2f} → {plate_weight:.2f} (depth inconsistency: {1.0-depth_consistency:.2f})")
        else:
            if debug_info:
                print(f"Using plate confidence: {plate_weight:.2f}")

        weights.append(plate_weight)

    # Add utensil estimate with adjusted confidence
    adjusted_utensil_confidence = _calculate_adjusted_utensil_confidence(
        utensil_confidence, depth_consistency, utensil_keys, focal_from_plate
    )

    available_estimates.append(focal_from_utensils)
    weights.append(adjusted_utensil_confidence)
    if debug_info:
        print(f"Adjusted utensil confidence: {utensil_confidence:.2f} → {adjusted_utensil_confidence:.2f} (depth consistency: {depth_consistency:.2f})")

    # Compute weighted average
    total_weight = sum(weights)
    if total_weight > 0:
        combined_focal = sum(est * weight for est, weight in zip(available_estimates, weights)) / total_weight
        if debug_info:
            print(f"✓ Combined all estimates → f_px={combined_focal:.1f} px (unscaled)")

            # Show breakdown for transparency
            for i, (est, wt) in enumerate(zip(available_estimates, weights)):
                source = "plate" if i == 0 and focal_from_plate is not None else "utensil"
                print(f"  - {source}: {est:.1f} px (weight: {wt:.2f})")
    else:
        # Fallback to utensil estimate
        combined_focal = focal_from_utensils
        if debug_info:
            print(f"✓ Using utensil estimate → f_px={combined_focal:.1f} px (unscaled)")

    if debug_info:
        print(f"--- END FOCAL LENGTH ESTIMATION STRATEGY ---")
    return combined_focal

def _calculate_adjusted_utensil_confidence(utensil_confidence: float,
                                         depth_consistency: float,
                                         utensil_keys: List[str],
                                         plate_estimate: Optional[float]) -> float:
    """
    Calculate adjusted confidence for utensil estimates based on consistency and knife presence.

    This function applies penalties to utensil-based focal length estimates based on
    depth consistency and the presence of knives (which are less reliable for estimation).
    Lower depth consistency and knife presence both reduce the confidence in utensil estimates.

    Parameters
    ----------
    utensil_confidence : float
        Base confidence value (0.0-1.0) for utensil-based estimates.
    depth_consistency : float
        Consistency score (0.0-1.0) indicating agreement between depth measurements
        of different objects. 1.0 = perfect consistency.
    utensil_keys : List[str]
        List of utensil object keys used to detect knife presence.
    plate_estimate : float or None
        Plate-based focal length estimate in pixels. Used to determine if additional
        caps should be applied to utensil confidence.

    Returns
    -------
    float
        Adjusted confidence value (0.0-1.0) for the utensil estimate, with penalties
        applied based on depth inconsistency and knife presence.
    """
    is_knife_present = any("knife" in key.lower() for key in utensil_keys)

    # Apply penalties based on depth consistency
    if is_knife_present and depth_consistency < 0.7:
        # Cubic penalty for knives with inconsistent depths
        consistency_factor = depth_consistency ** 3
        knife_penalty = 0.3  # Only 30% of original confidence
        adjusted_confidence = utensil_confidence * consistency_factor * knife_penalty
    else:
        # Standard quadratic penalty for other utensils
        consistency_factor = depth_consistency ** 2
        adjusted_confidence = utensil_confidence * consistency_factor

    # Apply maximum cap when plate is available and inconsistency exists
    if plate_estimate is not None and depth_consistency < 0.7:
        max_allowed_confidence = 0.2  # Maximum 20% weight
        adjusted_confidence = min(adjusted_confidence, max_allowed_confidence)

    return adjusted_confidence

def find_closest_plate_size(depth: np.ndarray,
                            plate_mask: np.ndarray,
                            plate_ellipse: Optional[Tuple] = None,
                            f_px: Optional[float] = None,
                            debug_info: bool = False) -> Optional[Tuple[str, float, float, float, float]]:
    """
    Find the closest standard plate size from a plate mask or ellipse data.

    Parameters
    ----------
    depth : np.ndarray
        Depth map (H, W)
    plate_mask : np.ndarray
        (N, 2) array of plate mask pixel coordinates
    plate_ellipse : tuple, optional
        Ellipse data from fit_ellipse_to_mask:
        ((center_x, center_y), (major_radius, minor_radius), angle).
        If None, will compute from plate_mask.
    f_px : float, optional
        Focal length in pixels (needed for depth calculations). Required.
    debug_info : bool, optional
        Whether to print debug information

    Returns
    -------
    tuple[str, float, float, float, float]
        Tuple containing:
        - plate_type : str (e.g., 'plate_dinner')
        - plate_diameter : float (in meters)
        - min_depth : float (in meters)
        - max_depth : float (in meters)
        - median_depth : float (in meters)
        Returns ("plate", 0.254, 0, 0, 0) as default if estimation fails
    """
    # ─── Input Validation and Early Returns ─────────────────────
    if plate_mask is None or plate_mask.shape[0] < 10:
        print("WARNING: Cannot proceed - no plate mask data available or too few points")
        return "plate", _REF_DIMS['plate'], 0.0, 0.0, 0.0

    if f_px is None:
        print("WARNING: No focal length provided for plate size estimation")
        return "plate", _REF_DIMS['plate'], 0.0, 0.0, 0.0

    # ─── Get Depth Data from Plate Mask ─────────────────────
    # Sample depths at plate mask points
    h, w = depth.shape
    xs_i = np.clip(plate_mask[:,0].astype(int), 0, w-1)
    ys_i = np.clip(plate_mask[:,1].astype(int), 0, h-1)
    plate_depths = depth[ys_i, xs_i]

    min_depth = np.min(plate_depths)
    max_depth = np.max(plate_depths)
    depth_range = max_depth - min_depth
    median_depth = np.median(plate_depths)

    # ─── Get or Generate Ellipse Data ─────────────────────
    if plate_ellipse is None:
        plate_ellipse = _fit_ellipse_to_mask(plate_mask)
        if plate_ellipse is None:
            print("WARNING: Could not fit ellipse to plate mask")
            return "plate", _REF_DIMS['plate'], min_depth, max_depth, median_depth

    # Extract ellipse parameters
    (_, _), (major_radius, minor_radius), _ = plate_ellipse
    #circularity = 4 * np.pi * major_radius * minor_radius / (major_radius ** 2 + minor_radius ** 2)
    circularity = minor_radius * 2 / major_radius * 2

    # ─── Determine Plate Type and Size ─────────────────────
    estimated_diameter = (major_radius * 2 * median_depth) / f_px
    plate_entries = {k: v for k, v in _REF_DIMS.items() if k.startswith('plate_')}
    plate_type, plate_diameter = min(plate_entries.items(),
                                key=lambda x: abs(x[1] - estimated_diameter))

    if debug_info:
        print(f"\n----- 🍽️ PLATE SIZE DETECTION DIAGNOSTICS -----")
        print(f"Raw estimated diameter: {estimated_diameter*100:.1f}cm")
        print(f"Depth statistics:")
        print(f"  Min depth: {min_depth:.3f}, Max depth: {max_depth:.3f}")
        print(f"  Median depth: {median_depth:.3f}, Depth range: {max_depth - min_depth:.3f}")
        print(f"Circularity: {circularity:.3f}")
        print(f"Standard plate sizes:")
        for p_type, p_diam in plate_entries.items():
            print(f"  {p_type}: {p_diam*100:.1f}cm (difference: {abs(p_diam - estimated_diameter)*100:.1f}cm)")
        print(f"  Identified plate type: {plate_type} (diameter: {plate_diameter*100:.1f}cm)")
        print(f"------------------------------------")

    # Return the closest plate size and depth statistics
    return plate_type, plate_diameter, min_depth, max_depth, median_depth

def refine_f_with_plate(f_px: float, plate_mask: np.ndarray,
                        depth: np.ndarray,
                        plate_key: str = "plate",
                        debug_info: bool = False,
                        ellipse_data: Optional[tuple] = None) -> tuple[float, float, float]:
    """
    Refine focal length using the plate's known circular shape.
    Identifies the most likely plate size from standard options.

    Parameters
    ----------
    f_px : float
        Initial focal length in pixels
    plate_mask : np.ndarray
        (N,2) array of pixel coordinates for the plate mask
    depth : np.ndarray
        Depth map in meters
    plate_key : str
        The key of the plate in the masks dictionary
    debug_info : bool
        Whether to print debug information
    ellipse_data : tuple, optional
        Pre-computed ellipse data to avoid redundant fitting
        ((center_x, center_y), (major_radius, minor_radius), angle)

    Returns
    -------
    tuple
        - Refined focal length in pixels (input focal length combined with plate-based estimate)
        - Raw plate-based focal estimate in pixels
        - Confidence weight (0-1) for the plate estimate
    """
    # Get image dimensions from depth map
    H, W = depth.shape
    if plate_mask.shape[0] < 50:  # Need enough points for reliable estimation
        print("Too few points in plate mask for refinement")
        return f_px, f_px, 0.0

    # Clean the plate mask to remove outlier points
    original_count = plate_mask.shape[0]
    plate_mask = _remove_mask_outliers(plate_mask, distance_threshold=15)
    if plate_mask.shape[0] < 50:
        print(f"Too few points in plate mask after outlier removal ({plate_mask.shape[0]} points)")
        return f_px, f_px, 0.0

    print(f"Cleaned plate mask: {original_count} → {plate_mask.shape[0]} points")

    # Use pre-computed ellipse data if provided, otherwise compute it
    if ellipse_data is None:
        ellipse_data = _fit_ellipse_to_mask(plate_mask)

    if ellipse_data is None:
        print("Failed to fit ellipse to plate mask")
        return f_px, f_px, 0.0

    (center_x, center_y), (major_radius, minor_radius), angle = ellipse_data
    major_axis = major_radius * 2
    minor_axis = minor_radius * 2

    # Use find_closest_plate_size to get plate type, diameter, and depth statistics
    result = find_closest_plate_size(
        depth=depth,
        plate_mask=plate_mask,
        plate_ellipse=ellipse_data,
        f_px=f_px,
        debug_info=debug_info
    )

    if result is None:
        print("Could not determine plate size")
        return f_px, f_px, 0.0

    plate_type, plate_diameter, min_depth, max_depth, median_depth = result

    if median_depth <= 0 or np.isnan(median_depth):
        print("Unreliable depth for plate")
        return f_px, f_px, 0.0

    # Calculate estimated diameter for validation
    estimated_diameter = (major_radius * 2 * median_depth) / f_px

    # Check if there's a large discrepancy between estimated and standard size
    size_diff = abs(plate_diameter - estimated_diameter) * 100  # Convert to cm
    if size_diff > 10:
        print(f"⚠️ WARNING: Estimated plate size ({estimated_diameter*100:.1f}cm) differs significantly "
              f"from closest standard size {plate_type} ({plate_diameter*100:.1f}cm) by {size_diff:.1f}cm")
        print(f"   This may indicate an incorrect focal length or depth scale.")

    # Estimate focal length from the major axis (represents true diameter when viewed at angle)
    f_estimate = (major_radius * 2 * median_depth) / plate_diameter
    if debug_info:
      # Print detailed diagnostics for plate measurements
      print(f"\n===== PLATE MEASUREMENT DIAGNOSTICS =====")
      print(f"Plate axes in pixels: major={major_axis:.1f}, minor={minor_axis:.1f}")
      print(f"Plate radii in pixels: major={major_radius:.1f}, minor={minor_radius:.1f}")
      print(f"Detected plate size: {plate_type} ({plate_diameter*100:.1f}cm)")
      print(f"Estimated actual diameter: {estimated_diameter*100:.1f}cm")
      print(f"Circularity: {minor_radius/major_radius if major_radius > 0 else 0:.3f}")
      print(f"Median depth from depth map: {median_depth:.3f}")
      print(f"Raw depth range: {min_depth:.3f} → {max_depth:.3f}")
      print(f"Raw plate focal estimate: {f_estimate:.1f} px")
      print(f"====================================\n")

    # Calculate how circular the plate appears (1.0 = perfect circle)
    circularity = minor_radius / major_radius if major_radius > 0 else 0

    # Calculate confidence in the plate-based estimate
    # More circular (viewed head-on) = more reliable
    plate_confidence = circularity ** 2  # Square to penalize non-circular views more

    # If plate is very elliptical (highly angled view), reduce confidence further
    if circularity < 0.5:
        plate_confidence *= 0.5

    # Base confidence for plate-based estimates
    base_confidence = 0.7  # Higher base confidence for plates

    plate_confidence = base_confidence + (1 - base_confidence) * plate_confidence  # Minimum confidence = base_confidence

    # Weighted average between original and plate-based estimate
    # Bias toward plate estimate as it's generally more reliable
    refined_f = plate_confidence * f_estimate + (1 - plate_confidence) * f_px

    print(f"Plate-based focal estimate: {f_estimate:.1f} px (using {plate_type}, circularity: {circularity:.2f}, axes: {major_axis:.1f}×{minor_axis:.1f}px)")
    return float(refined_f), float(f_estimate), float(plate_confidence)


def find_best_plate_scale(depth_map: np.ndarray,
                          plate_mask: np.ndarray,
                          f_px_exif: float,
                          standard_plate_sizes: dict,
                          plate_prior_probabilities: dict,
                          ellipse_data: Optional[tuple] = None) -> tuple[float, float, float, Optional[tuple]]:
    """
    Find the scale factor that makes depth geometrically consistent with plate circularity.

    Uses the constraints that a plate is a perfect circle in 3D space, and its appearance
    as an ellipse in 2D is purely due to perspective, combined with depth variations.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map of the scene with depth values per pixel, in arbitrary units
    plate_mask : np.ndarray
        (N,2) array of pixel coordinates for the plate mask
    f_px_exif : float
        Focal length from EXIF data, in pixels
    standard_plate_sizes : dict
        Dictionary mapping plate types to their diameters in meters
    plate_prior_probabilities : dict
        Dictionary mapping plate types to their prior probabilities
    ellipse_data : tuple, optional
        Pre-computed ellipse data to avoid redundant fitting
        ((center_x, center_y), (major_radius, minor_radius), angle)

    Returns
    -------
    best_scale : float
        Best scale factor to convert depth values to meters
    best_size : float
        Estimated plate diameter in meters
    best_error : float
        Error metric for the best match (lower is better)
    ellipse_data : tuple, optional
        Additional data about the ellipse fit: center, axes, angle, circularity
    """

    # Calculate image dimensions
    h, w = depth_map.shape

    # We'll use the actual plate size in pixels to calculate distance
    # Using the formula: distance = (real_world_size * focal_length) / pixel_size

    # Typical distance ranges for food photography (in meters)
    MIN_REASONABLE_DISTANCE = 0.15  # 15cm minimum
    MAX_REASONABLE_DISTANCE = 1.50  # 150cm maximum

    # Analyze depth distribution across the entire image
    # This helps determine if the photo was taken close-up or from far away
    valid_depths = depth_map[depth_map > 0]  # Exclude zero/invalid depths
    if len(valid_depths) > 0:
        depth_mean = np.mean(valid_depths)
        depth_std = np.std(valid_depths)
        depth_relative_std = depth_std / depth_mean if depth_mean > 0 else 0
        # Calculate percentiles for more robust analysis
        depth_5th = np.percentile(valid_depths, 5)
        depth_95th = np.percentile(valid_depths, 95)
        depth_range_ratio = (depth_95th - depth_5th) / depth_mean if depth_mean > 0 else 0

        print(f"Depth distribution analysis:")
        print(f"  Mean depth: {depth_mean:.3f} (arbitrary units)")
        print(f"  Depth std deviation: {depth_std:.3f} (relative: {depth_relative_std:.3f})")
        print(f"  5-95% depth range ratio: {depth_range_ratio:.3f}")
        print(f"  Interpretation: {'Close-up shot (less than 15cm)' if depth_range_ratio < 1.0 else 'Medium distance (15cm to 30cm)' if depth_range_ratio < 1.5 else 'Far shot (more than 30cm)'}")
    else:
        depth_relative_std = 0
        depth_range_ratio = 0

    # Use pre-computed ellipse data if provided, otherwise compute it
    if ellipse_data is None:
        ellipse_data = _fit_ellipse_to_mask(plate_mask)

    if ellipse_data is None:
        print(f"Error in plate scale analysis, could not fit ellipse to the mask.")
        return 1.0, 0.0, float('inf'), None

    (center, (major_radius, minor_radius), angle) = ellipse_data
    major_axis = major_radius * 2
    minor_axis = minor_radius * 2

    # Calculate circularity - ratio of minor to major axis
    circularity = minor_axis / major_axis if major_axis > 0 else 1.0

    # Calculate the angle of the plate in 3D space
    # circularity = cos(tilt_angle) in a perfect scenario
    tilt_angle = np.arccos(min(circularity, 0.99))  # Clamp to avoid numerical issues

    # Sample depth values across the plate
    h, w = depth_map.shape
    xs = np.clip(plate_mask[:,0].astype(int), 0, w-1)
    ys = np.clip(plate_mask[:,1].astype(int), 0, h-1)
    plate_depths = depth_map[ys, xs]

    # Use percentiles to reduce impact of outliers
    min_depth = np.percentile(plate_depths, 5)
    max_depth = np.percentile(plate_depths, 95)
    median_depth = np.median(plate_depths)
    depth_range = max_depth - min_depth

    print(f"\n--- PLATE GEOMETRY ANALYSIS ---")
    print(f"Plate circularity: {circularity:.3f}")
    print(f"Estimated tilt angle: {np.degrees(tilt_angle):.1f}°")
    print(f"Depth range: {depth_range:.6f} (arbitrary units)")
    print(f"Major axis in pixels: {major_axis:.1f}, Minor axis: {minor_axis:.1f}")
    print(f"Image dimensions: {w}x{h} pixels")

    # APPROACH: Work backward from the geometry and focal length
    # 1. Start with the relationship: real_size = (pixel_size * depth) / focal_length
    # 2. Knowing the camera's focal length (from EXIF) and the pixel size (from the image),
    #    we can calculate what the real-world size would be at different depths

    # Use the actual plate size in pixels (major axis) to calculate the implied distance
    # for each standard plate size using the formula:
    # distance = (real_world_size * focal_length) / pixel_size

    # Calculate reasonable distance bounds based on observed pixel size
    for plate_type, plate_diameter in standard_plate_sizes.items():
        # What would the distance be if this were a plate of this standard size?
        implied_distance = (plate_diameter * f_px_exif) / major_axis

    # Plate proportional to frame size - calculate what percent of the frame width the plate takes up
    plate_to_frame_ratio = major_axis / w
    print(f"Plate takes up {plate_to_frame_ratio*100:.1f}% of the frame width")

    # Find which standard plate size is closest to our estimated size
    # after applying appropriate scaling
    best_error = float('inf')
    best_scale = 1.0
    best_size = _REF_DIMS['plate_dinner'] # default to dinner plate size

    results = []
    for plate_type, plate_diameter in standard_plate_sizes.items():
        # Calculate the implied distance for this plate size
        implied_distance_m = (plate_diameter * f_px_exif) / major_axis

        # Calculate scale factor based on physical constraints
        # For a plate of size plate_diameter, what scale factor would we need to apply to convert
        # the arbitrary depth units to meters?
        # Scale factor = actual_distance / arbitrary_depth_value
        scale_factor = implied_distance_m / median_depth

        # If we apply this scale factor, how well does our depth variation match expectations?
        expected_depth_variation = plate_diameter * np.sin(tilt_angle)
        actual_depth_variation = depth_range * scale_factor

        # Calculate depth error metric
        depth_error = abs(expected_depth_variation - actual_depth_variation) / expected_depth_variation if expected_depth_variation > 0 else float('inf')

        # Calculate a distance likelihood score (0-1) - how likely is this distance?
        # Give higher scores to distances in the expected range for food photography
        if implied_distance_m < MIN_REASONABLE_DISTANCE:
            distance_likelihood = implied_distance_m / MIN_REASONABLE_DISTANCE  # Penalize if too close
        elif implied_distance_m > MAX_REASONABLE_DISTANCE:
            distance_likelihood = MAX_REASONABLE_DISTANCE / implied_distance_m  # Penalize if too far
        else:
            distance_likelihood = 1.0  # Perfect score for distances in the expected range

        # Calculate how much of the frame we'd expect this plate to take up at this distance
        # Using the same formula: pixels = (real_size * focal_length) / distance
        expected_plate_pixel_width = (plate_diameter * f_px_exif) / implied_distance_m
        expected_plate_to_frame_ratio = expected_plate_pixel_width / w

        # Frame composition likelihood - how likely is a photographer to frame the shot this way?
        # Ideal framing typically has plate taking up 30-70% of the frame
        if plate_to_frame_ratio < 0.2:
            framing_likelihood = plate_to_frame_ratio / 0.2  # Penalize if plate is too small in frame
        elif plate_to_frame_ratio > 0.8:
            framing_likelihood = 0.8 / plate_to_frame_ratio  # Penalize if plate is too large in frame
        else:
            framing_likelihood = 1.0  # Perfect score for good framing

        # Look up prior probability for this plate type
        prior = plate_prior_probabilities.get(plate_type, 0.25)  # Default 0.25 if not found

        # Combined error metric (lower is better) - include multiple factors:
        # 1. Depth variation error
        # 2. Distance likelihood (favors plates at typical food photography distances)
        # 3. Framing likelihood (favors plates that would be framed well at this distance)
        # 4. Depth distribution likelihood (incorporating distance and camera angle)
        # 5. Prior probability (favors more common plate sizes)

        # Calculate depth distribution likelihood for this distance
        # Close-up shots tend to have less depth variation (smaller relative std)
        # Far shots tend to have more depth variation (larger relative std)

        # Improved model that accounts for both distance and viewing angle:
        # 1. Base variation (even when looking directly down): 0.5
        # 2. Distance factor: increases linearly with distance
        # 3. Angle factor: derived from plate circularity (1.0 = perfect circle, lower = more angled view)

        # Calculate angle factor from circularity
        # circularity of 1.0 = view from directly above (no angle)
        # circularity of 0.0 = view from side (90° angle)
        angle_factor = (1.0 - circularity) * 1.5  # More angled views have higher depth variation

        # Calculate distance factor (increases with distance)
        distance_factor = implied_distance_m * 2.0

        # Combined model (base + distance effect + angle effect)
        expected_depth_variation_ratio = 0.8 + distance_factor + angle_factor

        if 'depth_range_ratio' in locals() and depth_range_ratio > 0:
            # Use a more forgiving error calculation - we want to reward when the trend is correct
            # (i.e., farther distances have higher variation, closer have less)
            depth_dist_error = abs(expected_depth_variation_ratio - depth_range_ratio) / max(expected_depth_variation_ratio, depth_range_ratio)
            # Apply a non-linear transformation to be more forgiving of small errors
            depth_dist_likelihood = max(0, 1 - (depth_dist_error ** 0.7))

            # Print diagnostic info for each plate size
            print(f"{plate_type} Expected depth ratio: {expected_depth_variation_ratio:.3f} vs actual: {depth_range_ratio:.3f} (error: {depth_dist_error:.3f})")
            print(f"      Components: base=0.8, distance={distance_factor:.2f}, angle={angle_factor:.2f} (circularity={circularity:.2f})")
        else:
            print(f"❌ DEBUG: Skipping diagnostic for {plate_type} - condition failed")
            depth_dist_likelihood = 0.5  # Neutral if we couldn't calculate

        # Convert likelihoods to penalties (lower is better)
        distance_penalty = 1.0 - distance_likelihood
        framing_penalty = 1.0 - framing_likelihood
        depth_dist_penalty = 1.0 - depth_dist_likelihood
        prior_penalty = 1.0 - prior

        # Weighted combination of penalties
        combined_error = (
            0.1 * depth_error +              # Weight: 10% - Depth variation consistency
            0.2 * distance_penalty +         # Weight: 20% - Favor expected distances
            0.2 * framing_penalty +          # Weight: 20% - Favor expected framing
            0.3 * depth_dist_penalty +       # Weight: 30% - Depth distribution consistency (increased weight)
            0.2 * prior_penalty              # Weight: 20% - Favor common plate sizes
        )

        results.append({
            'plate_type': plate_type,
            'diameter': plate_diameter,
            'scale_factor': scale_factor,
            'expected_depth_variation': expected_depth_variation * 100,  # convert to cm for display
            'actual_depth_variation': actual_depth_variation * 100,  # convert to cm for display
            'implied_distance': implied_distance_m * 100,  # convert to cm for display
            'depth_error': depth_error,
            'distance_likelihood': distance_likelihood,
            'framing_likelihood': framing_likelihood,
            'depth_dist_likelihood': depth_dist_likelihood,
            'prior': prior,
            'combined_error': combined_error
        })

        if combined_error < best_error:
            best_error = combined_error
            best_scale = scale_factor
            best_size = plate_diameter

    # Print results for all plate sizes
    print(f"\nGeometric analysis for standard plate sizes:")
    for result in sorted(results, key=lambda x: x['combined_error']):
        print(f"  {result['plate_type']} ({result['diameter']*100:.1f}cm):")
        print(f"    Scale factor: {result['scale_factor']:.6f}")
        print(f"    Implied distance: {result['implied_distance']:.1f}cm")
        print(f"    Depth variation: {result['expected_depth_variation']:.1f}cm vs {result['actual_depth_variation']:.1f}cm")
        print(f"    Errors - Depth: {result['depth_error']:.3f}")
        print(f"    Distance likelihood: {result['distance_likelihood']:.3f}, Framing likelihood: {result['framing_likelihood']:.3f}")
        print(f"    Depth distribution likelihood: {result['depth_dist_likelihood']:.3f}")
        print(f"    Prior probability: {result['prior']:.2f}")
        print(f"    Combined error: {result['combined_error']:.3f}")

    print(f"\nBest match: {best_size*100:.1f}cm plate (error: {best_error:.3f})")
    print(f"Recommended scale factor: {best_scale:.6f}")
    print(f"--- END PLATE GEOMETRY ANALYSIS")

    # Return best scale, best size, error, and the ellipse data
    ellipse_data_circularity = (center, (minor_axis, major_axis), angle, circularity)
    return best_scale, best_size, best_error, ellipse_data_circularity

def calculate_scale_factor_from_refs(depth: np.ndarray,
                                    masks: Dict[str, np.ndarray],
                                    plate_key: Optional[str] = None,
                                    plate_ellipse_data: Optional[Tuple] = None,
                                    focal_length_px: Optional[float] = None) -> Tuple[float, float]:
    """
    Uses a plate with fitted ellipse data to determine the scale factor
    to go from arbitrary units to metric units in a depth map.

    Parameters
    ----------
    depth : np.ndarray
        Original depth map in arbitrary units
    masks : Dict[str, np.ndarray]
        Dictionary of object masks
    plate_key : Optional[str]
        Key for the plate mask in the masks dictionary
    plate_ellipse_data : Optional[Tuple]
        Ellipse data from _fit_ellipse_to_mask:
        ((center_x, center_y), (major_radius, minor_radius), angle)
    focal_length_px : Optional[float]
        Focal length in pixels (needed for depth calculation)

    Returns
    -------
    Tuple[float, float]
        - Scale factor for the depth map
        - Adjusted focal length to maintain correct XY scaling
    """
    # ─── Input Validation and Early Returns ─────────────────────
    if plate_key is None or plate_key not in masks:
        print("WARNING: Cannot rescale depth to metric units - no plate data available")
        return 1.0, focal_length_px or 1000.0

    if focal_length_px is None:
        print("WARNING: No focal length provided for depth rescaling")
        return 1.0, 1000.0

    # ─── Get Plate Mask and Depth Data ─────────────────────
    plate_mask = masks[plate_key]
    if plate_mask.shape[0] < 10:
        print("WARNING: Plate mask has too few points for depth rescaling")
        return 1.0, focal_length_px

    # ─── Determine Plate Type and Size Using New Function ─────────────────────
    result = find_closest_plate_size(
        depth=depth,
        plate_mask=plate_mask,
        plate_ellipse=plate_ellipse_data,
        f_px=focal_length_px,
        debug_info=True
    )

    if result is None:
        print("WARNING: Could not determine plate size for depth rescaling")
        return 1.0, focal_length_px

    plate_type, plate_diameter, min_depth, max_depth, median_depth = result

    # ─── Get or Generate Ellipse Data for Tilt Calculation ─────────────────────
    if plate_ellipse_data is None:
        plate_ellipse_data = _fit_ellipse_to_mask(plate_mask)
        if plate_ellipse_data is None:
            print("WARNING: Could not fit ellipse to plate mask")
            return 1.0, focal_length_px

    # Extract ellipse parameters for tilt calculation
    (_, _), (major_radius, minor_radius), _ = plate_ellipse_data
    major_axis = major_radius * 2
    minor_axis = minor_radius * 2
    circularity = minor_axis / major_axis if major_axis > 0 else 1.0

    print(f"  Major axis: {major_axis:.1f}px, Minor axis: {minor_axis:.1f}px")
    print(f"  Plate median depth: {median_depth:.3f}")
    print(f"  Current focal length: {focal_length_px:.1f} px")

    print(f"\n--- DEPTH RESCALING ---")
    print(f"Rescaling depth using plate '{plate_key}' of type '{plate_type}' with diameter: {plate_diameter*100:.1f}cm")

    # ─── Calculate Scale Factor Using Geometric Constraints ─────────────────────
    tilt_angle = np.arccos(min(circularity, 0.99))

    # Expected depth variation for this plate at this angle
    # For a tilted circular plate: depth_variation = diameter * sin(tilt_angle)
    expected_depth_variation = plate_diameter * np.sin(tilt_angle)

    # Calculate scale factor to match expected depth variation
    depth_range = max_depth - min_depth
    scale_factor = expected_depth_variation / depth_range if depth_range > 0 else 1.0

    # Adjust focal length to maintain correct XY scaling
    adjusted_f_px = focal_length_px * scale_factor

    print(f"  Plate circularity: {circularity:.3f} (tilt angle: {np.degrees(tilt_angle):.1f}°)")
    print(f"  Observed depth range: {depth_range:.3f} (arbitrary units)")
    print(f"  Expected depth variation: {expected_depth_variation:.3f}m")
    print(f"  Scale factor: {scale_factor:.6f}")
    print(f"  Adjusted focal length: {adjusted_f_px:.1f} px")
    print(f"  Scaled depth range: {depth_range*scale_factor:.3f}m")
    print(f"--- END DEPTH RESCALING ---")

    return scale_factor, adjusted_f_px





def calc_final_depth_focal_scale(
    depth: np.ndarray,
    masks: Dict[str, np.ndarray],
    focal_length_mm: Optional[float] = None,
    sensor_width_mm: Optional[float] = None,
    image_file: Optional[Union[str, Image.Image]] = None,
    device: str = 'cuda',
    debug_info: bool = False
) -> Tuple[float, float]:
    """
    Calculate final focal length and depth scale factor using multi-modal estimation.

    This function performs a comprehensive focal length and depth scale estimation pipeline by:
    1) Extracting camera specifications from user input or EXIF data to get initial focal length estimate
    2) Using reference objects (utensils, cards) for focal length estimation with known dimensions
    3) Applying geometric constraints from plates to determine depth map scaling
    4) Fusing multiple focal length estimates with uncertainty weighting

    Parameters
    ----------
    depth : np.ndarray
        Depth map of the scene with depth values per pixel in arbitrary units.
    masks : Dict[str, np.ndarray]
        Dictionary mapping object names to their mask coordinates as (N, 2) arrays.
    focal_length_mm : Optional[float], optional
        User-provided focal length in millimeters. If None, uses default iPhone specs.
    sensor_width_mm : Optional[float], optional
        Camera sensor width in millimeters. If None, uses default iPhone sensor width.
    image_file : Optional[Union[str, Image.Image]], optional
        Image file path or PIL Image object for EXIF data extraction.
    device : str, optional
        Device to use for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
    debug_info : bool, optional
        Whether to print detailed debug information during processing. Defaults to False.

    Returns
    -------
    Tuple[float, float]
        - depth_scale_factor : float
            Scale factor to convert depth values to meters.
        - f_final_px : float
            Final focal length used for reconstruction, in pixels.

    """
    # ─── Helper functions ─────────────────────

    if torch is not None and not torch.cuda.is_available():
        device = torch.device("cpu") if torch is not None else "cpu"
    else:
        device = "cuda" if torch is not None else "cpu"

    if np is None:
        raise ImportError("NumPy is required for this function")

    H, W = depth.shape
    cx, cy = W / 2, H / 2

    # ─── 1️⃣  Gather all possible focal estimates with confidence values ─────────
    # Initialize primary focal length in pixels from camera specs
    focal_length_px_from_specs = None

    # Camera specifications as ground truth
    # If the user-provided these they're most likely accurate!
    if focal_length_mm is not None and sensor_width_mm is not None:
        # Higher confidence for user-provided specs than defaults
        specs_confidence = 0.8
        if debug_info:
            print(f"\n📏 Using provided focal_length={focal_length_mm} mm (confidence: {specs_confidence:.2f})")
    else:
        # Default iPhone specifications when user doesn't provide camera parameters
        focal_length_mm = _IPHONE_FOCAL_MM
        sensor_width_mm = _IPHONE_SENSOR_WIDTH_MM
        specs_confidence = 0.2  # Very low confidence for default specs
        if debug_info:
            print(f"\n📱 Using default iPhone focal_length={focal_length_mm} mm (confidence: {specs_confidence:.2f})")

    # Convert focal length from mm to pixels using the formula: f_px = (f_mm * image_width) / sensor_width_mm
    focal_length_px_from_specs = (focal_length_mm * W) / sensor_width_mm
    if debug_info:
        print(f"Converted to f_px={focal_length_px_from_specs:.1f} px")
    current_estimation_method = "camera specs (user)"

    # ─── Extract focal length from EXIF metadata if available ─────────────────
    # EXIF data as another source of focal length information
    focal_length_px_from_exif = None
    exif_confidence = 0.0  # Initialize confidence value

    if image_file is not None:
        exif_data = extract_exif_focal_length(image_file)
        if exif_data is not None:
            focal_mm, focal_35mm = exif_data
            if focal_35mm is not None:
                # 35mm equivalent focal length is often more reliable for digital cameras
                # as it normalizes for different sensor sizes
                focal_length_px_from_exif = convert_35mm_to_pixels(focal_35mm, W)
                exif_confidence = 0.5  # Moderate confidence for 35mm equivalent
                if debug_info:
                    print(f"\n📷 Found 35mm equivalent focal length in EXIF: {focal_35mm}mm → {focal_length_px_from_exif:.1f}px (confidence: {exif_confidence:.2f})")
                current_estimation_method = "EXIF (35mm equivalent)"
            elif focal_mm is not None and sensor_width_mm is not None:
                # Use actual focal length with sensor width for direct conversion
                focal_length_px_from_exif = (focal_mm * W) / sensor_width_mm
                exif_confidence = 0.4  # Lower confidence for direct focal length
                if debug_info:
                    print(f"\n📷 Found focal length in EXIF: {focal_mm}mm → {focal_length_px_from_exif:.1f}px (confidence: {exif_confidence:.2f})")
                # Set current estimation method to EXIF
                current_estimation_method = "EXIF"

    # Fusion of EXIF and specs based on confidence
    if focal_length_px_from_exif and focal_length_px_from_specs:
        total_confidence = exif_confidence + specs_confidence
        best_available_focal_px = (
            focal_length_px_from_exif * exif_confidence +
            focal_length_px_from_specs * specs_confidence
        ) / total_confidence
    else:
        best_available_focal_px = focal_length_px_from_exif or focal_length_px_from_specs

    if debug_info:
        print(f"✅ Best available focal length estimate from EXIF and specs: {best_available_focal_px}")

    # ─── Initial depth map scaling using camera parameters ─────────────────────
    # Calculate our first guess at scale factor (s1)
    # This is critical when using EXIF focal length (which is in metric scale)
    # with a depth map in arbitrary units
    focal_length_px_plate_raw = None
    depth_scale_factor_s1 = None
    plate_scale_est = None
    detected_plate_diameter_m = None  # Store plate diameter in meters for validation

    # Find all plate instances in the masks for later use
    plate_keys = [key for key in masks if any(p_type in key.lower() for p_type in ["plate", "dish", "saucer"])]

    if best_available_focal_px:
        if debug_info:
            print(f"\n===== GEOMETRIC SCALING OF DEPTH MAP =====")
            print(f"EXIF focal length is in metric scale but depth map is unscaled")
            print(f"Applying geometric analysis to find correct depth scale factor")

        if not plate_keys:
            if debug_info:
                print("⚠️ WARNING: No plates found in the scene for geometric analysis")
                print(f"===== END GEOMETRIC PRE-SCALING =====\n")
        else:
            # Look at the first plate to determine scale factor
            plate_mask = masks[plate_keys[0]]
            plate_ellipse_fitted = _fit_ellipse_to_mask(plate_mask)

            if plate_ellipse_fitted is not None:
                # Standard plate sizes in meters for reference - dynamically extract all plate types
                standard_plate_diameters = {k: v for k, v in _REF_DIMS.items() if 'plate_' in k}

                # Find best scale factor through geometric analysis
                result = find_best_plate_scale(
                    depth, plate_mask, best_available_focal_px, standard_plate_diameters, _PLATE_PROBS, ellipse_data=plate_ellipse_fitted)

                if result is not None:
                    depth_scale_factor_s1, best_plate_size_m, scale_error, _ = result

                    if depth_scale_factor_s1 and scale_error < 0.5:
                        if debug_info:
                            print(f"✅ Geometric analysis successful")
                            print(f"Scale factor: {depth_scale_factor_s1:.6f}")
                            print(f"Most likely plate size: {best_plate_size_m*100:.1f}cm")

                        # Store the detected plate diameter for later use in validation
                        detected_plate_diameter_m = best_plate_size_m or 0.254  # Default to dinner plate size
                    else:
                        print(f"⚠️ WARNING: Geometric analysis failed or had high error ({scale_error:.3f})")
                        detected_plate_diameter_m = 0.254  # Default fallback

                    # ─── Calculate raw focal length from plate parameters ─────────────────────
                    median_depth = _get_median_depth(plate_mask, depth)
                    if detected_plate_diameter_m and median_depth > 0:
                        focal_length_px_plate_raw = 2 * plate_ellipse_fitted[1][0] * median_depth / detected_plate_diameter_m

                    if debug_info:
                        print(f"🔭 Raw focal length estimate from plate parameters: {focal_length_px_plate_raw}")
                        print(f"===== END GEOMETRIC SCALING =====\n")

                    # Convert sigma (uncertainty) to meters (same units as s)
                    if depth_scale_factor_s1:
                        sigma_s1 = depth_scale_factor_s1 * sigma_scale_exif_plate() # σ₁
                        # Store scale estimate for later combination
                        plate_scale_est = ScaleEstimate(depth_scale_factor_s1, sigma=sigma_s1, source='Specs+plate', corr_with_plate=True)

    # ─── 2️⃣  Utensil-based focal estimation ─────────────────────
    # Use reference objects (utensils) for focal length estimation
    # This method uses known dimensions of common utensils to estimate focal length

    if debug_info:
        print(f"\n ===== UTENSIL-BASED FOCAL ESTIMATION =====")

    # Initialize variables
    utensil_confidence = 0.0
    focal_length_px_refs_raw = None
    depth_scale_factor_s2 = None
    utensil_scale_est = None

    # Try to estimate focal length from utensils (knives, forks, spoons, etc.)
    try:
        focal_length_px_from_utensils = focal_from_refs(masks, depth)
        current_estimation_method = "utensils"
        utensil_confidence = 0.8  # Base confidence for utensil estimates (relatively high)
        if debug_info:
            print(f"Utensil-based focal length: {focal_length_px_from_utensils:.1f} px (confidence: {utensil_confidence:.2f})")
    except (RuntimeError, ValueError, KeyError) as e:
        focal_length_px_from_utensils = None
        if debug_info:
            print(f"Could not estimate focal length from utensils: {e}")

    # ─── Plate-based refinement ────────────────────────────
    if focal_length_px_from_utensils is None:
        if debug_info:
            print("No utensils found, skipping plate-based refinement and scale estimation")
    elif not plate_keys:
        if debug_info:
            print(f"✗ IMMEDIATE: No plates detected. Skipping utensil-based depth scale.")
        # TODO: Implement utensil-based depth scale estimation
    else:
        # Find all utensil instances for depth consistency check
        utensil_keys = [key for key in masks if any(u_type in key.lower() for u_type in ["knife", "fork", "spoon", "card"])]

        # Perform depth consistency analysis
        depth_consistency = 1.0  # Default to perfect consistency
        if utensil_keys:
            if debug_info:
                print(f"\n--- 👀 DEPTH CONSISTENCY ANALYSIS ---")
            consistency_results = _compute_depth_consistency(plate_keys, utensil_keys, masks, depth, debug_info)
            depth_consistency = consistency_results["overall_consistency"]

            if debug_info:
                print(f"Overall consistency score (1.0 = perfectly consistent): {depth_consistency:.3f}")

                # Print detailed comparison results for each plate-utensil pair
                for comp in consistency_results["comparisons"]:
                    plate_key, utensil_key = comp["objects"]
                    plate_depth, utensil_depth = comp["depths"]
                    ratio = comp["consistency"]
                    print(f"  {plate_key} ({plate_depth:.3f}) vs {utensil_key} ({utensil_depth:.3f}): {ratio:.3f}")

                # Warn about depth inconsistency
                if depth_consistency < 0.7:
                    print(f"⚠️ WARNING: Significant depth inconsistency detected! (consistency: {depth_consistency:.3f})")
                    print(f"This may affect the accuracy of focal length estimation.")

                print(f"--- END DEPTH CONSISTENCY ANALYSIS ---")

        # Handle severe depth inconsistency - early exit
        if depth_consistency < 0.6:
            if debug_info:
                print(f"\n🚨 EMERGENCY OVERRIDE: Severe depth inconsistency detected ({depth_consistency:.3f})")
                print(f"    Problem: Utensil depth values don't match plate depth values")
                print(f"    Action: Discarding utensil estimates and skipping to scale merging")
            utensil_scale_est = None
        else:
            # Process plates for focal length refinement
            focal_length_px_from_plate, plate_confidence = _process_plates_for_focal_refinement(
                plate_keys, masks, depth, focal_length_px_from_utensils, debug_info
            )

            # Combine focal length estimates using weighted fusion
            focal_length_px_refs_raw = _combine_utensil_and_plate_estimates(
                focal_length_px_from_utensils, utensil_confidence,
                focal_length_px_from_plate, plate_confidence,
                depth_consistency, utensil_keys, debug_info
            )

            # Calculate final scale factor using combined estimate
            if focal_length_px_refs_raw and plate_keys:
                plate_ellipse_data = _fit_ellipse_to_mask(masks[plate_keys[0]])
                depth_scale_factor_s2, focal_length_px_refs_scaled = calculate_scale_factor_from_refs(
                    depth, masks, plate_keys[0], plate_ellipse_data, focal_length_px_refs_raw
                )

                sigma_s2 = depth_scale_factor_s2 * sigma_scale_utensil_plate()
                utensil_scale_est = ScaleEstimate(
                    depth_scale_factor_s2, sigma=sigma_s2,
                    source='utensil+plate', corr_with_plate=True
                )

    # ─── 3️⃣  Depth scale merging ────────────────────────────
    print(f"\n===== FINAL FOCAL ESTIMATES =====")
    if debug_info:
        if depth_scale_factor_s1:
            print(f"Plate scale estimate: {depth_scale_factor_s1} ±{depth_scale_factor_s1 * sigma_scale_exif_plate()}")
        if depth_scale_factor_s2:
            print(f"Utensil scale estimate: {depth_scale_factor_s2} ±{depth_scale_factor_s2 * sigma_scale_utensil_plate()}")

    final_depth_scale_factor, depth_scale_final_sigma, note = combine_scale_estimates(plate_scale_est, utensil_scale_est)
    print(f"✅ Final depth scale factor = {final_depth_scale_factor:.5f}  ±{depth_scale_final_sigma:.5f}   ({note})")

    # ─── 4️⃣  Focal length fusion ────────────────────────────
    rel_sigma_scale = depth_scale_final_sigma / final_depth_scale_factor # from step-3 merging

    # propagate sigma
    rel_sigma_refs = np.sqrt(SIG_REL_UTENSIL**2 + rel_sigma_scale**2)
    rel_sigma_plate = np.sqrt(SIG_REL_PLATE**2 + rel_sigma_scale**2)

    # ─── Build focal estimates ────────────────────────────
    focal_length_estimates = []
    f_px_from_specs_est = None
    f_px_from_plate_est = None
    f_px_from_refs_est = None
    focal_length_px_plate_scaled = None
    focal_length_px_refs_scaled = None

    if best_available_focal_px:
        f_px_from_specs_est = FocalEstimate(best_available_focal_px, sigma=SIG_REL_EXIF * best_available_focal_px, source='exif', independent=True)
        focal_length_estimates.append(f_px_from_specs_est)

    if focal_length_px_plate_raw:
        focal_length_px_plate_scaled = focal_length_px_plate_raw * final_depth_scale_factor
        f_px_from_plate_est = FocalEstimate(focal_length_px_plate_scaled, sigma=rel_sigma_plate * focal_length_px_plate_scaled, source='plate', independent=False)
        focal_length_estimates.append(f_px_from_plate_est)

    if focal_length_px_refs_raw:
        focal_length_px_refs_scaled = focal_length_px_refs_raw * final_depth_scale_factor
        f_px_from_refs_est = FocalEstimate(focal_length_px_refs_scaled, sigma=rel_sigma_refs * focal_length_px_refs_scaled, source='references', independent=False)
        focal_length_estimates.append(f_px_from_refs_est)

    if debug_info:
        if f_px_from_specs_est:
            print(f"    🔭 Focal length estimates from specs: {best_available_focal_px} ±{SIG_REL_EXIF * best_available_focal_px:.1f}")
        if f_px_from_plate_est and focal_length_px_plate_scaled:
            print(f"    🔭 Focal length estimates from plate: {focal_length_px_plate_scaled} ±{rel_sigma_plate * focal_length_px_plate_scaled:.1f}")
        if f_px_from_refs_est and focal_length_px_refs_scaled:
            print(f"    🔭 Focal length estimates from references: {focal_length_px_refs_scaled} ±{rel_sigma_refs * focal_length_px_refs_scaled:.1f}")

    f_final_px, f_final_sigma, f_method = combine_focal_estimates(focal_length_estimates)
    print(f"✅ Final focal length (merged): {f_final_px:.1f} px ±{f_final_sigma:.1f} ({f_method})")

    return final_depth_scale_factor, f_final_px

# ================= Camera Posing ====================
def rotation_matrix_to_euler_zyx(R: np.ndarray):
    """
    Given R (3×3), compute intrinsic rotations
      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Returns roll, pitch, yaw in degrees.
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2,1],  R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0],  sy     ))
        yaw   = np.degrees(np.arctan2( R[1,0],  R[0,0]))
    else:
        # gimbal lock: set yaw=0
        roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0],  sy    ))
        yaw   = 0.0
    return roll, pitch, yaw

def camera_height_and_pitch_from_plane(
    plane_params: np.ndarray,
    ideal_normal: np.ndarray = np.array([0.0, 0.0, 1.0])
) -> tuple[float, float]:
    """
    From plane ax+by+cz+d=0 (in camera coords: +X right, +Y down, +Z forward),
    assuming the plane is observed at the image center (optical axis),
    compute:
      - height: distance from camera to plane along Z
      - pitch: rotation about camera-X axis (downward positive) in degrees

    Parameters
    ----------
    plane_params : array-like, shape (4,)
        [a, b, c, d] for the equation a x + b y + c z + d = 0.
    ideal_normal : array-like, shape (3,), optional
        The “flat” plane normal (in camera coords) that corresponds to zero pitch.
        Defaults to [0,0,1] (i.e. a plane facing the camera head-on).

    Returns
    -------
    height : float
        Distance (in same units as your point cloud) from camera origin to plane
        along the principal ray.
    pitch : float
        Camera tilt downwards (degrees). 0° means optical axis ⟂ plane;
        positive means camera pitched downward.
    """
    a, b, c, d = plane_params

    # 1) Intersection of the principal ray (x=0, y=0) with the plane:
    if np.isclose(c, 0):
        raise ValueError("c ≈ 0: plane is (nearly) vertical; principal ray never meets it.")
    height = -d / c

    # 2) Unit normal of the fitted plane
    n = np.array([a, b, c], dtype=float)
    n_unit = n / np.linalg.norm(n)

    # 3) Use the passed-in ideal_normal for zero pitch
    ideal = np.asarray(ideal_normal, dtype=float)
    ideal_unit = ideal / np.linalg.norm(ideal)

    # 4) Angle between fitted normal and ideal
    cosang = np.dot(n_unit, ideal_unit)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_rad = np.arccos(cosang)
    angle_deg = np.degrees(angle_rad)

    # 5) Sign: if plane normal points forward (c>0), table slopes up toward viewer
    #    meaning camera must pitch up (negative down), so flip sign for +down.
    pitch = -np.sign(c) * angle_deg

    return height, pitch

def compute_camera_lateral_offsets_principal_ray(plane_params: np.ndarray):
    """
    Compute how far 'back' and 'side' the camera is from the *principal‐ray hit*
    on the fitted plane, instead of the cloud centroid.

    Parameters
    ----------
    plane_params : (4,) array-like
        [a, b, c, d] for the plane a x + b y + c z + d = 0.

    Returns
    -------
    back_offset : float
        Signed distance along the table's 'back' axis (e1).
    side_offset : float
        Signed distance along the table's 'side' axis (e2).
    """
    a, b, c, d = plane_params

    # 1) Principal‐ray hit P0 = (0,0,z0)
    if np.isclose(c, 0):
        raise ValueError("Plane is vertical; principal ray never intersects.")
    z0 = -d / c
    P0 = np.array([0.0, 0.0, z0], dtype=float)

    # 2) Normal and in‐plane axes (same as before)
    n = np.array([a, b, c], dtype=float)
    n_unit = n / np.linalg.norm(n)

    # pick a ground reference (global Z)
    global_z = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n_unit, global_z)
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n_unit, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(n_unit, e1)
    e2 /= np.linalg.norm(e2)

    # 3) Vector from P0 to camera origin = -P0
    cam_vec = -P0

    # 4) Project onto e1,e2
    back_offset = np.dot(cam_vec, e1)
    side_offset = np.dot(cam_vec, e2)

    return back_offset, side_offset

def compute_full_camera_pose(plane_params_np: np.ndarray):
    """
    Given plane_params_np = [a, b, c, d] from fit_plane_to_points,
    returns the full 6-DOF camera pose relative to the principal‐ray hit:
      - translation: x (right), y (up), z (forward)
      - rotation_degrees: roll, pitch, yaw
    """
    # 1) Height & pitch (flat = +Z plane)
    height, pitch = camera_height_and_pitch_from_plane(
        plane_params_np,
        ideal_normal=np.array([0.0, 0.0, 1.0])  # flat depth‐plane normal
    )

    # 2) lateral offsets
    back, side = compute_camera_lateral_offsets_principal_ray(plane_params_np)

    # 3) build axes & rotation matrix
    n_unit, e1, e2 = plane_axes(plane_params_np)
    R_tc = np.stack((e1, e2, n_unit), axis=1)  # plane → camera
    R_ct = R_tc.T                              # camera → table

    # 4) extract roll/yaw (pitch from step 1)
    roll, _, yaw = rotation_matrix_to_euler_zyx(R_ct)

    return {
        'translation': {
            'x': float(side),    # right of table-center
            'y': float(height),  # above the table
            'z': float(back)     # forward from table-center
        },
        'rotation_degrees': {
            'roll':  roll,
            'pitch': pitch,
            'yaw':   yaw
        }
    }

# %%

def visualize_camera_and_plane_with_axes(
    table_pts: np.ndarray,
    plane_params: np.ndarray,
    grid_size: int = 10,
    alpha: float = 0.5,
    point_size: float = 5,
    axis_length: Optional[float] = None
):
    """
    Visualize camera position and table plane with camera coordinate axes.

    Same as visualize_camera_and_table_plane, but also draws the camera's
    local X/Y/Z axes as red/green/blue arrows.

    Parameters
    ----------
    table_pts : np.ndarray
        Array of table points in camera coordinates with shape (N, 3).
    plane_params : np.ndarray
        Plane parameters [a, b, c, d] where ax + by + cz + d = 0.
    grid_size : int, optional
        Number of grid points along each axis for plane visualization.
        Default is 10.
    alpha : float, optional
        Transparency level for the plane surface (0.0 to 1.0).
        Default is 0.5.
    point_size : float, optional
        Size of scatter points for table points visualization.
        Default is 5.
    axis_length : float, optional
        Length of camera axis arrows. If None, automatically computed
        based on table extent. Default is None.

    Returns
    -------
    None
        Displays the 3D plot using matplotlib.

    Notes
    -----
    The visualization uses matplotlib's 3D plotting capabilities and shows:
    - Camera position at origin (black triangle)
    - Table points as red scatter points
    - Fitted plane as a blue semi-transparent surface
    - Camera axes: X (red, right), Y (green, down), Z (blue, forward)
    """
    # compute centroid & plane axes
    P0 = table_pts.mean(axis=0)
    n_unit, e1, e2, centroid = plane_axes_and_centroid(plane_params)

    # project table_pts into u,v to get extents
    deltas = table_pts - P0
    u = deltas.dot(e1); v = deltas.dot(e2)
    umin,umax = u.min(), u.max()
    vmin,vmax = v.min(), v.max()

    # plane grid in camera coords
    u_lin = np.linspace(umin, umax, grid_size)
    v_lin = np.linspace(vmin, vmax, grid_size)
    uu, vv = np.meshgrid(u_lin, v_lin)
    grid = P0[None,None,:] + uu[:,:,None]*e1[None,None,:] + vv[:,:,None]*e2[None,None,:]
    Xc, Yc, Zc = grid[:,:,0], grid[:,:,1], grid[:,:,2]

    # remap to matplotlib coords
    Xp, Yp, Zp = Xc, -Zc, -Yc
    pts_plot = np.stack([table_pts[:,0], -table_pts[:,2], -table_pts[:,1]], axis=1)

    # figure
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # plane & points
    ax.plot_surface(Xp, Yp, Zp, color='lightblue', alpha=alpha)
    ax.scatter(pts_plot[:,0], pts_plot[:,1], pts_plot[:,2],
               c='red', s=point_size, label='table points')
    ax.scatter([0],[0],[0], c='k', s=50, marker='^', label='camera')

    # camera axes
    L = axis_length or max(umax-umin, vmax-vmin)*0.5
    # camera-local axes in camera coords
    axes_cam = np.eye(3) * L  # columns = X,Y,Z axes
    # remap each to plot coords
    axes_plot = np.stack([
        [ axes_cam[0,0], -axes_cam[2,0], -axes_cam[1,0] ],  # X axis red
        [ axes_cam[0,1], -axes_cam[2,1], -axes_cam[1,1] ],  # Y axis green
        [ axes_cam[0,2], -axes_cam[2,2], -axes_cam[1,2] ]   # Z axis blue
    ])
    colors = ['r','g','b']
    labels = ['cam X (right)', 'cam Y (down)', 'cam Z (forward)']
    for i in range(3):
        dx, dy, dz = axes_plot[i]
        ax.quiver(0,0,0, dx, dy, dz, color=colors[i], linewidth=2, arrow_length_ratio=0.1)
        ax.text(dx, dy, dz, labels[i], color=colors[i])

    ax.set_xlabel('X (m right)')
    ax.set_ylabel('Y (m forward)')
    ax.set_zlabel('Z (m up)')
    ax.set_title('Camera & Table Plane w/ Camera Axes')
    ax.legend(loc='upper left')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

def visualize_camera_and_plane_with_axes_plotly(
    table_pts: np.ndarray,
    plane_params: np.ndarray,
    grid_size: int = 10,
    alpha: float = 0.5,
    point_size: float = 5,
    axis_length: Optional[float] = None,
    title: str = "Camera & Table Plane w/ Camera Axes"
):
    """
    Interactive Plotly version of visualize_camera_and_plane_with_axes.
    Shows camera position, table plane, and camera axes with full 3D interactivity.

    Parameters
    ----------
    table_pts : np.ndarray
        Table points in camera coordinates
    plane_params : np.ndarray
        Plane parameters [a, b, c, d] where ax + by + cz + d = 0
    grid_size : int
        Size of the plane grid
    alpha : float
        Transparency of the plane surface
    point_size : float
        Size of the scatter points
    axis_length : float, optional
        Length of camera axes arrows
    title : str
        Plot title

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠️ Plotly not installed. Install with: pip install plotly")
        print("   Falling back to matplotlib version...")
        visualize_camera_and_plane_with_axes(table_pts, plane_params, grid_size, alpha, point_size, axis_length)
        return None

    # compute centroid & plane axes
    P0 = table_pts.mean(axis=0)
    # Correctly calculate plane axes using the dedicated function
    # Ensure 'plane_axes' function is defined and available in the same script or imported
    # Assuming plane_axes is defined elsewhere and takes plane_params as input
    try:
        n_unit, e1, e2, centroid = plane_axes_and_centroid(plane_params)
    except NameError:
        print("Error: 'plane_axes_and_centroid' function is not defined.")
        # Basic fallback for n_unit, e1, e2 if plane_axes is not available
        # This fallback is NOT a complete solution but allows the code to proceed
        # You should define the plane_axes function properly based on your needs
        a, b, c, d = plane_params
        n = np.array([a, b, c], dtype=float)
        n_unit = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else np.array([0., 0., 1.])
        # Simple orthogonal vectors (may not align with table axes)
        e1 = np.cross(n_unit, np.array([1., 0., 0.]))
        if np.linalg.norm(e1) < 1e-6:
             e1 = np.cross(n_unit, np.array([0., 1., 0.]))
        e1 /= np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1., 0., 0.])
        e2 = np.cross(n_unit, e1)
        e2 /= np.linalg.norm(e2) if np.linalg.norm(e2) > 1e-6 else np.array([0., 1., 0.])
        print("Using simplified plane axes - results may be incorrect.")


    # project table_pts into u,v to get extents
    deltas = table_pts - P0
    u = deltas.dot(e1); v = deltas.dot(e2)
    umin,umax = u.min(), u.max()
    vmin,vmax = v.min(), v.max()

    # plane grid in camera coords
    u_lin = np.linspace(umin, umax, grid_size)
    v_lin = np.linspace(vmin, vmax, grid_size)
    uu, vv = np.meshgrid(u_lin, v_lin)
    grid = P0[None,None,:] + uu[:,:,None]*e1[None,None,:] + vv[:,:,None]*e2[None,None,:]
    Xc, Yc, Zc = grid[:,:,0], grid[:,:,1], grid[:,:,2]

    # remap to plot coords (same as matplotlib version)
    Xp, Yp, Zp = Xc, -Zc, -Yc
    pts_plot = np.stack([table_pts[:,0], -table_pts[:,2], -table_pts[:,1]], axis=1)

    # Create figure
    fig = go.Figure()

    # Add plane surface
    fig.add_trace(go.Surface(
        x=Xp,
        y=Yp,
        z=Zp,
        opacity=alpha,
        colorscale='Blues',
        showscale=False,
        name='Table Plane',
        hovertemplate="Table Plane<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
    ))

    # Add table points
    fig.add_trace(go.Scatter3d(
        x=pts_plot[:,0],
        y=pts_plot[:,1],
        z=pts_plot[:,2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='red',
            opacity=0.8
        ),
        name='Table Points',
        hovertemplate="Table Point<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
    ))

    # Add camera position
    fig.add_trace(go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=point_size * 2,
            color='black',
            symbol='diamond',
            opacity=1.0
        ),
        name='Camera',
        hovertemplate="Camera Origin<br>X: 0.000<br>Y: 0.000<br>Z: 0.000<extra></extra>"
    ))

    # Camera axes
    L = axis_length or max(umax-umin, vmax-vmin)*0.5
    # camera-local axes in camera coords
    axes_cam = np.eye(3) * L  # columns = X,Y,Z axes
    # remap each to plot coords
    axes_plot = np.stack([
        [ axes_cam[0,0], -axes_cam[2,0], -axes_cam[1,0] ],  # X axis red
        [ axes_cam[0,1], -axes_cam[2,1], -axes_cam[1,1] ],  # Y axis green
        [ axes_cam[0,2], -axes_cam[2,2], -axes_cam[1,2] ]   # Z axis blue
    ])
    colors = ['red', 'green', 'blue']
    labels = ['cam X (right)', 'cam Y (down)', 'cam Z (forward)']

    for i in range(3):
        dx, dy, dz = axes_plot[i]
        # Add axis line
        fig.add_trace(go.Scatter3d(
            x=[0, dx],
            y=[0, dy],
            z=[0, dz],
            mode='lines+markers',
            line=dict(color=colors[i], width=6),
            # Changed 'arrow' to 'circle' as a valid symbol
            marker=dict(size=[3, 8], color=colors[i], symbol=['circle', 'circle']),
            name=labels[i],
            hovertemplate=f"{labels[i]}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>"
        ))

        # Add axis label
        fig.add_trace(go.Scatter3d(
            x=[dx * 1.1],
            y=[dy * 1.1],
            z=[dz * 1.1],
            mode='text',
            text=[labels[i]],
            textfont=dict(color=colors[i], size=12),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (m right)',
            yaxis_title='Y (m forward)',
            zaxis_title='Z (m up)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)  # Similar to elev=20, azim=-60
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
        width=900
    )

    # Add interactive instructions
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

pose = compute_full_camera_pose(plane_params_np)
print("Camera pose relative to principal-ray hit:", pose)
