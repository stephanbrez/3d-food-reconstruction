# 3d-food-reconstruction
Real-world accurate 3D food reconstruction &amp; volume estimation

## Challenges
Primary challenges:

### Unseen geometry
It's relatively straight-forward to calculate measurements for visible elements using focal-length in pixels combined with a depth-map. Monocular images "hide" the geometry on the "backside" not captured by the image plane. Projecting voxels "downward" from the camera to the surface will over-estimate volume increasingly as the camera tilt (pitch) angle reduces (0 is parallel to the surface).

### Camera intrinsics & depth
Using the pin-hole camera model it's possible to generate real-world accurate measurements from monocular images provided two things are true;
- Focal-length in pixels is accurate.
- Depth as measured from camera lens to region-of-interest (ROI) is accurate.

#### Focal Length
Image EXIF data commonly includes focal-length measured in milimeters. The pin-hole camera model requires a focal length measured in pixels. This conversion can be done using the camera sensor's dimensions and the image dimensions. Sensor measurements are often available via manufacturer specifications, but it requires the camera make and model (sometimes available in EXIF). Some cameras will include in the EXIF data the focal length converted to 35mm equivalent, which can be easily converted to pixels. All of these measurements break down if any of the data is inaccurate or missing.

#### Depth
Ground truth depth measurements can be gathered outside of the camera, but not within. Pictures taken "in the wild" will almost never have depth data measured simultaneously. Additionally, to accurately calculate dimensions using the pin-hole camera model, depth data needs correspond to every pixel in the ROI to create a depth-map. This opens up areas for inaccuracies and "noise" in the depth-map.

## Solution

### Methodology
The solution was based on a 4 key areas:
1. Depth map generation and scaling.
2. Focal length estimation via known priors.
3. Point cloud projection & measurement.
4. 3D reconstruction via 3D mesh generative models.

### Implementation
1. Depth Map generation
#### Challenges
There are many depth estimation models available. Some as single purpose, others as part of multi-modal vision-language-models (VLM). Accuracy varies along with functionality. Some models attempt to provide true metric depth estimation, while others use relative depth. Depth-Pro will even attempt to predict the focal length.

#### Solution

#### Methodology


### Implementation
2. Focal Length Estimation
#### Challenges
EXIF is inaccurate/unavailable.
Depth is at an unknown scale.
Reference objects aren't present.
#### Solution
#### Methodology
Depth Scaling Methodology:
Two scenarios:
1. EXIF focal length is correct, but depth-map is in an arbitrary and unknown scale.
2. Reference objects provide a focal length in pixels, but in the same unknown arbitrary scale as the depth-map.

Use EXIF and known priors for plates to infer depth. Scale the depth-map to match inferred depth.
Use known dimensions for refenece objects to infer focal length. Then use plates to determine scale factor.

Key Features:
- Camera pose estimation (position, orientation) from fitted planes
- Interactive visualization of camera poses and scene geometry

Focal Length Estimation from Reference Objects Methodology:
1. Reference Object-Based Estimation:
    - Uses objects with known real-world dimensions as references (credit cards, utensils, plates)
    - Intelligent utensil type validation using shape analysis
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

### Implementation
3. Point cloud projection & measurement.

#### Challenges
Points can be projected into a 3D space, which requires accurate depth estimation and geometric transformations. This 3D space is relative to the camera view, so only captured points can be projected and measured. This means that only portions of the object of interest will be in the point cloud, leading to incomplete object dimensions.

#### Solution
To address these challenges, we employ a multi-step process that includes depth estimation, geometric transformations, and outlier detection. We use a combination of machine learning algorithms and geometric constraints to estimate the depth of each point in the point cloud. We then apply geometric transformations to project the points into a 3D space. Finally, we use outlier detection to remove any points that are not part of the object of interest.
#### Methodology
Using the merged scaling factor from the focal length estimation process, scale the depth-map.
Create geometric transformations using the pin-hole camera model with the focal length from the estimation process and the scaled depth-map. Apply these transformations to project the points into a 3D space.

### Implementation
#### Challenges
#### Solution
#### Methodology
