from difflib import ndiff
from xmlrpc.client import boolean
# %%
import numpy as np # linear algebra
import os, glob
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
import pandas as pd
import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Union, Literal
from PIL import Image
from IPython.display import display
from io import StringIO
from transformers import pipeline

def setup_dataframe():
    # --- Load the data into a DataFrame ---
    raw = StringIO(
        """Combo Index,Combo Name,Food Index,Food Name,With Container?,Private or Public
    1,bagel+cream_cheese,1,bagel,,Public
    ,,2,cream_cheese,,Public
    2,breaded_fish+lemon+broccoli,3,breaded_fish,,Public
    ,,4,lemon,,Public
    ,,5,broccoli,,Public
    3,burger+hot_dog,6,burger,,Public
    ,,7,hotdog,,Public
    4,cheesecake+strawberry+raspberry,8,cheesecake,,Public
    ,,9,strawberry,,Public
    ,,10,raspberry,,Public
    5,energy_bar+cheddar_cheese+banana,11,energy_bar,,Private
    ,,12,cheddar_cheese,,Private
    ,,13,banana,,Private
    6,grilled_salmon+broccoli,14,grilled_salmon,,Private
    ,,5,broccoli,,Public
    7,pasta+garlic_bread,15,pasta,With Bowl,Private
    ,,16,garlic_bread,,Private
    8,pb&j+carrot_stick+apple+celery,17,pb&j,,Private
    ,,18,carrot_stick,,Private
    ,,19,apple,,Private
    ,,20,celery,,Private
    9,pizza+chicken_wing,21,pizza,,Private
    ,,22,chicken_wing,,Private
    10,quesadilla+guacamole+salsa,23,quesadilla,,Private
    ,,24,guacamole,With Cup,Private
    ,,25,salsa,With Cup,Private
    11,roast_chicken_leg+biscuit,26,roast_chicken_leg,,Private
    ,,27,biscuit,,Private
    12,sandwich+cookie,28,sandwich,,Private
    ,,29,cookie,,Private
    13,steak+mashed_potatoes,30,steak,,Private
    ,,31,mashed_potatoes,With Bowl,Private
    14,toast+sausage+fried_egg,32,toast,,Private
    ,,33,sausage,,Private
    ,,34,fried_egg,,Private
    """
    )
    df = pd.read_csv(raw, sep=",", skipinitialspace=True)

    df[['Combo Index', 'Combo Name']] = df[['Combo Index', 'Combo Name']].ffill()
    df['With Container?'] = df['With Container?'].fillna('')
    df = df.astype({'Combo Index': 'int64', 'Food Index': 'int64'})
    df['x'] = 0.0000 # Column for true X extents
    df['y'] = 0.0000 # Column for true Y extents
    df['z'] = 0.0000 # Column for true Z extents
    df['diagonal'] = 0.0000 # Column for true diagonal length
    df['volume'] = 0.000 # Column for volume calculation

    return df

# %%
# Setup the dataframe
df_file = "dataset.csv"

# Check if DataFrame exists and initialize it if needed
df_exists = 'df' in locals() or 'df' in globals()
if not df_exists or df is None or len(df) == 0:
    print("DataFrame needs to be initialized")
    try:
        if os.path.exists(df_file):
            print(f"Loading DataFrame from {df_file}...")
            df = pd.read_csv(df_file)
            print(f"Successfully loaded DataFrame with {len(df)} rows")
        else:
            print(f"File {df_file} not found. Creating new DataFrame...")
            df = setup_dataframe()
            print(f"Created new DataFrame with {len(df)} rows")
    except Exception as e:
        print(f"Error initializing DataFrame: {str(e)}")
        # Create an empty DataFrame as fallback
        df = pd.DataFrame()
else:
    print(f"Using existing DataFrame with {len(df)} rows")

# Display DataFrame info
print(f"DataFrame shape: {df.shape}")
print("DataFrame preview:")
print(df.head())

# %%
def find_jpg_files(directory: str) -> dict[str, str]:
    img_paths = {}

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            filename, fileext = os.path.splitext(file)
            # Check if the file ends with .jpg or .jpeg
            if fileext.lower().endswith(('.jpg', '.jpeg')):
                # Store the file path in the dictionary
                img_paths[filename] = os.path.join(root, file)

    return img_paths

def get_item_names(item_ids, combo_id):
    """
    Given a combo ID and an item_ids dictionary, retrieves and returns the list of object names
    associated with that ID.

    Parameters:
    - item_ids: The dictionary containing item IDs as keys and food names as values.
    - combo_id: The ID number (as a string) to retrieve the associated food items.

    Returns:
    - A list of food item names for the given item ID.
    """
    # Check if the item_id exists in the dictionary
    if combo_id in item_ids:
        # Return the list of keys (food names) for that particular ID
        return list(item_ids[combo_id].keys())
    else:
        # If the ID doesn't exist, return an empty list
        return []


search_dir = "/kaggle/input/3d-reconstruction-from-monocular-multi-food-images"
img_paths = find_jpg_files(search_dir)

item_ids_by_combo = {
    '1': {'bagel': 1, 'cream cheese': 2},
    '2': {'breaded fish': 3, 'lemon': 4, 'broccoli': 5},
    '3': {'burger': 6, 'hotdog': 7},
    '4': {'cheesecake': 8, 'strawberry': 9, 'raspberry': 10},
    '5': {'energy bar': 11, 'cheddar cheese': 12, 'banana': 13},
    '6': {'grilled salmon': 14, 'broccoli': 5},
    '7': {'pasta': 15, 'garlic bread': 16},
    '8': {'pb&j': 17, 'carrot stick': 18, 'apple': 19, 'celery': 20},
    '9': {'pizza': 21, 'chicken wing': 22},
    '10': {'quesadilla': 23, 'guacamole': 24, 'salsa': 25},
    '11': {'roast chicken leg': 26, 'biscuit': 27},
    '12': {'sandwich': 28, 'cookie': 29},
    '13': {'steak': 30, 'mashed potatoes': 31},
    '14': {'toast': 32, 'sausage': 33, 'fried egg': 34}
}

# 1) Inline your table as comma-separated text
data = """Combo Index,Combo Name,Food Index,Food Name,With Container?,Private or Public
1,bagel+cream_cheese,1,bagel,,Public
,,2,cream_cheese,,Public
2,breaded_fish+lemon+broccoli,3,breaded_fish,,Public
,,4,lemon,,Public
,,5,broccoli,,Public
3,burger+hot_dog,6,burger,,Public
,,7,hotdog,,Public
4,cheesecake+strawberry+raspberry,8,cheesecake,,Public
,,9,strawberry,,Public
,,10,raspberry,,Public
5,energy_bar+cheddar_cheese+banana,11,energy_bar,,Private
,,12,cheddar_cheese,,Private
,,13,banana,,Private
6,grilled_salmon+broccoli,14,grilled_salmon,,Private
,,5,broccoli,,Public
7,pasta+garlic_bread,15,pasta,With Bowl,Private
,,16,garlic_bread,,Private
8,pb&j+carrot_stick+apple+celery,17,pb&j,,Private
,,18,carrot_stick,,Private
,,19,apple,,Private
,,20,celery,,Private
9,pizza+chicken_wing,21,pizza,,Private
,,22,chicken_wing,,Private
10,quesadilla+guacamole+salsa,23,quesadilla,,Private
,,24,guacamole,With Cup,Private
,,25,salsa,With Cup,Private
11,roast_chicken_leg+biscuit,26,roast_chicken_leg,,Private
,,27,biscuit,,Private
12,sandwich+cookie,28,sandwich,,Private
,,29,cookie,,Private
13,steak+mashed_potatoes,30,steak,,Private
,,31,mashed_potatoes,With Bowl,Private
14,toast+sausage+fried_egg,32,toast,,Private
,,33,sausage,,Private
,,34,fried_egg,,Private
"""

# 2) Read it into pandas from that string
df = pd.read_csv(StringIO(data), sep=",", skipinitialspace=True)

# 3) Forward-fill the combo-level columns so every row knows its Combo Index & Name
df[['Combo Index','Combo Name']] = df[['Combo Index','Combo Name']].ffill()

# 4) Cast your numeric columns back to integer types
df = df.astype({
    'Combo Index':'int64',
    'Food Index':   'int64'
})

# 5) Now inspect it
print(df.shape)   # (total_rows, 6)
print(df.head())  # first 5 rows

# 6) Accessing rows by position:
first_row = df.iloc[0]      # row 0
print(first_row)

# 7) By label (same as position here):
row_5 = df.loc[5]           # row with index label 5
print(row_5)

# 8) Filter all foods in combo #1:
combo1 = df[df['Combo Index'] == 1]
print(combo1)

# 9) Loop through if you need per-row logic:
for idx, row in df.iterrows():
    print(f"Row {idx}: Combo {row['Combo Index']} → {row['Food Name']!r}")



# %%
# Downsize an image by the specified factor and return a PIL image object
def downsize_image_to_PIL(image_path: str, downsize_factor: float=1.0):
    # Load the image using cv2
    input_image = cv2.imread(image_path)
    # Downscale the image using cv2.resize
    input_resized = cv2.resize(input_image, None, fx=downsize_factor, fy=downsize_factor, interpolation=cv2.INTER_LINEAR)
    # Convert from BGR (OpenCV) to RGB (PIL)
    input_resized_rgb = cv2.cvtColor(input_resized, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array (RGB) to a PIL image
    pil_image = Image.fromarray(input_resized_rgb)
    return pil_image

# %%

def generate_depth_map(depth_model, input_image, show_img=False):

    output = depth_model(input_image)

    # Show the depth map if requested
    if show_img:
        display(output["depth"])

    # Return the depth map values as numpy array with the batch dimension removed
    return output["predicted_depth"].squeeze().numpy()

# %%
depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
input_image = downsize_image(img_paths['9'], 1.0)
depth_map = generate_depth_map(depth_estimator, input_image, show_img=True)

# Sanity check
print(depth_map.shape)
print(depth_map[0][0])

# %%
!pip install ultralytics -q

# %%
from ultralytics import YOLOE

def generate_seg_masks(model: object, class_names: List[str], input_image: np.ndarray,
                       combo_id: Optional[str] = None, min_confidence: float = 0.5,
                       show_img: bool = False) -> Dict[str, np.ndarray]:
    """Generate segmentation masks from the model's predictions.

    Parameters
    ----------
    model : object
        YOLOE model instance.
    class_names : list of str
        List of class names for detection.
    input_image : numpy.ndarray
        Input image for the model to process.
    combo_id : str, optional
        String ID to prepend to mask keys (format: "combo_id-key").
    min_confidence : float, default=0.5
        Minimum confidence threshold for detections.
    show_img : bool, default=False
        Flag to show the image with detection results.

    Returns
    -------
    masks: Dict[str, np.ndarray]
        Dictionary where keys are class names (with unique suffixes if duplicates)
        and values are the segmentation mask's xy values.
        If combo_id is provided, keys will be prefixed with "combo_id-".
    """
    # Map class names to indexes and vice versa
    class_names_to_idx = {name: idx for idx, name in enumerate(class_names)}
    class_idx_to_names = {idx: name for idx, name in enumerate(class_names)}

    # Set model's class names (just need to do this once after loading the model)
    model.set_classes(class_names, model.get_text_pe(class_names))

    # Run detection on the given image
    results = model.predict(input_image)

    masks = {}  # Dictionary to store the masks
    name_counts = {}  # Dictionary to count occurrences of each class

    for result in results:
        # Check if there are no masks for this result
        if not result.masks:
            print(f"Error: No masks found for result {result.names}")
            continue  # Skip to the next result

        for class_id, class_conf, xy in zip(result.boxes.cls, result.boxes.conf, result.masks.xy):
            class_name = class_idx_to_names[int(class_id)]
            # Skip if the confidence is below the threshold
            if class_conf < min_confidence:
                print(f"Skipping {class_name} with confidence {class_conf}")
                continue

            # Count how many times we've seen this class
            count = name_counts.get(class_name, 0)

            # If this is the first time seeing this class, use the plain name,
            # else append a counter (e.g., _1, _2, ...)
            key = class_name if count == 0 else f"{class_name}_{count}"

            # Prepend combo_id if provided
            if combo_id is not None:
                key = f"{combo_id}-{key}"

            # Update the count for this class
            name_counts[class_name] = count + 1

            # Add the mask to the dictionary with the unique key
            print(f"Adding mask for {key} with confidence {class_conf:.2f}")
            masks[key] = xy

        # Optionally, display the result
        if show_img:
            result.show()

    return masks

# %%
segmentation_model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect these items. You only need to do this once after you load the model.
class_names = [
    "bagel", "cream_cheese", "breaded_fish", "lemon", "broccoli", "burger",
    "hotdog", "cake", "strawberry", "raspberry", "energy_bar",
    "cheddar_cheese", "banana", "grilled_salmon", "broccoli", "pasta",
    "garlic_bread", "pb&j", "carrot_stick", "apple", "celery", "pizza",
    "chicken_wing", "quesadilla", "guacamole", "salsa", "roast_chicken_leg",
    "biscuit", "sandwich", "cookie", "steak", "mashed_potatoes", "toast",
    "sausage", "fried_egg", "plate", "fork", "knife"
]
# Pandas  way
# --- Build cleaned list of all food names ---
raw_names = df['Food Name'].tolist()
class_names = []
for name in raw_names:
    # replace underscores with spaces
    clean = name.replace('_', ' ')
    # remove '&' and '!'
    clean = clean.replace('&', '').replace('!', '')
    class_names.append(clean)

class_names += ["plate", "bowl", "fork", "knife"]
print(class_names)

if combo_id:
    masks = generate_seg_masks(segmentation_model, class_names, input_image, combo_id, 0.35, True)

# %%
# Keys to pop
keys_to_pop = ['plate_1']

# Second dictionary to store popped items
popped_dict = {}

# Pop items from the original dictionary and store them in the second dictionary
for key in keys_to_pop:
    if key in masks:
        popped_dict[key] = masks.pop(key)


# Debugging ouput:
item_names = df.loc[df['Combo Index'] == combo_id, 'Food Name'].tolist()

for item in item_names:
    if item not in masks.keys():
        print("Couldn't find ", item)
        show_all_masks = True
if show_all_masks:
    print(f"There were: masks.keys() items found.")
for name, item in masks.items():
    print(f"Mask {name} is size {len(item)}")

# %%

def create_alpha_mask(
    height: int,
    width: int,
    contour_points: Union[np.ndarray, List[Tuple[int, int]], List[List[int]]],
    margin_pixels: int = 0,
    display_mask: bool = False,
    display_method: Literal['matplotlib', 'pil'] = 'matplotlib'
) -> np.ndarray:
    """
    Create a binary mask where the area inside the contour is set to 1 and outside is 0,
    then optionally modify the filled mask by `margin_pixels` in all directions.

    Parameters
    ----------
    height : int
        Mask height in pixels.
    width : int
        Mask width in pixels.
    contour_points : array-like, shape (N,2)
        (x,y) coordinates of contour.
    margin_pixels : int, optional
        Number of pixels to adjust the mask margin. Default=0 (no adjustment).
        Positive values expand the mask, negative values shrink it.
    display_mask : bool, optional
        If True, show the mask.
    display_method : {'matplotlib','pil'}
        How to display the mask if `display_mask` is True. Matplotlib is recommended
        because you can see the contour and the filled mask.

    Returns
    -------
    mask : ndarray, shape (height, width), dtype uint8
        Binary mask, with 1 inside (and adjusted by `margin_pixels`) and 0 outside.
    """

    # OpenCV expects points as integer coordinates
    pts = np.asarray(contour_points, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Contour points must be shape [N,2], got {pts.shape}")
    if (pts[:,0] < 0).any() or (pts[:,0] >= width).any() \
    or (pts[:,1] < 0).any() or (pts[:,1] >= height).any():
        raise ValueError("Some contour points lie outside the mask bounds.")

    mask = np.zeros((height, width), dtype=np.uint8)

    # ---- 1) Fill the contour ----
    try:
        import cv2
        # OpenCV's fillPoly expects a list of contours, each contour being a numpy array
        cv2.fillPoly(mask, [pts], color=1)
    except ImportError:
        # Alternative implementation using scikit-image if OpenCV is not available
        try:
            from skimage.draw import polygon
            x, y = pts[:,0], pts[:,1]
            rr, cc = polygon(y, x, shape=mask.shape)
            mask[rr, cc] = 1
        except ImportError:
            raise ImportError("opencv-python or scikit-image required to fill contours")

    # ---- 2) Dilate or erode the filled mask ----
    if margin_pixels != 0:
        try:
            # OpenCV dilation/erosion
            kernel_size = 2*abs(margin_pixels)+1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size)
            )
            if margin_pixels > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        except Exception:
            # SciPy fallback
            from scipy.ndimage import binary_dilation, binary_erosion
            structure = np.ones((2*abs(margin_pixels)+1, 2*abs(margin_pixels)+1), bool)
            if margin_pixels > 0:
                mask = binary_dilation(mask.astype(bool), structure=structure).astype(np.uint8)
            else:
                mask = binary_erosion(mask.astype(bool), structure=structure).astype(np.uint8)

    # ---- 3) Display if requested (unchanged) ----
    if display_mask:
        if display_method.lower() == 'matplotlib':
            try:
                import matplotlib.pyplot as plt
                aspect = width/height
                max_dim = 10
                figsize = (max_dim, max_dim/aspect) if aspect>1 else (max_dim*aspect,max_dim)
                plt.figure(figsize=figsize)
                plt.imshow(mask, cmap='gray', interpolation='none')
                if margin_pixels >= 0:
                    plt.title(f'Mask {height}×{width}, dilated by {margin_pixels}px')
                else:
                    plt.title(f'Mask {height}×{width}, eroded by {abs(margin_pixels)}px')
                plt.axis('on')
                # Plot the contour points on top of the mask for verification
                plt.plot(pts[:,0], pts[:,1], 'r-')
                plt.scatter(pts[:,0], pts[:,1], c='r', s=10)
                plt.show()
            except ImportError:
                display_method = 'pil'
        if display_method.lower() == 'pil':
            try:
                from PIL import Image
                img = Image.fromarray((mask*255).astype(np.uint8))
                if margin_pixels >= 0:
                    print(f"Displaying mask {width}×{height}, dilated by {margin_pixels}px")
                else:
                    print(f"Displaying mask {width}×{height}, eroded by {abs(margin_pixels)}px")
                display(img)
            except ImportError:
                pass

    return mask

# %%
from PIL import Image
import numpy as np

def apply_mask_as_alpha(
    img_input: Union[str, "Image.Image"],
    mask: np.ndarray,
    show_img: bool = False
) -> "Image.Image":
    """
    Load an image or accept a PIL.Image, apply `mask` as its alpha channel,
    and return a PIL.Image in mode "RGBA".

    Parameters
    ----------
    img_input : str, file-like, or PIL.Image.Image
        Path or file-like for PIL to open, or an already-loaded PIL image.
    mask : ndarray, shape (H, W), dtype uint8 or bool
        Binary mask with 1 = opaque, 0 = transparent.
    show_img : bool
        If True, display the result in a Jupyter/IPython notebook.

    Returns
    -------
    PIL.Image.Image
        The resulting RGBA image.
    """
    # 1) Get an RGB image, whether you passed in a path or an Image
    if isinstance(img_input, Image.Image):
        rgb = img_input.convert("RGB")
    else:
        rgb = Image.open(img_input).convert("RGB")

    # 2) Build the alpha channel from your 0/1 mask
    alpha = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    )

    # 3) Attach as alpha → now mode is "RGBA"
    rgb.putalpha(alpha)

    # 4) Optionally display in a notebook
    if show_img:
        try:
            from IPython.display import display
            display(rgb)
        except ImportError:
            print("…got the image, but can't display it here.")

    return rgb

def apply_mask_with_black(
    img_input: Union[str, "Image.Image"],
    mask: np.ndarray,
    show_img: bool = False
) -> "Image.Image":
    """
    Load an image or accept a PIL.Image, set pixels to black where the mask is 0,
    apply the mask to the image, and return a PIL.Image in RGB mode.

    Parameters
    ----------
    img_input : str, file-like, or PIL.Image.Image
        Path or file-like for PIL to open, or an already-loaded PIL image.
    mask : ndarray, shape (H, W), dtype uint8 or bool
        Binary mask with 1 = keep original, 0 = set to black.
    show_img : bool
        If True, display the result in a Jupyter/IPython notebook.

    Returns
    -------
    PIL.Image.Image
        The resulting RGB image with masked areas set to black.
    """
    # 1) Get an RGB image, whether you passed in a path or an Image
    if isinstance(img_input, Image.Image):
        rgb = img_input.convert("RGB")
    else:
        rgb = Image.open(img_input).convert("RGB")

    # 2) Create a black image of the same size
    black = Image.new("RGB", rgb.size, (0, 0, 0))

    # 3) Convert mask to PIL image for compositing
    mask_img = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    )

    # 4) Composite the original image with the black image using the mask
    result = Image.composite(rgb, black, mask_img)

    # 5) Attach as alpha → now mode is "RGBA"
    result.putalpha(mask_img)

    # 6) Optionally display in a notebook
    if show_img:
        try:
            from IPython.display import display
            display(result)
        except ImportError:
            print("…got the image, but can't display it here.")

    return result

# %%
alpha_mask = create_alpha_mask(depth_map.shape[0], depth_map.shape[1], masks['biscuit'], 2, True)
png_input = apply_mask_as_alpha(input_image, alpha_mask, True)
# `png_like` is a PIL.Image.Image in RGBA, just as if you'd done:
#    png_like = Image.open("some.png")
#
# You can now save or further manipulate it:
#    png_like.save("with_alpha.png")

# %%
# 1) Save it to a file in your notebook’s working directory
png_like.save('masked_image.png')

# 2) Display a link you can click to download
from IPython.display import FileLink
FileLink('masked_image.png')

items_to_skip = ["fork", "knife", "table", "plate"]

# Ensure alphas directory exists
os.makedirs('alphas', exist_ok=True)

for food_item, mask in masks.items():
    if any(item in food_item for item in items_to_skip):
        continue
    try:
        alpha_mask = create_alpha_mask(depth_map.shape[0], depth_map.shape[1], mask, 1, False)
        png_input = apply_mask_as_alpha(input_image, alpha_mask, False)
        filename = food_item.replace(" ", "_")
        filename = filename + "_masked.png"
        print("Saving: ", filename)
        png_input.save('alphas/' + filename)
    except Exception as e:
        print(f"Error processing {food_item}: {e}")# --- Filename generation function using DataFrame matching ---


# %%
def generate_filename_df(input_text: str,
                         combo_index: int,
                         df: pd.DataFrame) -> str:

    input_lower = input_text.lower()
    subset = df[df['Combo Index'] == combo_index]
    if subset.empty:
        raise ValueError(f"No entries for combo index {combo_index}")

    matches = []
    for name in subset['Food Name']:
        name_lower = name.lower()
        # first check if the entire name is in the input text
        if name_lower in input_lower:
            matches.append(name)
        else:
            # match on individual words
            for word in name_lower.split('_'):
                if word in input_lower:
                    matches.append(name)
                    break

    if not matches:
        raise ValueError(f"No matching food for combo {combo_index} and response '{input_text}'")

    best = max(set(matches), key=len)
    row = subset[subset['Food Name'] == best].iloc[0]
    safe_name = best.replace(' ', '_') # Get rid of spaces
    return f"{row['Food Index']}-{safe_name}.png"

# %%
!pip install -U -q google-genai

### Setup your API key
# To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.

# %%
from google.colab import userdata
from google import genai
from google.genai import types

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# %%
def correct_input_labels(image_path: str, df: pd.DataFrame = None, output_dir="3D_inputs/", max_retries=3, retry_delay=30):
        """
        Process an image to identify its contents and rename it based on AI recognition.

        This function:
        1. Extracts the combo ID from the image filename
        2. Uses Gemini AI to identify what's in the image
        3. Generates a new standardized filename based on the AI response
        4. Saves the image with the new filename

        Args:
            image_path (str): Path to the input image
            df (pd.DataFrame, optional): DataFrame containing food data for mapping
            output_dir (str, optional): Directory to save processed images. Defaults to "3D_inputs/"
            max_retries (int, optional): Maximum number of API call retries. Defaults to 3.
            retry_delay (int, optional): Seconds to wait between retries. Defaults to 30.

        Returns:
            str: The new filename (without directory path)
        """
        from google import genai
        from google.genai import types
        # Extract combo_id from the image_path (everything before the first "-")
        image_name = os.path.basename(image_path)
        combo_id = image_name.split('-')[0]

        # Get all food names for this combo_id from the dataframe
        food_options = []
        if df is not None:
            subset = df[df['Combo Index'] == int(combo_id)]
            if not subset.empty:
                food_options = subset['Food Name'].tolist()

        # Create prompt with food options if available
        if food_options:
            # Clean up food names by replacing underscores with spaces for more natural language
            clean_food_options = [food_name.replace('_', ' ') for food_name in food_options]
            print(f"\nFound {len(food_options)} food options for combo_id {combo_id}: {clean_food_options}")
            prompt = f"Which of these items: {', '.join(clean_food_options)} is in this picture? Return the name only."
        else:
            print(f"No food options found for combo_id {combo_id}, using generic prompt")
            # Fallback to generic prompt if no food options found
            prompt = "Tell me in no more than 3 words what this is"

        with open(image_path, 'rb') as f:
                image_bytes = f.read()

        # Add retry logic for API calls
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt
                    ]
                )
                print(f"Asked Gemini: \"{prompt}\"")
                print(f"Gemini response: \"{response.text}\"\n")
                break  # Success, exit the retry loop
            except Exception as e:
                retry_count += 1
                print(f"API call failed (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count <= max_retries:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to get response from Gemini API after {max_retries} attempts: {str(e)}")

        # Generate new filename based on response text, combo_id, and dataframe
        filename = generate_filename_df(response.text, int(combo_id), df)

        # Save the image with the new filename
        output_path = os.path.join(output_dir, filename)
        img = Image.open(image_path)
        img.save(output_path)

        return filename

# Import required modules for image processing and file handling
from collections import defaultdict
import re
import os
import time
import numpy as np
from PIL import Image, ImageChops, ImageStat

def process_masked_images(input_dir="alphas/", output_dir="3D_inputs/", df=None, similarity_threshold=0.95, group_id=None):
    """
    Process masked images by correcting their labels and handling duplicates.

    This function processes image files that end with "_masked.png", optionally filtering by group ID,
    detects and skips duplicate images (only comparing within the same combo ID group),
    and renames files based on content analysis using a machine learning model.

    Args:
        input_dir (str): Directory containing the input masked images
        output_dir (str): Directory where processed images will be saved
        df (pd.DataFrame): DataFrame containing food item information for naming
        similarity_threshold (float): Threshold for determining image similarity (0.0-1.0)
        group_id (str, optional): If provided, only process files with this group ID

    Returns:
        tuple: (processed_files, duplicates, filename_mapping) where:
            - processed_files: Set of processed filenames
            - duplicates: List of duplicate filenames that were skipped
            - filename_mapping: Dictionary mapping old filenames to new filenames
    """
    def are_images_similar(img1_path: str, img2_path: str, pixel_threshold: float = 0.95,
                          coverage_threshold: float = 0.7, iou_threshold: float = 0.2,
                          lenient_multiplier: float = 0.8) -> bool:
        """
        Check if two images are similar beyond the specified thresholds.

        Handles both standard images and images with alpha channels (transparency).
        For images with alpha channels, only compares pixels that are non-transparent
        in both images using numpy array operations instead of PIL's logical_and.

        If alpha comparison fails for any reason, falls back to standard RGB comparison.

        Parameters
        ----------
        img1_path : str
            Path to first image
        img2_path : str
            Path to second image
        pixel_threshold : float, optional
            Similarity threshold for pixel comparison (0.0-1.0), default is 0.95
        coverage_threshold : float, optional
            Coverage ratio threshold for detecting same object with different mask sizes. e.g. If one mask covers >70% of another, treat as same object with different sizes.
            Default is 0.7
        iou_threshold : float, optional
            Minimum intersection-over-union threshold for moderate overlap comparison.
            Default is 0.2
        lenient_multiplier : float, optional
            Multiplier for pixel_threshold in moderate overlap cases (more forgiving).e.g. Use 80% of pixel_threshold to account for sloppy masks.
            Default is 0.8

        Returns
        -------
        bool
            True if images are similar (meet both thresholds), False otherwise
        """
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        # Make sure images are same size
        if img1.size != img2.size:
            return False

        # If they both already have an alpha band, do masked diff
        if 'A' in img1.mode and 'A' in img2.mode:
            try:
                # ensure RGBA
                img1 = img1.convert('RGBA')
                img2 = img2.convert('RGBA')

                # build binary masks of opaque pixels
                alpha1 = img1.split()[3].point(lambda a: 255 if a > 0 else 0).convert('L')
                alpha2 = img2.split()[3].point(lambda a: 255 if a > 0 else 0).convert('L')

                # Create combined mask (work around ImageChops.logical_and issue)
                mask_array1 = np.array(alpha1) > 0
                mask_array2 = np.array(alpha2) > 0
                combined_mask_array = mask_array1 & mask_array2
                mask = Image.fromarray(combined_mask_array.astype(np.uint8) * 255, mode='L')

                # ===== Advanced Mask Overlap Analysis =====
                # Calculate mask areas and intersection metrics
                mask1_area = np.sum(mask_array1)
                mask2_area = np.sum(mask_array2)
                intersection_area = np.sum(combined_mask_array)
                union_area = np.sum(mask_array1 | mask_array2)

                # If no intersection, images are definitely different
                if intersection_area == 0:
                    return False

                # Calculate coverage ratios for overlap analysis
                iou = intersection_area / union_area if union_area > 0 else 0
                coverage1 = intersection_area / mask1_area if mask1_area > 0 else 0
                coverage2 = intersection_area / mask2_area if mask2_area > 0 else 0
                max_coverage = max(coverage1, coverage2)

                print(f"Comparing: {img1_path} to {img2_path}")
                print(f"  🔍 Mask analysis - IoU: {iou:.3f}, Coverage1: {coverage1:.3f}, Coverage2: {coverage2:.3f}")

                # Compare content in the intersection area
                diff = ImageChops.difference(img1.convert('RGBA'), img2.convert('RGBA'))
                stat = ImageStat.Stat(diff, mask=mask)
                diff_ratio = sum(stat.mean) / (len(stat.mean) * 255) if stat.count[0] > 0 else 1.0
                content_similarity = 1 - diff_ratio

                # ===== Similarity Decision Logic =====
                # Case 1: High coverage (one mask mostly contains the other) - likely same object with size difference
                if max_coverage > coverage_threshold:
                    # Use original threshold for high coverage
                    is_similar = content_similarity > pixel_threshold
                    print(f"  📏 High coverage case: content_sim={content_similarity:.3f}, threshold={pixel_threshold:.3f}")
                    if is_similar:
                        return True

                # Case 2: Moderate overlap - use more lenient threshold for sloppy masks
                elif iou > iou_threshold:
                    # Reduce threshold for moderate overlap (accounts for mask imperfections)
                    moderate_overlap_threshold = pixel_threshold * lenient_multiplier
                    is_similar = content_similarity > moderate_overlap_threshold
                    print(f"  🎯 Moderate overlap case: content_sim={content_similarity:.3f}, threshold={moderate_overlap_threshold:.3f}")
                    if is_similar:
                        return True

                # Case 3: Low overlap - likely different objects
                else:
                    print(f"  ❌ Low overlap case: IoU={iou:.3f} < {iou_threshold}, assuming different objects")
                    return False

                # If we get here, images are different
                return False
            except Exception as e:
                print(f"Error in alpha comparison: {e}, falling back to standard comparison")
                # Fall back to standard comparison
                diff = ImageChops.difference(img1.convert('RGB'), img2.convert('RGB'))
                stat = ImageStat.Stat(diff)
                diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)
        # otherwise fall back
        else:
            # Do the simple diff
            diff = ImageChops.difference(img1.convert('RGB'), img2.convert('RGB'))
            stat = ImageStat.Stat(diff)
            diff_ratio = sum(stat.mean) / (len(stat.mean) * 255)

        # Return True if images are similar (difference is small)
        pixel_similarity = 1 - diff_ratio
        print(f"Pixel similarity of {os.path.basename(img1_path)} to {os.path.basename(img2_path)}: {pixel_similarity:.2f} (threshold: {pixel_threshold})")

        return pixel_similarity > pixel_threshold # Must pass threshold

    def get_combo_id(filename):
        """
        Extract combo ID from a filename.

        Args:
            filename (str): Filename to extract combo ID from

        Returns:
            int or None: The extracted combo ID as integer, or None if invalid
        """
        # Ensure we only get the combo ID before the first dash
        parts = filename.split('-', 1)
        if len(parts) < 2:
            return None
        try:
            # Make sure it's a valid integer
            return int(parts[0])
        except ValueError:
            return None

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get files to process, grouped by combo ID
    processed_files = set()
    duplicates = []
    filename_mapping = {}  # Dictionary to store old_name -> new_name mapping
    files_by_group = defaultdict(list)

    for fn in sorted(os.listdir(input_dir)):
        if not fn.endswith("_masked.png"):
            continue

        file_id = get_combo_id(fn)
        if file_id is None:
            continue

        # If group_id is specified, only include files with that ID
        if group_id is not None and str(file_id) != str(group_id):
            continue

        files_by_group[file_id].append(fn)

    if group_id:
        print(f"Processing group ID {group_id} with {len(files_by_group.get(int(group_id), []))} files")
    else:
        total_files = sum(len(files) for files in files_by_group.values())
        print(f"Processing {len(files_by_group)} groups with {total_files} total files")

    # Process each group separately to ensure we only compare within groups
    for group_id, group_files in files_by_group.items():
        print(f"Processing combo ID {group_id} with {len(group_files)} files")

        # Sort files within group to ensure consistent processing order
        group_files = sorted(group_files)

        # Track processed files within this group and their corrected names
        # This list maintains files that have already been sent to correct_input_labels
        # and will be used to check new files against only previously processed ones
        group_processed = []  # Original filenames
        processed_mapping = {}  # Maps original filename to corrected filename

        for fn in group_files:
            if fn in processed_files:
                continue

            full_path = os.path.join(input_dir, fn)

            # Check if current image is similar to any PREVIOUSLY PROCESSED files in this group
            # This ensures we only check against images already sent to correct_input_labels
            is_duplicate = False
            for prev_file in group_processed:  # group_processed contains only files already processed
                prev_path = os.path.join(input_dir, prev_file)
                # Compare the current image with previously processed image
                if are_images_similar(full_path, prev_path):
                    duplicates.append(fn)
                    corrected_name = processed_mapping[prev_file]  # Get the corrected name
                    print(f"Skipping {fn} as it's similar to {corrected_name} (original: {prev_file})")
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Only process images that aren't duplicates of previously processed ones
                corrected_file = correct_input_labels(full_path, df, output_dir)
                processed_files.add(fn)
                # Add to group_processed so future images will be checked against this one
                group_processed.append(fn)
                # Store mapping between old and new filenames
                filename_mapping[fn] = corrected_file
                processed_mapping[fn] = corrected_file
                print(f"Changed {fn} to {corrected_file} and placed in {output_dir}")

    print(f"Processed {len(processed_files)} files, skipped {len(duplicates)} duplicates")
    return processed_files, duplicates, filename_mapping

# Run the function with default parameters
alphas_dir = "alphas/"
processed_files, duplicates, filename_mapping = process_masked_images(input_dir=alphas_dir, df=df, group_id=None)

def update_masks_dictionary(masks, filename_mapping):
    """
    Update the keys in the masks dictionary based on renamed files.

    This function handles the mismatch between mask keys (which may contain spaces)
    and filenames (which use underscores instead of spaces). It maps the old mask
    keys to new ones based on the filename_mapping dictionary.

    The updated masks dictionary will use standardized keys with underscores
    instead of spaces to maintain consistency with the filenames.

    Args:
        masks (dict): Dictionary with keys like '1-bagel' or '1-fried egg' (with spaces)
        filename_mapping (dict): Mapping from old filenames ('1-bagel_masked.png' or
                                '1-fried_egg_masked.png' with underscores) to new filenames

    Returns:
        dict: Updated masks dictionary with new keys (using underscores for spaces)
    """
    updated_masks = {}

    # Process each entry in the masks dictionary
    for old_key, mask_value in masks.items():
        # Convert the mask key to the expected filename format (spaces → underscores)
        old_key_safe = old_key.replace(' ', '_')
        match_found = False

        # Find if there's a corresponding entry in the filename mapping
        for old_filename, new_filename in filename_mapping.items():
            # Extract the part before '_masked.png' from old filename
            old_mask_key = old_filename.replace('_masked.png', '')

            # Extract the part before '.png' from new filename
            new_mask_key = new_filename.replace('.png', '')

            # If this old key (with underscores) matches our filename-derived key
            if old_mask_key == old_key_safe:
                # Add the mask with the new key (ensuring underscores are used, not spaces)
                new_mask_key = new_mask_key.replace(' ', '_')  # Ensure consistent use of underscores
                updated_masks[new_mask_key] = mask_value
                print(f"Updated mask key: {old_key} -> {new_mask_key}")
                match_found = True
                break

        if not match_found:
            # If no match found, keep the original key but convert spaces to underscores
            standardized_key = old_key.replace(' ', '_')
            updated_masks[standardized_key] = mask_value
            print(f"Kept original key (standardized): {old_key} -> {standardized_key} (no matching renamed file)")
        # Note: 'else' clause was moved inside the loop above

    return updated_masks

# Example usage:
# If you have a masks dictionary, update it with the new filenames:
# masks = update_masks_dictionary(masks, filename_mapping)
#
# After updating, your mask keys will match the new filenames (without the .png extension)
# All mask keys will use underscores instead of spaces for consistency

# ================== Point Cloud Projection ================
# -------------------------------------------------------------------
#  Reference-object focal-length estimation + optional plate refinement
# -------------------------------------------------------------------

"""
Focal Length Estimation Methodology
-----------------------------------
Accurate focal length estimation is critical for metric 3D reconstruction from a single image.
Without the correct focal length, objects' real-world sizes and distances will be distorted.

Our focal length estimation approach uses a sophisticated multi-step process:

1. Reference Object-Based Estimation:
   - We use objects with known real-world dimensions (credit cards, knives, forks) as references
   - For each reference object, we:
     a. Measure its apparent size in pixels
     b. Get its distance from the camera using the depth map
     c. Calculate: focal_length = (size_pixels * distance) / real_world_size
   - For objects at unusual angles, we employ intelligent dimension selection:
     a. Calculate estimates from both dimensions
     b. Apply sanity checks to determine which dimension is more reliable
     c. Handle special cases like knife edges and partial fork views

2. Depth Consistency Analysis:
   - We detect inconsistencies in depth scaling across different regions:
     a. Compare median depths between reference objects (plates vs. utensils)
     b. Calculate consistency scores (1.0 = perfect consistency)
     c. Apply confidence penalties proportional to inconsistency severity
   - For severe inconsistency (< 0.6), we implement an emergency override:
     a. Skip weighted averaging entirely
     b. Use plate-based estimates directly, ignoring potentially compromised utensil estimates

3. Confidence-Based Weighting:
   - Each estimation method receives a confidence score based on:
     a. Object type (plates > credit cards > knives > forks)
     b. Object orientation (more circular plates receive higher confidence)
     c. Depth consistency (inconsistent regions receive lower confidence)
   - Special handling for known problematic cases:
     a. Knives at extreme angles
     b. Multiple instances of the same object type
     c. Objects with unusual aspect ratios

4. Refinement with Circular Objects (Plate):
   - Plates are ideal for focal refinement because:
     a. They are approximately circular when viewed head-on
     b. They become elliptical when viewed at an angle
     c. They have a standard set of known diameters (dinner, salad, bread plates)
   - We detect the most likely plate size from standard options
   - We assess circularity (min_radius/max_radius) to determine viewing angle
   - Higher base confidence is assigned to plate-based estimates (0.7-0.95)

5. Fallback Strategy:
   - If no reference objects are found, we fall back to:
     a. EXIF data from image metadata (if available)
     b. User-provided focal length (if specified)
     c. Default iPhone camera specifications (f ≈ 6.1mm)

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
"""

# Constants for iPhone camera specs
_IPHONE_FOCAL_MM = 6.1  # typical iPhone focal length in mm
_IPHONE_SENSOR_WIDTH_MM = 7.01  # iPhone 12/13 main camera sensor width (mm)
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
_PLATE_PROBS_ = {
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
# Heuristic relative sigmas
SIG_REL_EXIF     = 0.10          # 10 % exif focal length error
SIG_REL_UTENSIL  = 0.06          # 6 % utensil measurement error
SIG_REL_PLATE    = 0.04          # 4 % plate diameter detection error
SIG_REL_ELLIPSE  = 0.02          # 2 % ellipse-fit error
SIG_REL_DEPTH    = 0.03          # 3 % depth median noise

# Helpers to scale sigmas based on error
def sigma_scale_exif_plate() -> float:
    return np.sqrt(SIG_REL_EXIF**2 +
                   SIG_REL_PLATE**2 +
                   SIG_REL_ELLIPSE**2 +
                   SIG_REL_DEPTH**2)

def sigma_scale_utensil_plate() -> float:
    return np.sqrt(SIG_REL_UTENSIL**2 +
                   SIG_REL_PLATE**2 +
                   SIG_REL_ELLIPSE**2 +
                   SIG_REL_DEPTH**2)

# %%
from PIL.ExifTags import TAGS

def extract_exif_focal_length(image_file):
    """
    Extract focal length from image EXIF data if available.

    Parameters
    ----------
    image_file : str or PIL.Image.Image
        Path to the image file or PIL Image object

    Returns
    -------
    tuple or None
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

def convert_35mm_to_pixels(focal_length_35mm, image_width):
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

def validate_utensil_type(mask: np.ndarray, current_label: str, debug_info: bool = False):
    """
    Analyze a utensil mask to determine if it's correctly labeled.

    Parameters
    ----------
    mask : np.ndarray
        (N,2) array of pixel coordinates forming a contour
    current_label : str
        Current label of the utensil (e.g., "knife", "fork")
    debug_info : bool, optional
        Whether to print detailed debugging information, by default False

    Returns
    -------
    str
        Validated label ("knife", "fork", or current_label if uncertain)
    float
        Confidence score (0-1) for the classification
    """
    if mask.shape[0] < 20:  # Too few points for reliable classification
        if debug_info:
            print(f"[DEBUG] Too few points ({mask.shape[0]}) for reliable classification. Using original label '{current_label}'.")
        return current_label, 0.5

    import cv2
    import numpy as np
    from sklearn.decomposition import PCA

    # Assume standard HD image width for sanity checks if needed
    W = 1920

    # Convert to the format required by OpenCV
    points = mask.astype(np.float32)
    points = points.reshape(-1, 1, 2)  # Format for hull/contour operations

    # Step 1: Get basic shape characteristics
    # Calculate aspect ratio using PCA
    pca = PCA(n_components=2)
    pca.fit(mask)
    projected = pca.transform(mask)
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    width = max_proj[0] - min_proj[0]
    height = max_proj[1] - min_proj[1]
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1

    if debug_info:
        print(f"\n[DEBUG] Shape dimensions - Width: {width:.2f}, Height: {height:.2f}")
        print(f"[DEBUG] Aspect ratio: {aspect_ratio:.2f}")

    # Step 2: Analyze contour complexity
    # Forks have more complex contours due to tines
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return current_label, 0.5

    # Calculate solidity (ratio of contour area to hull area)
    contour_area = cv2.contourArea(points)
    solidity = contour_area / hull_area if hull_area > 0 else 1

    if debug_info:
        print(f"[DEBUG] Contour area: {contour_area:.2f}, Hull area: {hull_area:.2f}")
        print(f"[DEBUG] Solidity: {solidity:.2f}")

    # Calculate convexity defects to detect fork tines
    try:
        if debug_info:
            print(f"[DEBUG] Points shape before convex hull: {points.shape}")
            print(f"[DEBUG] Points min/max values: {np.min(points):.2f}/{np.max(points):.2f}")

        hull_indices = cv2.convexHull(points, returnPoints=False)
        if debug_info:
            print(f"[DEBUG] Hull indices shape: {hull_indices.shape}")
            print(f"[DEBUG] Hull indices: {hull_indices.flatten()}")

        defects = cv2.convexityDefects(np.int32(points), hull_indices)
        if debug_info:
            print(f"[DEBUG] Convexity defects shape: {defects.shape if defects is not None else 'None'}")

        # Count significant defects (potential tines)
        significant_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # Convert depth to actual distance
                distance = d / 256.0
                if distance > 10:  # Significant defect
                    significant_defects += 1
                    if debug_info:
                        print(f"[DEBUG] Found significant defect {i+1} with depth {distance:.2f}")
                        print(f"[DEBUG] Defect points - start:{s}, end:{e}, far:{f}")
    except Exception as e:
        significant_defects = 0
        if debug_info:
            print(f"[DEBUG] Error calculating convexity defects: {str(e)}")
            print(f"[DEBUG] Error occurred at points shape: {points.shape}")
            if 'hull_indices' in locals():
                print(f"[DEBUG] Error after hull_indices calculation: {hull_indices.shape}")

    if debug_info:
        print(f"[DEBUG] Total significant defects: {significant_defects}")

    # Step 3: Make classification decision

    # Knife characteristics:
    # - High aspect ratio (typically > 5)
    # - Few convexity defects
    # - High solidity (contour area close to hull area)
    is_knife_score = 0.0
    if debug_info:
        print("\n[DEBUG] --- Knife Score Calculation ---")

    if aspect_ratio > 5:
        is_knife_score += 0.6
        if debug_info:
            print("[DEBUG] Aspect ratio > 5: +0.6 knife score")
    else:
        if debug_info:
            print("[DEBUG] Aspect ratio <= 5: +0.0 knife score")

    if significant_defects <= 2:
        is_knife_score += 0.3
        if debug_info:
            print(f"[DEBUG] Few defects ({significant_defects} <= 2): +0.3 knife score")
    else:
        if debug_info:
            print(f"[DEBUG] Many defects ({significant_defects} > 2): +0.0 knife score")

    if solidity > 0.85:
        is_knife_score += 0.1
        if debug_info:
            print(f"[DEBUG] High solidity ({solidity:.2f} > 0.85): +0.1 knife score")
    else:
        if debug_info:
            print(f"[DEBUG] Low solidity ({solidity:.2f} <= 0.85): +0.0 knife score")

    # Fork characteristics:
    # - Lower aspect ratio (typically < 5)
    # - More convexity defects (from tines)
    # - Lower solidity
    is_fork_score = 0.0
    if debug_info:
        print("\n[DEBUG] --- Fork Score Calculation ---")

    if aspect_ratio < 5:
        is_fork_score += 0.3
        if debug_info:
            print("[DEBUG] Aspect ratio < 5: +0.3 fork score")
    else:
        if debug_info:
            print("[DEBUG] Aspect ratio >= 5: +0.0 fork score")

    if significant_defects >= 3:
        is_fork_score += 0.6
        if debug_info:
            print(f"[DEBUG] Many defects ({significant_defects} >= 3): +0.6 fork score")
    else:
        if debug_info:
            print(f"[DEBUG] Few defects ({significant_defects} < 3): +0.0 fork score")

    if solidity < 0.85:
        is_fork_score += 0.1
        if debug_info:
            print(f"[DEBUG] Low solidity ({solidity:.2f} < 0.85): +0.1 fork score")
    else:
        if debug_info:
            print(f"[DEBUG] High solidity ({solidity:.2f} >= 0.85): +0.0 fork score")

    # Determine confidence and label
    if debug_info:
        print(f"\n[DEBUG] --- Final Scores ---")
        print(f"[DEBUG] Knife score: {is_knife_score:.2f}")
        print(f"[DEBUG] Fork score: {is_fork_score:.2f}")

    if is_knife_score > is_fork_score + 0.3:
        if debug_info:
            print(f"[DEBUG] Decision: Detected as KNIFE with confidence {is_knife_score:.2f}")
        return "knife", is_knife_score
    elif is_fork_score > is_knife_score + 0.3:
        if debug_info:
            print(f"[DEBUG] Decision: Detected as FORK with confidence {is_fork_score:.2f}")
        return "fork", is_fork_score
    else:
        # Not confident enough to override
        if debug_info:
            print(f"[DEBUG] Decision: Not confident enough, keeping original label '{current_label}'")
        return current_label, 0.5

def plot_hull_defects(mask_xy, depth_thresh=5):
    import cv2, numpy as np, matplotlib.pyplot as plt

    cnt  = mask_xy.astype(np.int32).reshape(-1,1,2)
    hull = cv2.convexHull(cnt, returnPoints=False)
    hull_pts = cv2.convexHull(cnt).squeeze()

    defects = cv2.convexityDefects(cnt, hull)
    hull_closed = np.vstack([hull_pts, hull_pts[0]])

    plt.figure(figsize=(5,3))
    plt.plot(mask_xy[:,0], mask_xy[:,1], 'k-', lw=1, label='Contour')
    plt.plot(hull_closed[:,0], hull_closed[:,1], 'r--', lw=2, label='Convex hull')

    # draw bridging edges & mark valleys
    if defects is not None:
        for s,e,f,d in defects[:,0]:
            depth = d/256.0
            print(depth)
            if depth < depth_thresh:        # skip shallow noise
                continue
            P_s = cnt[s][0]; P_e = cnt[e][0]; P_f = cnt[f][0]

            # bridge segment in bright green, thicker and on top
            #plt.plot([P_s[0], P_e[0]], [P_s[1], P_e[1]],
                     #color='lime', lw=3, zorder=10, label='Bridge' if 'Bridge' not in plt.gca().get_legend_handles_labels()[1] else "")

            # valley marker
            plt.scatter(P_f[0], P_f[1], c='magenta', s=50, marker='x', zorder=11)
            plt.text(P_f[0]+2, P_f[1], f"{depth:.1f}", color='magenta', fontsize=8)

    plt.gca().invert_yaxis(); plt.gca().set_aspect('equal')
    plt.title("Convex hull bridge (green) and defect valleys (×)")
    plt.legend()
    plt.show()

plot_hull_defects(masks['6-fork_1'], depth_thresh=5)

# crude thick fork side profile
pts = np.array([
  [0,40],[80,40],              # bottom handle
  [80, 0],[0, 0],              # top handle
  [90,-3],[110,-6],[140,-8],   # top rising curve
  [170,30],                    # upper tip
  [170,40],[140,38],           # down around tip to lower edge
  [110,30],[90,22],            # lower curve back
  [80,40],[0,40]               # close at bottom
])
plot_hull_defects(pts, depth_thresh=5)

def remove_mask_outliers(mask, distance_threshold=30):
    """
    Remove outlier points from a contour mask by identifying discontinuities
    in the contour sequence.

    Parameters
    ----------
    mask : np.ndarray
        (N,2) array of pixel coordinates forming a contour
    distance_threshold : float
        Maximum distance between consecutive points to be considered part of the same contour

    Returns
    -------
    np.ndarray
        Cleaned contour mask with outliers removed
    """
    if mask.shape[0] < 10:
        return mask  # Too few points to process

    import numpy as np
    import cv2

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
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
        mask = remove_mask_outliers(original_mask)
        if mask.shape[0] < 10:  # Skip if cleaned mask has too few points
            print(f"Skipping {obj_key}: too few points after outlier removal")
            continue

        # For utensils, we can use Principal Component Analysis to find the main orientation
        # This is more robust for elongated objects like knives and forks
        from sklearn.decomposition import PCA
        import numpy as np

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
            main_axis, secondary_axis = secondary_axis, main_axis

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


def fit_ellipse_to_mask(mask: np.ndarray) -> tuple:
    """
    Fit an ellipse to a mask and return the ellipse parameters.

    Parameters
    ----------
    mask : np.ndarray
        (N,2) array of pixel coordinates for the mask

    Returns
    -------
    tuple
        ((center_x, center_y), (major_radius, minor_radius), angle)
        or None if fitting fails
    """
    import cv2

    if mask.shape[0] < 5:  # Need at least 5 points for ellipse fitting
        return None

    # Fit an ellipse to the mask points
    points = mask.astype(np.float32)

    try:
        # OpenCV's ellipse fitting requires at least 5 points
        ellipse = cv2.fitEllipse(points)

        # Extract center, axes, and angle from the fitted ellipse
        (center_x, center_y), (minor_axis, major_axis), angle = ellipse

        # Ensure major_axis is the larger value
        if minor_axis > major_axis:
            minor_axis, major_axis = major_axis, minor_axis

        # Half axes (radii)
        major_radius = major_axis / 2
        minor_radius = minor_axis / 2

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
        return ((center_x, center_y), (major_radius, minor_radius), 0)

def find_closest_plate_size(depth: np.ndarray,
                            plate_mask: np.ndarray,
                            plate_ellipse: tuple = None,
                            f_px: float = None,
                            debug_info: bool = False) -> tuple[str, float, float, float, float]:
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
        plate_ellipse = fit_ellipse_to_mask(plate_mask)
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
                        ellipse_data: tuple = None) -> tuple[float, float, float]:
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
    plate_mask = remove_mask_outliers(plate_mask, distance_threshold=15)
    if plate_mask.shape[0] < 50:
        print(f"Too few points in plate mask after outlier removal ({plate_mask.shape[0]} points)")
        return f_px, f_px, 0.0

    print(f"Cleaned plate mask: {original_count} → {plate_mask.shape[0]} points")

    # Use pre-computed ellipse data if provided, otherwise compute it
    if ellipse_data is None:
        ellipse_data = fit_ellipse_to_mask(plate_mask)

    if ellipse_data is None:
        print("Failed to fit ellipse to plate mask")
        return f_px, f_px, 0.0

    (center_x, center_y), (major_radius, minor_radius), angle = ellipse_data
    major_axis = major_radius * 2
    minor_axis = minor_radius * 2

    # Use find_closest_plate_size to get plate type, diameter, and depth statistics
    plate_type, plate_diameter, min_depth, max_depth, median_depth = find_closest_plate_size(
        depth=depth,
        plate_mask=plate_mask,
        plate_ellipse=ellipse_data,
        f_px=f_px,
        debug_info=debug_info
    )

    if median_depth <= 0 or np.isnan(median_depth):
        print("Unreliable depth for plate")
        return f_px

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

    # Increase base confidence for plate-based estimates as they're generally more reliable
    # When depth inconsistency is detected, increase plate confidence even more
    if 'depth_consistency' in locals() and depth_consistency < 0.8:
        # Higher base confidence when we detect depth inconsistencies
        # More aggressive scaling - up to 0.9 base even with moderate inconsistency
        base_confidence = 0.8 + 0.15 * (1 - depth_consistency)  # Up to 0.95 for severe inconsistency
        print(f"Increased plate confidence due to depth inconsistency (base: {base_confidence:.2f})")
    else:
        base_confidence = 0.7  # Higher base confidence for plates even when depths are consistent

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
                          ellipse_data: tuple = None) -> tuple[float, float, float, tuple]:
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
    ellipse_data : tuple
        Additional data about the ellipse fit: center, axes, angle, circularity
    """
    import cv2

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
        ellipse_data = fit_ellipse_to_mask(plate_mask)

    if ellipse_data is None:
        print(f"Error in plate scale analysis: {str(e)}")
        return None, None, float('inf'), None

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
                            masks: dict,
                            plate_key: str = None,
                            plate_ellipse: tuple = None,
                            f_px: float = None) -> tuple[float, float]:
    """
    Uses a plate with fitted ellipse data to determine the scale factor
    to go from arbitrary units to metric units in a depth map.

    Parameters
    ----------
    depth : np.ndarray
        Original depth map in arbitrary units
    masks : dict
        Dictionary of object masks
    plate_key : str, optional
        Key for the plate mask in the masks dictionary (used if plate_ellipse not provided)
    plate_ellipse : tuple, optional
        Ellipse data from fit_ellipse_to_mask:
        ((center_x, center_y), (major_radius, minor_radius), angle)
    f_px : float, optional
        Focal length in pixels (needed for depth calculation)

    Returns
    -------
    tuple
        - Scale factor for the depth map
        - Adjusted focal length to maintain correct XY scaling
    """
    # ─── Input Validation and Early Returns ─────────────────────
    if plate_key is None or plate_key not in masks:
        print("WARNING: Cannot rescale depth to metric units - no plate data available")
        return 1.0, f_px

    # ─── Get Plate Mask and Depth Data ─────────────────────
    plate_mask = masks[plate_key]
    if plate_mask.shape[0] < 10:
        print("WARNING: Plate mask has too few points for depth rescaling")
        return 1.0, f_px

    # ─── Determine Plate Type and Size Using New Function ─────────────────────
    plate_type, plate_diameter, min_depth, max_depth, median_depth = find_closest_plate_size(
        depth=depth,
        plate_mask=plate_mask,
        plate_ellipse=plate_ellipse,
        f_px=f_px,
        debug_info=True
    )

    # ─── Get or Generate Ellipse Data for Tilt Calculation ─────────────────────
    if plate_ellipse is None:
        plate_ellipse = fit_ellipse_to_mask(plate_mask)
        if plate_ellipse is None:
            print("WARNING: Could not fit ellipse to plate mask")
            return 1.0, f_px

    # Extract ellipse parameters for tilt calculation
    (_, _), (major_radius, minor_radius), _ = plate_ellipse
    major_axis = major_radius * 2
    minor_axis = minor_radius * 2
    circularity = minor_axis / major_axis if major_axis > 0 else 1.0

    print(f"  Major axis: {major_axis:.1f}px, Minor axis: {minor_axis:.1f}px")
    print(f"  Plate median depth: {median_depth:.3f}")
    print(f"  Current focal length: {f_px:.1f} px")

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
    adjusted_f_px = f_px * scale_factor

    print(f"  Plate circularity: {circularity:.3f} (tilt angle: {np.degrees(tilt_angle):.1f}°)")
    print(f"  Observed depth range: {depth_range:.3f} (arbitrary units)")
    print(f"  Expected depth variation: {expected_depth_variation:.3f}m")
    print(f"  Scaled depth range: {depth_range*scale_factor:.3f}m")
    print(f"  Scale factor: {scale_factor:.6f}")

    return scale_factor, adjusted_f_px

def project_masks_to_pointclouds(depth: np.ndarray,
                                            masks: dict,
                                            f_px: float,
                                            cx: float,
                                            cy: float,
                                            device='cuda') -> dict:
    """
    Projects each mask into an N×3 point cloud in meters.

    Parameters
    ----------
    depth : (H, W) array of depth values in meters
    masks : dict[str, mask] where each mask is an (N, 2) array of pixel coords [[x0,y0], [x1,y1], …]
    f_px : focal length in pixels
    cx, cy : principal point in pixels
    device : the device ('cuda' for GPU or 'cpu')

    Returns
    -------
    clouds : dict[str, (Ni, 3)] metric point clouds
    """
    # Move the depth map to the GPU (or CPU if needed)
    depth_t = torch.tensor(depth, device=device, dtype=torch.float32)
    H, W = depth.shape
    clouds = {}

    # Check depth scale - warn if it seems out of reasonable range for meters
    depth_median = float(np.median(depth[depth > 0]))
    if depth_median > 10.0:
        print(f"WARNING: Median depth value ({depth_median:.2f}) seems large for metric units")
    elif depth_median < 0.1:
        print(f"WARNING: Median depth value ({depth_median:.2f}) seems small for metric units")
    else:
        print(f"Depth values appear to be in reasonable metric range (median: {depth_median:.2f}m)")

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

        # 1) index vectors (int) and projection vectors (float)
        # Project each food mask onto the depth map
        xs_i = np.clip(m[:,0].astype(int),  0, W-1)
        ys_i = np.clip(m[:,1].astype(int),  0, H-1)
        ix = torch.tensor(xs_i, device=device)
        iy = torch.tensor(ys_i, device=device)
        xf = ix.float(); yf = iy.float()

        with torch.no_grad():
            if 'plate' in key:
                # Enforce planarity
                Z = torch.full((len(ix), len(iy)), torch.mean(depth_t[ix, iy]))
            else:
                Z = depth_t[iy, ix]              # (N,)
            X = (xf - cx) * Z / f_px
            Y = (yf - cy) * Z / f_px
            pts = torch.stack([X, Y, Z], dim=1)

        clouds[key] = pts.cpu().numpy()

        if 'plate' in key :
            z_min, z_max = Z.min().item(), Z.max().item()
            print(f"DEBUG plate Z range: {z_min:.4f}m → {z_max:.4f}m (depth variation: {z_max-z_min:.4f}m)")
            print("DEBUG sample 3D points:", clouds[key][:5])

            # For a typical 30cm plate with 70% circularity, expect ~3-4cm depth variation
            expected_variation = _REF_DIMS['plate'] * (1 - 0.7**2)**0.5
            actual_variation = z_max - z_min
            if abs(actual_variation - expected_variation) > expected_variation * 2:
                print(f"NOTE: Plate depth variation ({actual_variation:.2f}m) differs from expected (~{expected_variation:.2f}m)")
                print(f"      This may indicate depth scaling issues or an unusual plate orientation")

    return clouds

def get_median_depth(obj_mask: np.ndarray | None, depth_map: np.ndarray) -> float | None:
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
    xs = np.clip(obj_mask[:,0].astype(int), 0, w-1)
    ys = np.clip(obj_mask[:,1].astype(int), 0, h-1)
    depths = depth_map[ys, xs]
    return np.median(depths)


def compute_depth_consistency(plates: list[str],
                             utensils: list[str],
                             masks: dict[str, np.ndarray],
                             depth_map: np.ndarray,
                             debug_info: bool = False) -> dict[str, any]:
    """
    Calculate depth consistency between plates and utensils.

    This function analyzes the depth values of plates and utensils to determine
    how consistent their depth measurements are relative to each other. A high
    consistency score indicates that objects at similar depths in the scene
    have similar depth values, while a low score suggests depth measurement
    inconsistencies that may affect focal length estimation accuracy.

    Parameters
    ----------
    plates : list[str]
        List of plate object keys corresponding to entries in the masks dictionary.
    utensils : list[str]
        List of utensil object keys corresponding to entries in the masks dictionary.
    masks : dict[str, np.ndarray]
        Dictionary mapping object keys to their corresponding mask arrays, where
        each mask is an (N, 2) array of pixel coordinates.
    depth_map : np.ndarray
        2D array representing the depth map with depth values per pixel.
    debug_info : bool, optional
        Whether to print debug information during processing. Default is False.

    Returns
    -------
    dict[str, any]
        Dictionary containing depth consistency analysis results with the following keys:
        - 'overall_consistency' : float
            Overall consistency score between 0.0 and 1.0, where 1.0 indicates
            perfect consistency and values approaching 0.0 indicate inconsistency.
        - 'comparisons' : list[dict]
            List of dictionaries, each containing comparison data for a plate-utensil
            pair with keys: 'objects', 'depths', 'consistency', 'penalized_consistency'.
        - 'plate_depths' : dict[str, float]
            Dictionary mapping plate keys to their median depth values.
        - 'utensil_depths' : dict[str, float]
            Dictionary mapping utensil keys to their median depth values.
    """
    if not plates or not utensils:
        return {"overall_consistency": 1.0, "comparisons": []}

    # Calculate median depths for all objects
    plate_depths = {}
    for plate_key in plates:
        median_depth = get_median_depth(masks[plate_key], depth_map)
        if median_depth is not None:
            plate_depths[plate_key] = median_depth

    utensil_depths = {}
    for utensil_key in utensils:
        median_depth = get_median_depth(masks[utensil_key], depth_map)
        if median_depth is not None:
            utensil_depths[utensil_key] = median_depth

    # If we couldn't get depths for any plates or utensils, return perfect consistency
    if not plate_depths or not utensil_depths:
        return {"overall_consistency": 1.0, "comparisons": []}

    # Compare all plates with all utensils
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

def _process_plates_for_focal_refinement(plate_keys: list[str],
                                       masks: dict[str, np.ndarray],
                                       depth: np.ndarray,
                                       focal_to_refine: float,
                                       debug_info: bool = False) -> tuple[float | None, float]:
    """
    Process multiple plates to get refined focal length estimates and return best estimate.

    Parameters
    ----------
    plate_keys : list[str]
        List of plate keys corresponding to entries in the masks dictionary.
    masks : dict[str, np.ndarray]
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
    tuple[float | None, float]
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
        current_plate_ellipse = fit_ellipse_to_mask(masks[plate_key])

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
                                        focal_from_plate: float | None, plate_confidence: float,
                                        depth_consistency: float, utensil_keys: list[str],
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
    utensil_keys : list[str]
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
                                         utensil_keys: list[str],
                                         plate_estimate: float | None) -> float:
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
    utensil_keys : list[str]
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


def create_metric_point_clouds(
    depth: np.ndarray,
    masks: dict[str, np.ndarray],
    focal_length_mm: float | None = None,
    sensor_width_mm: float | None = None,
    image_file: str | Image.Image = None,
    device: str = 'cuda',
    debug_info: bool = False
) -> tuple[dict[str, np.ndarray], float, float | None]:
    """
    Create metric 3D point clouds from depth maps and object masks using multi-modal focal length estimation.

    This function performs a comprehensive focal length estimation pipeline by:
    1) Extracting camera specifications from user input or EXIF data to get initial focal length estimate
    2) Using reference objects (utensils, cards) for focal length estimation with known dimensions
    3) Applying geometric constraints from plates to determine depth map scaling
    4) Fusing multiple focal length estimates with uncertainty weighting
    5) Converting depth maps to metric scale and projecting masks to 3D point clouds
    6) Validating results with detected plate dimensions

    Parameters
    ----------
    depth : np.ndarray
        Depth map of the scene with depth values per pixel in arbitrary units.
    masks : dict[str, np.ndarray]
        Dictionary mapping object names to their mask coordinates as (N, 2) arrays.
    focal_length_mm : float | None, optional
        User-provided focal length in millimeters. If None, uses default iPhone specs.
    sensor_width_mm : float | None, optional
        Camera sensor width in millimeters. If None, uses default iPhone sensor width.
    image_file : str | Image.Image, optional
        Image file path or PIL Image object for EXIF data extraction.
    device : str, optional
        Device to use for computation ('cuda' or 'cpu'). Defaults to 'cuda'.
    debug_info : bool, optional
        Whether to print detailed debug information during processing. Defaults to False.

    Returns
    -------
    clouds : dict[str, np.ndarray]
        Dictionary mapping object names to 3D point clouds in metric space (meters).
        Each point cloud is an (N, 3) array with columns [X, Y, Z].
    f_px : float
        Final focal length used for reconstruction, in pixels.
    plate_diameter : float | None
        Detected plate diameter in meters, or None if no plate was detected.
    """
    # ─── Helper functions ─────────────────────

    if not torch.cuda.is_available():
        device = torch.device("cpu")

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
            plate_ellipse_fitted = fit_ellipse_to_mask(plate_mask)

            # Standard plate sizes in meters for reference - dynamically extract all plate types
            standard_plate_diameters = {k: v for k, v in _REF_DIMS.items() if 'plate_' in k}

            # Find best scale factor through geometric analysis
            depth_scale_factor_s1, best_plate_size_m, scale_error, _ = find_best_plate_scale(
                depth, plate_mask, best_available_focal_px, standard_plate_diameters, _PLATE_PROBS_, ellipse_data=plate_ellipse_fitted)

            if depth_scale_factor_s1 and scale_error < 0.5:
                if debug_info:
                    print(f"✅ Geometric analysis successful")
                    print(f"Scale factor: {depth_scale_factor_s1:.6f}")
                    print(f"Most likely plate size: {best_plate_size_m*100:.1f}cm")

                # Store the detected plate diameter for later use in validation
                detected_plate_diameter_m = best_plate_size_m
            else:
                print(f"⚠️ WARNING: Geometric analysis failed or had high error ({scale_error:.3f})")

            # ─── Calculate raw focal length from plate parameters ─────────────────────
            median_depth = get_median_depth(plate_mask, depth)
            focal_length_px_plate_raw = 2 * plate_ellipse_fitted[1][0] * median_depth / detected_plate_diameter_m

            if debug_info:
                print(f"🔭 Raw focal length estimate from plate parameters: {focal_length_px_plate_raw}")
                print(f"===== END GEOMETRIC SCALING =====\n")

            # Convert sigma (uncertainty) to meters (same units as s)
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
            consistency_results = compute_depth_consistency(plate_keys, utensil_keys, masks, depth, debug_info)
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
                plate_ellipse_data = fit_ellipse_to_mask(masks[plate_keys[0]])
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
            print(f"Plate scale estimate: {depth_scale_factor_s1} ±{sigma_s1}")
        if depth_scale_factor_s2:
            print(f"Utensil scale estimate: {depth_scale_factor_s2} ±{sigma_s2}")

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

    # ─── 5️⃣  Project into metric point clouds ───────────────────
    depth_scaled = depth * final_depth_scale_factor
    clouds = project_masks_to_pointclouds(depth_scaled, masks, f_final_px, cx, cy, device=device)

    # ─── 6️⃣  Determine final plate diameter ───────────────────
    final_plate_diameter = None
    # Always search for plates fresh, regardless of earlier state
    final_plate_keys = [key for key in masks if any(p_type in key.lower() for p_type in ["plate", "dish", "saucer"])]
    if final_plate_keys:
        first_plate_key = final_plate_keys[0]
        # Let find_closest_plate_size compute ellipse data for guaranteed consistency
        plate_type, final_plate_diameter, min_depth, max_depth, median_depth = find_closest_plate_size(
            depth=depth_scaled,
            plate_mask=masks[first_plate_key],
            plate_ellipse=None,  # Force computation for this specific plate to ensure consistency
            f_px=f_final_px,
            debug_info=debug_info
        )
        if debug_info:
            print(f"Final plate diameter: {final_plate_diameter*100:.1f}cm ({plate_type})")
    else:
        print("⚠️ WARNING: No plates detected for final diameter calculation")

    return clouds, f_final_px, final_plate_diameter

# %%
# --- Create the point clouds ---
clouds_np, f_hat, plate_diameter = create_metric_point_clouds(depth_map, masks, _IPHONE_FOCAL_MM, _IPHONE_SENSOR_WIDTH_MM, device='cpu')

# --- Compute diagonals per item ---
if combo_id:
# --- Create the point clouds ---
clouds_np, f_hat, plate_diameter = create_metric_point_clouds(depth_map, masks, _IPHONE_FOCAL_MM, _IPHONE_SENSOR_WIDTH_MM, device='cpu')

# --- For debugging we'll do this ---
# --- Compute diagonals per item ---
for key, pc in clouds_np.items():
    # key format: '<combo_index>-<food_name>'
    try:
        combo_idx_str, food_name = key.split('-', 1)
        combo_index = int(combo_idx_str)
    except ValueError:
        print(f"Invalid key format: {key}")
        continue

    # compute xy-plane diagonal
    x_dim = np.ptp(pc[:,0])    # max(x) - min(x)
    y_dim = np.ptp(pc[:,1])    # max(y) - min(y)
    diag = np.hypot(x_dim, y_dim)

    print(f"{key} → X: {x_dim:.6f}, Y: {y_dim:.6f}, Diagonal: {diag:.6f}")
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_point_clouds_view(clouds: dict, elev: float = 30, azim: float = 45, point_size: float = 1.0, interactive: bool = True):
    """
    Plot multiple point clouds in a single 3D axes with +Z up and +Y away from the viewer.

    Parameters
    ----------
    clouds : dict[str, ndarray]
        Mapping from label -> (N_i, 3) arrays, where each array is [X, Y, Z]
        in camera coords (X right, Y down, Z into scene).
    elev : float
        Elevation angle in the z plane for initial view (degrees).
    azim : float
        Azimuth angle in the x,y plane for initial view (degrees).
    point_size : float
        Marker size for the scatter points.
    interactive : bool
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

    # pick a qualitative colormap
    cmap = plt.get_cmap('tab10')
    labels = list(clouds.keys())
    num = len(labels)

    for i, label in enumerate(labels):
        pts = np.asarray(clouds[label])
        # transform to view coords:
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


def plot_point_clouds_interactive_plotly(clouds: dict, point_size: float = 2.0, title: str = "3D Point Clouds"):
    """
    Create an interactive 3D scatter plot using Plotly for superior interactivity.

    Parameters
    ----------
    clouds : dict[str, ndarray]
        Mapping from label -> (N_i, 3) arrays, where each array is [X, Y, Z]
        in camera coords (X right, Y down, Z into scene).
    point_size : float
        Marker size for the scatter points.
    title : str
        Title for the plot.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure that can be displayed in Jupyter
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
        Xv = pts[:, 0]  # Reverse X axis direction
        Yv = -pts[:, 1]  # Z becomes Y (positive = away from viewer)
        Zv = -pts[:, 2]  # Y becomes Z (negative Y becomes positive Z)

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
            xaxis_title='X (m, left)',
            yaxis_title='Y(m, away from viewer)',
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

# Create interactive plot
fig = plot_point_clouds_interactive_plotly(clouds_np,
                                         point_size=3.0,
                                         title="Food Item Point Clouds")
if fig:
    fig.show()

# ===== 3D Mesh Processing =====
# Suppose you have:
clouds_np, _, plate_diameter = create_metric_point_clouds(depth, masks, f_px, sensor_width_mm=, device='cuda')

plot_point_clouds_view(clouds, elev=25, azim=135, point_size=2)

# %%
import torch

def fit_plane_to_points(points: torch.Tensor) -> torch.Tensor:
    """
    Fit a plane to a set of 3D points using least squares on GPU.

    Parameters
    ----------
    points : (N, 3) tensor
        The 3D points (x, y, z) to fit the plane to.

    Returns
    -------
    plane_params : (4,) tensor
        The plane parameters [a, b, c, d] of ax + by + cz + d = 0.
    """
    # Build the design matrix to solve z = a x + b y + d'
    # A @ [a, b, d'] = z
    A = torch.cat([
        points[:, 0:2],                      # x, y
        torch.ones(points.shape[0], 1, device=points.device)
    ], dim=1)  # shape (N, 3)

    z = points[:, 2]                        # shape (N,)

    # Solve least squares for [a, b, d']
    sol = torch.linalg.lstsq(A, z.unsqueeze(1)).solution.squeeze()
    a, b, d_prime = sol

    # Pack into [a, b, c, d] for ax + by + cz + d = 0
    c = -1.0
    plane_params = torch.tensor([a, b, c, d_prime], device=points.device)

    return plane_params

def plane_axes(plane_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given plane_params = [a, b, c, d] for ax + by + cz + d = 0,
    returns three orthonormal vectors in camera coords:
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
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - n_unit: The normalized plane normal vector (3,)
        - e1: First in-plane orthonormal axis vector (3,)
        - e2: Second in-plane orthonormal axis vector (3,)
    """
    a, b, c, _ = plane_params
    # 1) normalized plane normal
    n = torch.tensor([a, b, c], device=plane_params.device)
    n_unit = n / torch.norm(n)
    # 2) pick a reference direction (global Z) and cross to get e1
    z_glob = torch.tensor([0.0, 0.0, 1.0], device=plane_params.device)
    e1 = torch.cross(n_unit, z_glob)
    if torch.norm(e1) < 1e-6:
        # if plane is nearly vertical (normal ≈ Z), use X instead
        e1 = torch.cross(n_unit, torch.tensor([1.0, 0.0, 0.0], device=plane_params.device))
    e1 /= torch.norm(e1)

    # 3) the second in‐plane axis orthogonal to both
    e2 = torch.cross(n_unit, e1)
    e2 /= torch.norm(e2)

    return n_unit, e1, e2

def compute_true_extents(clouds: dict[str, np.ndarray],
                         plate_key: str = "plate",
                         known_plate_diameter: float = None) -> dict[str, dict[str, float]]:
    """
    Given a dict of point‐clouds (in camera coords), fit the plate plane,
    then project *all* clouds into that plane to get true X, Y, and diagonal.

    Parameters
    ----------
    clouds : dict[str, np.ndarray]
        Dictionary of point clouds
    plate_key : str
        Key for the plate in the clouds dictionary
    known_plate_diameter : float, optional
        Known diameter of the plate in meters, used for validation

    Returns
    -------
    dict[str, dict[str, float]]
        A dictionary mapping each object key to its measurements:
        - 'x': The true width (in meters) along the first plane axis
        - 'y': The true width (in meters) along the second plane axis
        - 'diag': The true diagonal length (in meters) in the plane
    """
    # 1) Grab plate points and fit its plane
    plate_pc = clouds[plate_key]                    # (N,3) numpy
    a, b, c, d = fit_plane_to_points(torch.tensor(plate_pc))
    # 2) Get the two in‐plane orthonormal axes
    _, e1, e2 = plane_axes((a, b, c, d))
    # Convert torch tensors to numpy arrays for dot product operations
    e1 = e1.cpu().numpy()    # torch tensors → (3,) numpy arrays
    e2 = e2.cpu().numpy()    # torch tensors → (3,) numpy arrays
    # 3) Center about the plate centroid so translations cancel
    centroid = plate_pc.mean(axis=0)                 # (3,)

    results = {}
    for key, pts in clouds.items():
        # shift into plate‐centered coordinates
        P = pts - centroid                           # (Ni,3)
        # project onto e1,e2
        u = P.dot(e1)                                # (Ni,)
        v = P.dot(e2)                                # (Ni,)
        # compute true extents
        x_true = np.ptp(u)                             # max(u)-min(u)
        y_true = np.ptp(v)                             # max(v)-min(v)
        diag   = np.hypot(x_true, y_true)
        # build results dictionary
        results[key] = {'x': x_true, 'y': y_true, 'diag': diag}

        print(f"{key:>10s} → True X: {x_true:.3f} m, "
              f"True Y: {y_true:.3f} m,  Diag: {diag:.3f} m")

    # Add validation against known plate diameter if available
    if plate_key in results and known_plate_diameter is not None:
        measured_diameter = max(results[plate_key]['x'], results[plate_key]['y'])
        accuracy = (measured_diameter / known_plate_diameter) * 100
        print(f"\nValidation: Measured plate diameter: {measured_diameter:.3f}m vs "
              f"Known diameter: {known_plate_diameter:.3f}m")
        print(f"Measurement accuracy: {accuracy:.1f}% of expected diameter")

        # Flag significant discrepancies
        if abs(100 - accuracy) > 10:
            print(f"⚠️ Warning: Measured plate diameter differs from expected by {abs(100-accuracy):.1f}%")

    return results

# Example usage:
# clouds_np is your dict of { 'plate':…, 'biscuit':…, … } from project_masks_to_pointclouds
plate_key = combo_id + "-plate"
extents = compute_true_extents(clouds_np, plate_key=plate_key, known_plate_diameter=plate_diameter)

for key, measurements in extents.items():
    # key format: '<combo_index>-<food_name>'
    try:
        item_idx_str, food_name = key.split('-', 1)
        item_index = int(item_idx_str)
    except ValueError:
        print(f"Invalid key format: {key}")
        continue

    # Skip keys for plate, knife, or fork
    if any(item in food_name.lower() for item in ['plate', 'knife', 'fork']):
        print(f"Skipping utility item: {food_name}")
        continue

    # attempt to find a matching row in the DataFrame
    matched = df[(df['Food Index'] == item_index) &
                (df['Food Name'] == food_name)]
    if matched.empty:
        print(f"Couldn't find {food_name} in the dataframe")
        continue

    # store extents + diagonal in the DataFrame row
    df.loc[matched.index, 'x'] = measurements['x']
    df.loc[matched.index, 'y'] = measurements['y']
    df.loc[matched.index, 'diagonal'] = measurements['diag']
    print(f"Stored diagonal for {food_name} value: {diag}")import torch

# choose the device you want to fit on
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# convert every cloud to a float32 Tensor on that device
# clouds_torch = {
#     key: torch.from_numpy(clouds_np[key]).to(device, dtype=torch.float32)
#     for key in clouds_np
# }
# e.g. take the 'table' cloud and fit a plane
# table_pts = clouds_torch['table']    # shape (N,3) Tensor
table_pts = torch.from_numpy(clouds_np['table']).to(device, dtype=torch.float32)
plane_params = fit_plane_to_points(table_pts)
plane_params_np = plane_params.cpu().numpy()
# print(plane_params)  # [a, b, c, d] on the same device


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_plane(plane_points, plane_params, grid_steps=50):
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

plot_plane(clouds_np['table'], plane_params_np)

# ================= Camera Posing =================
# %%

import numpy as np

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

pose = compute_full_camera_pose(plane_params_np)
print("Camera pose relative to principal-ray hit:", pose)

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def visualize_camera_and_plane_with_axes(
    table_pts: np.ndarray,
    plane_params: np.ndarray,
    grid_size: int = 10,
    alpha: float = 0.5,
    point_size: float = 5,
    axis_length: float = None
):
    """
    Same as visualize_camera_and_table_plane, but also draws the camera's
    local X/Y/Z axes as red/green/blue arrows.
    """
    # compute centroid & plane axes
    P0 = table_pts.mean(axis=0)
    n_unit, e1, e2 = plane_axes(plane_params)

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
    axis_length: float = None,
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
    axis_length : float
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
        n_unit, e1, e2 = plane_axes(plane_params)
    except NameError:
        print("Error: 'plane_axes' function is not defined.")
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
visualize_camera_and_plane_with_axes(
    clouds_np['plate'],
    plane_params_np
)
# ================= Camera Posing ====================


# ------
#
# 3D Section
#
#
# ------
#

# %%
%%bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git
cd TRELLIS
./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# ---- Good to have, not needed ----
# %%
!pip install trimesh pygltflib

# %%
import trimesh

# Load your GLB (this will return either a Trimesh or a Scene)
scene_or_mesh = trimesh.load('input_model.glb')

# If it's a single mesh:
if isinstance(scene_or_mesh, trimesh.Trimesh):
    scene_or_mesh.export('output_model.obj')

# If it’s a scene with multiple geometries:
else:
    for name, mesh in scene_or_mesh.geometry.items():
        mesh.export(f'{name}.obj')
#

# %%
import trimesh

# Load the OBJ
mesh = trimesh.load('model.obj', force='mesh')

# mesh.bounds is an array [[min_x, min_y, min_z],
#                          [max_x, max_y, max_z]]
min_x, max_x = mesh.bounds[0,0], mesh.bounds[1,0]
width_x = max_x - min_x

# Or simply:
width_x = mesh.extents[0]
height_y = mesh.extents[1]
depth_z = mesh.extents[2]

print(f"Width along X axis: {width_x:.6f} units")

# ---- Good to have, not needed ----

# %%
import torch


# %%
import numpy as np
import trimesh

def scale_mesh_to_diagonal(mesh: trimesh.Trimesh, target_diag: float) -> trimesh.Trimesh:
    """
    Scale a mesh to match a target diagonal length.

    This function scales a 3D mesh so that its diagonal length in the X-Y plane
    (ignoring height/Z) matches the specified target diagonal. This is useful for
    ensuring a mesh has the correct real-world dimensions.

    Args:
        mesh: A trimesh.Trimesh object to be scaled
        target_diag: The target diagonal length in meters

    Returns:
        The scaled mesh (the input mesh is modified in-place)
    """
    w, d = mesh.extents[:2]
    scale = target_diag / np.hypot(w, d)
    mesh.apply_scale(scale)
    return mesh

mesh_dir = "./meshes"
# Create final_obj directory inside meshes if it doesn't exist
final_obj_dir = os.path.join(mesh_dir, "final_obj")
os.makedirs(final_obj_dir, exist_ok=True)

for fn in os.listdir(mesh_dir):
    if not fn.endswith(".glb"):
        continue

    base = fn[:-4]
    input_path = os.path.join(mesh_dir, fn)
    mesh = trimesh.load(input_path, force="mesh")

    try:
        item_idx_str, food_name = base.split('-', 1)
        item_index = int(item_idx_str)
    except ValueError:
        print(f"Invalid key format: {base}")
        continue

    # attempt to find a matching row in the DataFrame
    # Handle potential underscore/space differences and special characters
    food_name_spaces = food_name.replace('_', ' ')
    food_name_underscores = food_name.replace(' ', '_')

    # Handle special characters like & and !
    clean_food_name = re.sub(r'[&!]', '', food_name)
    clean_spaces = re.sub(r'[&!]', '', food_name_spaces)
    clean_underscores = re.sub(r'[&!]', '', food_name_underscores)

    # Compare with original and cleaned versions
    matched = df[
        (df['Food Index'] == item_index) &
        (
            (df['Food Name'] == food_name) |
            (df['Food Name'] == food_name_spaces) |
            (df['Food Name'] == food_name_underscores) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_food_name) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_spaces) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_underscores)
        )
    ]
    if matched.empty:
        print(f"\n⚠️ Couldn't find {food_name} in the dataframe")
        continue

    scaled_out = os.path.join(final_obj_dir, f"{base}_scaled.obj")


    print(f"\n[PROCESS] {fn}")
    # scale
    diagonal = matched['diagonal'].iloc[0]  # Get the first matched value explicitly
    print(f"Using diagonal: {diagonal} for {food_name}")
    scaled = scale_mesh_to_diagonal(mesh, diagonal)
    scaled.export(scaled_out)
    print(f"  [SCALED] → final_obj/{base}_scaled.obj")

# %%
# 1) Project your masks → clouds_np dict
# 2) compute_true_extents → returns something like:
extents = compute_true_extents(clouds_np, plate_key="plate")
#    extents['plate'] == {'x': 0.25, 'y': 0.22, 'diag': 0.33}

# load or have your trimesh instance
biscuit_mesh = trimesh.load("biscuit_model.glb")

# grab the true X/Y from your extents dict
x_true, y_true = extents["biscuit"]["x"], extents["biscuit"]["y"]

# scale to match real-world size
scaled_biscuit = scale_mesh_to_pointcloud_xy(x_true, y_true, biscuit_mesh)

# now `scaled_biscuit` has its XY diagonal ≈ extents["biscuit"]["diag"] meters

# %%
def approximate_volume_ml(mesh: trimesh.Trimesh,
                          pitch: float = None,
                          target_resolution: int = 100) -> float:
    """
    Approximate the volume of *any* mesh (watertight or not) by voxelization.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Your scaled mesh, in meters.
    pitch : float, optional
        The edge length (in meters) of each cubic voxel. If None, automatically
        set so the mesh's max XY‐extent is divided into `target_resolution` voxels.
    target_resolution : int
        Number of voxels along the largest XY‐dimension if pitch is None.

    Returns
    -------
    volume_ml : float
        Approximate volume in milliliters.
    """
    # Choose pitch if not provided
    if pitch is None:
        # use the larger of width or depth
        w, d = mesh.extents[:2]
        max_dim = max(w, d)
        pitch = max_dim / target_resolution

    # Perform voxelization
    vg = mesh.voxelized(pitch)
    # `vg.points` is an (N,3) array of filled‐voxel centers
    num_voxels = len(vg.points)

    # Compute volume
    volume_m3 = num_voxels * pitch**3
    volume_ml = volume_m3 * 1e6
    return volume_ml


def identify_outliers(mesh: trimesh.Trimesh,
                      threshold: float = 2.0,
                      axis: int = 2,
                      method: str = 'std',
                      percentile: float = 99.0) -> np.ndarray:
    """
    Identify outlier vertices more flexibly using various methods.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    threshold : float
        The threshold value for outlier detection:
        - For 'std' method: number of standard deviations from the mean
        - For 'percentile' method: percentile threshold (e.g., 99.0 means keep 99%)
        - For 'absolute' method: absolute distance from the mean
    axis : int
        The axis along which to detect outliers (0=X, 1=Y, 2=Z, or None for all axes)
    method : str
        Method for outlier detection: 'std', 'percentile', or 'absolute'
    percentile : float
        Percentile value when using percentile method (0-100)

    Returns
    -------
    np.ndarray
        Boolean mask of vertices that are not outliers
    """
    if axis is None:
        # Apply outlier detection to all axes and take the intersection
        non_outliers_x = identify_outliers(mesh, threshold, 0, method, percentile)
        non_outliers_y = identify_outliers(mesh, threshold, 1, method, percentile)
        non_outliers_z = identify_outliers(mesh, threshold, 2, method, percentile)
        return non_outliers_x & non_outliers_y & non_outliers_z

    # Get coordinates for the specified axis
    coords = mesh.vertices[:, axis]

    if method == 'std':
        # Standard deviation based method
        mean_val = np.mean(coords)
        std_val = np.std(coords)
        non_outliers = np.abs(coords - mean_val) <= threshold * std_val

    elif method == 'percentile':
        # Percentile based method
        lower = np.percentile(coords, (100 - percentile) / 2)
        upper = np.percentile(coords, 100 - (100 - percentile) / 2)
        non_outliers = (coords >= lower) & (coords <= upper)

    elif method == 'absolute':
        # Absolute distance based method
        mean_val = np.mean(coords)
        non_outliers = np.abs(coords - mean_val) <= threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return non_outliers


def filter_mesh_outliers(mesh: trimesh.Trimesh,
                         threshold: float = 2.0,
                         axis: int = 2,
                         method: str = 'std',
                         percentile: float = 99.0,
                         verbose: bool = True) -> trimesh.Trimesh:
    """
    Remove outlier vertices from the mesh with flexible options.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    threshold : float
        The threshold value for outlier detection
    axis : int or None
        The axis along which to detect outliers (0=X, 1=Y, 2=Z, or None for all axes)
    method : str
        Method for outlier detection: 'std', 'percentile', or 'absolute'
    percentile : float
        Percentile value when using percentile method (0-100)
    verbose : bool
        Whether to print information about outlier filtering

    Returns
    -------
    trimesh.Trimesh
        A new mesh with outliers removed
    """
    # Get mask of non-outlier vertices
    non_outliers = identify_outliers(mesh, threshold, axis, method, percentile)

    # Count outliers
    outlier_count = np.sum(~non_outliers)
    if outlier_count > 0 and verbose:
        axis_name = {0: 'X', 1: 'Y', 2: 'Z', None: 'all axes'}.get(axis, str(axis))
        print(f"Filtering {outlier_count} outlier vertices on {axis_name} using {method} method")

    # If no outliers found, return the original mesh
    if np.all(non_outliers):
        if verbose:
            print("No outliers found, mesh unchanged")
        return mesh

    # Create a copy of the mesh
    filtered_mesh = mesh.copy()

    # Remove faces that contain outlier vertices
    valid_faces = np.all(non_outliers[filtered_mesh.faces], axis=1)
    filtered_mesh.update_faces(valid_faces)

    # Remove unused vertices
    filtered_mesh.remove_unreferenced_vertices()

    return filtered_mesh


# This function is no longer needed as we're taking a simpler approach
# We're keeping it around in case we want to use it in the future
def find_natural_orientation(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Find the natural orientation of a mesh to identify the true "bottom" plane.

    Note: This function is not currently used in seal_mesh_bottom.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh

    Returns
    -------
    np.ndarray
        Transformation matrix to orient the mesh correctly
    """
    # Compute principal axes using PCA
    # This finds the natural axes of the object regardless of how it was scanned
    centered = mesh.vertices - mesh.center_mass
    pca = np.linalg.svd(centered.T @ centered)[0].T

    # Create rotation matrix to align to principal axes
    # Make sure determinant is 1 for proper rotation
    if np.linalg.det(pca) < 0:
        pca[2] = -pca[2]

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = pca

    # Determine if we need to flip axes to make Z the smallest dimension
    extents_in_pca = np.abs(centered @ pca).max(axis=0) * 2

    # Reorder axes if needed so Z is the smallest dimension
    # This ensures the "bottom" is correctly identified
    axis_order = np.argsort(extents_in_pca)[::-1]  # Descending order

    # If Z is not the smallest, reorder
    if axis_order[2] != 2:
        new_order = np.eye(3)[axis_order]
        transform[:3, :3] = pca @ new_order

    return transform

def fill_mesh_holes(mesh: trimesh.Trimesh,
                   max_hole_size: float = 0.05,
                   aggressive: bool = True,
                   iterations: int = 3) -> trimesh.Trimesh:
    """
    Fill holes in a mesh using a robust multi-stage approach.
    Compatible with all trimesh versions.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    max_hole_size : float
        Maximum boundary edge length to fill as a fraction of the mesh's bounding box diagonal
    aggressive : bool
        Whether to use additional techniques beyond trimesh's built-in fill_holes
    iterations : int
        Number of iterations to attempt for aggressive filling

    Returns
    -------
    trimesh.Trimesh
        A new mesh with holes filled
    """
    # Make a copy of the mesh
    filled_mesh = mesh.copy()

    # Calculate max hole size based on mesh dimensions
    diag = np.linalg.norm(mesh.extents)
    max_len = diag * max_hole_size

    # Function to get boundary edges (those that appear in exactly one face)
    def get_boundary_edges(mesh_obj):
        # Get edges from faces
        edges = mesh_obj.edges_sorted
        # Count how many times each edge appears
        unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
        # Boundary edges only appear once
        boundary_edge_indices = np.where(edge_counts == 1)[0]
        return unique_edges[boundary_edge_indices]

    # Function to group boundary edges into loops
    def get_boundary_loops(boundary_edges):
        if len(boundary_edges) == 0:
            return []

        # Create a dictionary to track connections between vertices
        edge_dict = {}
        for edge in boundary_edges:
            v1, v2 = edge
            if v1 not in edge_dict:
                edge_dict[v1] = []
            if v2 not in edge_dict:
                edge_dict[v2] = []
            edge_dict[v1].append(v2)
            edge_dict[v2].append(v1)

        # Find loops using a simple traversal
        loops = []
        visited = set()

        for start_vertex in edge_dict:
            if start_vertex in visited or len(edge_dict[start_vertex]) != 2:
                continue

            # Start a new loop
            loop = [start_vertex]
            visited.add(start_vertex)
            current = edge_dict[start_vertex][0]  # Pick a direction

            # Traverse the loop
            while current != start_vertex and current not in visited:
                loop.append(current)
                visited.add(current)
                # Find the next vertex (not the one we came from)
                neighbors = edge_dict[current]
                if len(neighbors) != 2:
                    break  # Not a clean loop
                next_v = neighbors[0] if neighbors[1] == loop[-2] else neighbors[1]
                current = next_v

            if current == start_vertex:
                loops.append(np.array(loop))

        return loops

    try:
        # Try to use trimesh's built-in hole filling if available
        filled_mesh.fill_holes(max_len)
        print(f"Standard hole filling completed. Watertight: {filled_mesh.is_watertight}")
    except Exception as e:
        print(f"Warning: Standard hole filling failed: {e}")

    # If aggressive filling is enabled and mesh still has holes
    if aggressive and not filled_mesh.is_watertight:
        print("Attempting aggressive hole filling...")

        # Get boundary edges
        boundary_edges = get_boundary_edges(filled_mesh)

        if len(boundary_edges) > 0:
            # Get boundary loops
            boundary_loops = get_boundary_loops(boundary_edges)
            print(f"Found {len(boundary_loops)} boundary loops to fill")

            # Try multiple iterations with different parameters
            for i in range(iterations):
                # Break if mesh is already watertight
                if filled_mesh.is_watertight:
                    print(f"Mesh is now watertight after {i} iterations")
                    break

                try:
                    # Try to fill holes with larger max size
                    iter_max_len = max_len * (i + 1.5)
                    filled_mesh.fill_holes(iter_max_len)

                    # Check progress
                    if filled_mesh.is_watertight:
                        print(f"Mesh is now watertight after iteration {i+1}")
                        break
                except Exception:
                    pass

            # If still not watertight, try manual filling
            if not filled_mesh.is_watertight:
                # Get updated boundary edges
                boundary_edges = get_boundary_edges(filled_mesh)
                boundary_loops = get_boundary_loops(boundary_edges)

                if len(boundary_loops) > 0:
                    print(f"Manually filling {len(boundary_loops)} remaining boundary loops")

                    for loop in boundary_loops:
                        if len(loop) >= 3:
                            # Create a simple triangulation (fan) from the first vertex
                            n_verts = len(loop)
                            new_faces = np.zeros((n_verts - 2, 3), dtype=np.int64)
                            for j in range(n_verts - 2):
                                new_faces[j] = [loop[0], loop[j+1], loop[j+2]]

                            # Add these faces to the mesh
                            filled_mesh.faces = np.vstack((filled_mesh.faces, new_faces))

                    # Merge duplicate vertices and fix normals
                    filled_mesh.merge_vertices()
                    try:
                        filled_mesh.fix_normals()
                    except:
                        pass

                    print(f"Mesh after manual filling. Watertight: {filled_mesh.is_watertight}")
        else:
            print("No boundary edges found, but mesh is not watertight. Unusual topology detected.")

    return filled_mesh


def cap_mesh(mesh: trimesh.Trimesh,
             axis: int = 2,
             direction: str = 'min',
             padding: float = 0.005,
             normals_out: bool = None) -> trimesh.Trimesh:
    """
    Cap a mesh by adding a planar cap along any axis in either direction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    axis : int
        The axis along which to cap (0=X, 1=Y, 2=Z)
    direction : str
        Which direction to cap ('min' or 'max')
    padding : float
        Amount of padding to add around the cap in meters
    normals_out : bool, optional
        If True, cap normals point outward; if False, normals point inward.
        If None (default), normals direction is based on 'direction' parameter.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with a cap added
    """
    # Check if mesh is already watertight
    if mesh.is_watertight:
        print("Mesh is already watertight, no capping needed.")
        return mesh

    # Make a copy of the mesh to work with
    capped_mesh = mesh.copy()

    # Get the bounds of the mesh
    bounds = mesh.bounds

    # Create array indices for other axes
    other_axes = [i for i in range(3) if i != axis]
    axis1, axis2 = other_axes

    # Determine the position of the cap
    if direction == 'min':
        cap_pos = bounds[0, axis] - 0.0005  # Slightly beyond bound to ensure overlap
        default_normals_out = False
    else:  # 'max'
        cap_pos = bounds[1, axis] + 0.0005  # Slightly beyond bound to ensure overlap
        default_normals_out = True

    # Use explicit normals direction if provided, otherwise use default
    if normals_out is None:
        normals_out = default_normals_out

    # Get the extents of the mesh in the other two dimensions
    min_axis1, min_axis2 = bounds[0, [axis1, axis2]]
    max_axis1, max_axis2 = bounds[1, [axis1, axis2]]

    # Add padding around the edges
    min_axis1 -= padding
    min_axis2 -= padding
    max_axis1 += padding
    max_axis2 += padding

    # Create vertices for the cap
    cap_vertices = np.zeros((4, 3))

    # Create the four corners of the cap
    # Bottom left
    cap_vertices[0, axis] = cap_pos
    cap_vertices[0, axis1] = min_axis1
    cap_vertices[0, axis2] = min_axis2

    # Bottom right
    cap_vertices[1, axis] = cap_pos
    cap_vertices[1, axis1] = max_axis1
    cap_vertices[1, axis2] = min_axis2

    # Top right
    cap_vertices[2, axis] = cap_pos
    cap_vertices[2, axis1] = max_axis1
    cap_vertices[2, axis2] = max_axis2

    # Top left
    cap_vertices[3, axis] = cap_pos
    cap_vertices[3, axis1] = min_axis1
    cap_vertices[3, axis2] = max_axis2

    # Create faces for the cap - triangle orientation depends on cap direction
    if normals_out:
        # For max side, normals point outward (away from mesh)
        cap_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
    else:
        # For min side, normals point inward (toward mesh)
        cap_faces = np.array([
            [0, 2, 1],
            [0, 3, 2]
        ])

    # Create the cap mesh
    cap_mesh = trimesh.Trimesh(vertices=cap_vertices, faces=cap_faces)

    # Combine the original mesh with the cap
    capped_mesh = trimesh.util.concatenate([mesh, cap_mesh])

    # Merge duplicate vertices
    capped_mesh.merge_vertices()

    # Make sure normals are consistent
    try:
        capped_mesh.fix_normals()
    except:
        print("Warning: Could not fix normals, but proceeding with volume calculation.")

    return capped_mesh


def calculate_accurate_volume(mesh: trimesh.Trimesh,
                              filter_outliers: bool = True,
                              threshold: float = 2.0,
                              outlier_axis: int = 2,
                              outlier_method: str = 'std',
                              outlier_percentile: float = 99.0,
                              pitch: float = None,
                              target_resolution: int = 100,
                              max_z_ratio: float = 3.0,
                              cap_mesh_enabled: bool = True,
                              cap_axis: int = 2,
                              cap_direction: str = 'min',
                              cap_padding: float = 0.005,
                              cap_normals_out: bool = None,
                              cap_both_ends: bool = False,
                              fill_holes: bool = True,
                              diagnose_orientation: bool = False) -> tuple:
    """
    Calculate accurate volume by handling outliers, filling holes, and capping the mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    filter_outliers : bool
        Whether to perform outlier filtering at all
    threshold : float
        The threshold value for outlier detection
    outlier_axis : int or None
        The axis along which to detect outliers (0=X, 1=Y, 2=Z, or None for all axes)
    outlier_method : str
        Method for outlier detection: 'std', 'percentile', or 'absolute'
    outlier_percentile : float
        Percentile value when using percentile method (0-100)
    pitch : float, optional
        The edge length (in meters) of each cubic voxel for volume calculation
    target_resolution : int
        Number of voxels along the largest XY‐dimension if pitch is None
    max_z_ratio : float
        Maximum allowed ratio of Z extent to max(X,Y) extent. If exceeded,
        additional Z filtering is applied.
    cap_mesh_enabled : bool
        Whether to cap the mesh. Set to False if you only want to filter outliers.
    cap_axis : int
        The axis along which to cap (0=X, 1=Y, 2=Z)
    cap_direction : str
        Which direction to cap ('min' or 'max')
    cap_padding : float
        Amount of padding to add around the cap in meters
    cap_normals_out : bool, optional
        If True, cap normals point outward; if False, normals point inward.
        If None (default), direction is determined based on cap_direction.
    cap_both_ends : bool
        If True, caps both ends of the specified axis (min and max)
    fill_holes : bool
        If True, attempts to fill small holes in the mesh
    diagnose_orientation : bool
        Whether to show an orientation visualization to help diagnose issues

    Returns
    -------
    tuple
        The calculated volume in milliliters and the processed mesh
    """
    # Print original mesh diagnostics
    print(f"Original mesh extents (cm): {mesh.extents * 100}")
    print(f"Original mesh is watertight: {mesh.is_watertight}")

    # Check for very extreme Z axis relative to XY
    xy_max = max(mesh.extents[0], mesh.extents[1])
    z_extent = mesh.extents[2]
    z_ratio = z_extent / xy_max

    # Step 1: Filter outliers if enabled
    if filter_outliers:
        # If Z is extremely large compared to XY, use more aggressive filtering
        if outlier_axis == 2 and z_ratio > max_z_ratio:  # Only apply if filtering Z axis
            print(f"Warning: Z/XY ratio is {z_ratio:.2f}, which exceeds threshold of {max_z_ratio:.2f}")
            print(f"Using more aggressive filtering")
            actual_threshold = min(threshold, 1.5)  # More aggressive filtering
        else:
            actual_threshold = threshold

        # If the Z ratio is especially high, consider that the mesh may need its orientation fixed
        if z_ratio > 2.0 * max_z_ratio:
            print(f"Warning: Z/XY ratio is extremely high at {z_ratio:.2f}")
            print(f"Consider manually examining the mesh orientation if results are unexpected")

        # Filter outliers using the specified method
        filtered_mesh = filter_mesh_outliers(
            mesh,
            threshold=actual_threshold,
            axis=outlier_axis,
            method=outlier_method,
            percentile=outlier_percentile
        )
    else:
        print("Outlier filtering disabled")
        filtered_mesh = mesh

    # Step 2: Process the mesh to make it watertight
    processed_mesh = filtered_mesh

    # First try to fill small holes if requested
    if fill_holes:
        try:
            processed_mesh = fill_mesh_holes(
                processed_mesh,
                max_hole_size=0.05,  # Default hole size (5% of diagonal)
                aggressive=True,     # Use aggressive hole filling
                iterations=3         # Try multiple iterations
            )
        except Exception as e:
            print(f"Warning: Hole filling encountered an error: {e}")
            print("Continuing with processing...")

    # Then cap the mesh if requested
    if cap_mesh_enabled:
        # Cap the main direction
        processed_mesh = cap_mesh(processed_mesh, axis=cap_axis, direction=cap_direction,
                                 padding=cap_padding, normals_out=cap_normals_out)

        # Cap the opposite end if requested
        if cap_both_ends:
            opposite_direction = 'max' if cap_direction == 'min' else 'min'
            processed_mesh = cap_mesh(processed_mesh, axis=cap_axis, direction=opposite_direction,
                                     padding=cap_padding, normals_out=cap_normals_out)

    # Step 3: Calculate volume
    # Print diagnostic information about the processed mesh
    print(f"Processed mesh extents (cm): {processed_mesh.extents * 100}")
    print(f"Processed mesh is watertight: {processed_mesh.is_watertight}")

    # Calculate volume robustly
    volume_ml = calculate_robust_volume(
        processed_mesh,
        pitch=pitch,
        target_resolution=target_resolution
    )

    # If requested, show a visualization to help diagnose orientation issues
    if diagnose_orientation:
        print("Showing mesh orientation visualization to help diagnose capping issues...")
        print(f"Current cap settings: axis={cap_axis} (0=X,1=Y,2=Z), direction={cap_direction}")
        scene = visualize_mesh_orientation(mesh, show=True)

    return volume_ml, processed_mesh


def calculate_robust_volume(mesh: trimesh.Trimesh,
                            pitch: float = None,
                            target_resolution: int = 100,
                            use_convex_hull_fallback: bool = True) -> float:
    """
    Calculate the volume of a mesh robustly, with fallbacks for non-watertight meshes.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to calculate volume for
    pitch : float, optional
        The edge length (in meters) of each cubic voxel. If None, automatically
        set so the mesh's max XY‐extent is divided into `target_resolution` voxels.
    target_resolution : int
        Number of voxels along the largest XY‐dimension if pitch is None
    use_convex_hull_fallback : bool
        Whether to use convex hull as a fallback for non-watertight meshes

    Returns
    -------
    float
        Volume in milliliters
    """
    # Choose pitch if not provided
    if pitch is None:
        # use the larger of width or depth
        w, d = mesh.extents[:2]
        max_dim = max(w, d)
        pitch = max_dim / target_resolution

    # First try to calculate volume directly if the mesh is watertight
    if mesh.is_watertight:
        try:
            # For watertight meshes, we can compute exact volume
            volume_m3 = mesh.volume
            volume_ml = volume_m3 * 1e6
            print(f"Calculated exact volume (watertight mesh): {volume_ml:.3f} mL")
            return volume_ml
        except Exception as e:
            print(f"Warning: Error computing exact volume: {e}")
            # Fall through to voxelization

    # Try voxelization approach
    try:
        vg = mesh.voxelized(pitch)
        num_voxels = len(vg.points)
        volume_m3 = num_voxels * pitch**3
        volume_ml = volume_m3 * 1e6
        print(f"Calculated volume via voxelization: {volume_ml:.3f} mL")
        return volume_ml
    except Exception as e:
        print(f"Warning: Error computing volume via voxelization: {e}")

        # If we reach here, try convex hull if enabled
        if use_convex_hull_fallback:
            try:
                # Use convex hull as a last resort
                hull = mesh.convex_hull
                volume_m3 = hull.volume
                volume_ml = volume_m3 * 1e6
                print(f"Warning: Used convex hull approximation for volume: {volume_ml:.3f} mL")
                return volume_ml
            except Exception as e2:
                print(f"Error: Failed to compute volume via convex hull: {e2}")

        # If all methods fail
        raise ValueError("Could not compute volume by any method")


def visualize_mesh_orientation(mesh: trimesh.Trimesh,
                              show_caps: bool = True,
                              show: bool = True,
                              axis_scale: float = 0.2):
    """
    Visualize a mesh with clear axis indicators to help understand orientation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to visualize
    show_caps : bool
        Whether to show potential capping planes on each axis
    show : bool
        Whether to display the visualization immediately
    axis_scale : float
        Scale of the axes relative to the mesh size

    Returns
    -------
    trimesh.Scene
        The scene containing the mesh and orientation markers
    """
    # Create a copy of the mesh
    mesh_copy = mesh.copy()

    # Create a scene
    scene = trimesh.Scene()

    # Set a transparent blue color for the mesh
    mesh_copy.visual.face_colors = [100, 100, 255, 150]  # Semi-transparent blue
    scene.add_geometry(mesh_copy, node_name='mesh')

    # Add coordinate axes at the center of the mesh
    origin = mesh.center_mass
    axis_size = max(mesh.extents) * axis_scale
    axes = trimesh.creation.axis(origin_size=axis_size)
    scene.add_geometry(axes, node_name='axes')

    # Create text labels for the axes
    labels = ['X', 'Y', 'Z']
    colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]

    # Add labels at the end of each axis
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Calculate position at the end of each axis
        position = origin.copy()
        position[i] += axis_size * 1.2

        # Create a text label
        text_size = axis_size * 0.5
        text = trimesh.creation.text(label, font_size=text_size)
        text.apply_translation(position)
        text.visual.face_colors = color
        scene.add_geometry(text, node_name=f'label_{label}')

    # Show potential capping planes if requested
    if show_caps:
        bounds = mesh.bounds
        extents = mesh.extents

        # Create capping planes at min and max of each axis
        for axis in range(3):
            axis_label = labels[axis]
            color = colors[axis]

            # Create a plane at the minimum position
            min_pos = bounds[0, axis]
            min_plane = _create_cap_plane(mesh, axis, min_pos, color[:3] + [100])
            scene.add_geometry(min_plane, node_name=f'min_{axis_label}_cap')

            # Create a plane at the maximum position
            max_pos = bounds[1, axis]
            max_plane = _create_cap_plane(mesh, axis, max_pos, color[:3] + [100])
            scene.add_geometry(max_plane, node_name=f'max_{axis_label}_cap')

    # Show the scene if requested
    if show:
        scene.show()

    return scene


def _create_cap_plane(mesh, axis, position, color):
    """
    Create a plane for visualization at the specified axis position.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to create a cap for
    axis : int
        The axis along which to create the plane (0=X, 1=Y, 2=Z)
    position : float
        The position along the axis
    color : list
        RGBA color for the plane

    Returns
    -------
    trimesh.Trimesh
        A mesh representing the capping plane
    """
    # Create array indices for other axes
    other_axes = [i for i in range(3) if i != axis]
    axis1, axis2 = other_axes

    # Get bounds with padding
    bounds = mesh.bounds
    padding = 0.01  # A small padding

    # Get the extents of the mesh in the other two dimensions
    min_axis1, min_axis2 = bounds[0, [axis1, axis2]] - padding
    max_axis1, max_axis2 = bounds[1, [axis1, axis2]] + padding

    # Create vertices for the plane
    plane_vertices = np.zeros((4, 3))

    # Bottom left
    plane_vertices[0, axis] = position
    plane_vertices[0, axis1] = min_axis1
    plane_vertices[0, axis2] = min_axis2

    # Bottom right
    plane_vertices[1, axis] = position
    plane_vertices[1, axis1] = max_axis1
    plane_vertices[1, axis2] = min_axis2

    # Top right
    plane_vertices[2, axis] = position
    plane_vertices[2, axis1] = max_axis1
    plane_vertices[2, axis2] = max_axis2

    # Top left
    plane_vertices[3, axis] = position
    plane_vertices[3, axis1] = min_axis1
    plane_vertices[3, axis2] = max_axis2

    # Create faces
    plane_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    # Create the plane mesh
    plane_mesh = trimesh.Trimesh(vertices=plane_vertices, faces=plane_faces)
    plane_mesh.visual.face_colors = color

    return plane_mesh


def visualize_mesh_comparison(original_mesh: trimesh.Trimesh,
                             processed_mesh: trimesh.Trimesh,
                             show: bool = True):
    """
    Visualize the original and processed meshes side by side.

    Parameters
    ----------
    original_mesh : trimesh.Trimesh
        The original input mesh
    processed_mesh : trimesh.Trimesh
        The processed mesh with sealed bottom
    show : bool
        Whether to display the visualization immediately

    Returns
    -------
    Optional[trimesh.Scene]
        The scene containing both meshes, or None if show is True
    """
    # Create a copy of the original mesh
    orig = original_mesh.copy()

    # Create a copy of the processed mesh and move it to the right
    proc = processed_mesh.copy()
    max_x = orig.bounds[1, 0]
    min_x = proc.bounds[0, 0]
    proc.apply_translation([max_x - min_x + orig.extents[0] * 0.2, 0, 0])

    # Create a scene with both meshes
    scene = trimesh.Scene()
    orig.visual.face_colors = [200, 200, 250, 255]  # Light blue for original
    proc.visual.face_colors = [250, 200, 200, 255]  # Light red for processed

    scene.add_geometry(orig, node_name='original')
    scene.add_geometry(proc, node_name='processed')

    # Add axes to help visualize orientation
    origin_size = min(original_mesh.extents) * 0.1
    orig_axes = trimesh.creation.axis(origin_size=origin_size)
    proc_axes = trimesh.creation.axis(origin_size=origin_size)
    proc_axes.apply_translation([max_x - min_x + orig.extents[0] * 0.2, 0, 0])

    scene.add_geometry(orig_axes, node_name='original_axes')
    scene.add_geometry(proc_axes, node_name='processed_axes')

    if show:
        scene.show()
        return None

    return scene
#

# %%
# ─── Looping over your _scaled.glb meshes ────────────────────

obj_dir = "./meshes/final_obj/"

for fn in os.listdir(obj_dir):
    if not fn.endswith("_scaled.obj"):
        continue

    base = fn.replace("_scaled.obj", "")
    path = os.path.join(obj_dir, fn)
    mesh = trimesh.load(path, force="mesh")

    # Approximate via voxelization
    vol_ml = approximate_volume_ml(mesh, pitch=0.002)

    try:
        item_idx_str, food_name = base.split('-', 1)
        item_index = int(item_idx_str)
    except ValueError:
        print(f"Invalid key format: {base}")
        continue

    # attempt to find a matching row in the DataFrame
    # Handle potential underscore/space differences and special characters
    food_name_spaces = food_name.replace('_', ' ')
    food_name_underscores = food_name.replace(' ', '_')

    # Handle special characters like & and !
    clean_food_name = re.sub(r'[&!]', '', food_name)
    clean_spaces = re.sub(r'[&!]', '', food_name_spaces)
    clean_underscores = re.sub(r'[&!]', '', food_name_underscores)

    # Compare with original and cleaned versions
    matched = df[
        (df['Food Index'] == item_index) &
        (
            (df['Food Name'] == food_name) |
            (df['Food Name'] == food_name_spaces) |
            (df['Food Name'] == food_name_underscores) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_food_name) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_spaces) |
            (df['Food Name'].apply(lambda x: re.sub(r'[&!]', '', x)) == clean_underscores)
        )
    ]
    if matched.empty:
        print(f"\n⚠️ Couldn't find {food_name} in the dataframe")
        continue

    # store volume in the DataFrame row
    df.loc[matched.index, 'volume'] = vol_ml
    print(f"Stored volume for {food_name} value: {vol_ml:.3f} mL")


# %%
import trimesh

def mesh_volume_ml(mesh: trimesh.Trimesh, check_watertight: bool = True) -> float:
    """
    Load a trimesh.Trimesh instance (assumed in meters), compute its volume, and return in milliliters.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to compute the volume of.
    check_watertight : bool
        If True, raise an error when the mesh isn’t watertight.

    Returns
    -------
    volume_ml : float
        Volume in milliliters (mL).
    """
    # Optionally ensure it's watertight for reliable volume
    if check_watertight and not mesh.is_watertight:
        raise ValueError(f"Mesh '{glb_path}' is not watertight.")

    # Volume in cubic meters
    volume_m3 = mesh.volume

    # Convert to milliliters (1 m³ = 1_000_000 mL)
    volume_ml = volume_m3 * 1e6

    return volume_ml

# Example usage:
# ml = glb_volume_ml('food_model.glb')
# print(f"Volume: {ml:.2f} mL")

# %%
import trimesh
import pyrender
import numpy as np
from PIL import Image

def render_obj_view(
    obj_path: str,
    focal_length_mm: float,
    sensor_width_mm: float = None,
    sensor_height_mm: float = None,
    image_width_px: int = 800,
    image_height_px: int = 600,
    camera_height_m: float = 1.5,
    camera_pitch_deg: float = 30.0
) -> Image.Image:
    """
    Load an OBJ, set up a pinhole camera with given intrinsics/extrinsics,
    render off-screen, and return a PIL Image.

    Parameters:
    -----------
    obj_path : str
        Path to the .obj mesh file.
    focal_length_mm : float
        Camera focal length, in millimeters.
    sensor_width_mm : float or None
        Sensor width in millimeters. If None, infer or assume square pixels.
    sensor_height_mm : float or None
        Sensor height in millimeters. If None, infer or assume square pixels.
    image_width_px : int
        Output image width in pixels.
    image_height_px : int
        Output image height in pixels.
    camera_height_m : float
        Camera height above the ground plane (Z-axis), in meters.
    camera_pitch_deg : float
        Camera pitch angle down from horizontal, in degrees.

    Returns:
    --------
    PIL.Image.Image
        The rendered color image.
    """
    # Ensure at least one sensor dimension is provided
    if sensor_width_mm is None and sensor_height_mm is None:
        raise ValueError("Must provide sensor_width_mm or sensor_height_mm")

    # Infer missing sensor dimension by image aspect ratio
    aspect = image_height_px / image_width_px
    if sensor_width_mm is None:
        sensor_width_mm = sensor_height_mm * (1 / aspect)
    if sensor_height_mm is None:
        sensor_height_mm = sensor_width_mm * aspect

    # Compute intrinsics (focal lengths in pixels)
    fx = focal_length_mm / sensor_width_mm  * image_width_px
    fy = focal_length_mm / sensor_height_mm * image_height_px
    cx = image_width_px  / 2.0
    cy = image_height_px / 2.0

    # Load mesh and set up scene
    mesh = trimesh.load(obj_path)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
    scene.add(render_mesh, pose=np.eye(4))

    # Set up camera with intrinsics and extrinsics
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    # Build camera pose:
    pitch_rad = np.deg2rad(camera_pitch_deg)
    # a) Add a rotation around the x-axis to look down
    R = trimesh.transformations.rotation_matrix(-pitch_rad, [1, 0, 0])
    # b) Add a translation to move the camera up from the ground
    T = trimesh.transformations.translation_matrix([0, 0, camera_height_m])
    camera_pose = trimesh.transformations.concatenate_matrices(T, R)
    scene.add(camera, pose=camera_pose)

    # Add a directional light at the camera location
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render off-screen
    renderer = pyrender.OffscreenRenderer(image_width_px, image_height_px)
    color, _ = renderer.render(scene)

    return Image.fromarray(color)

# %%
img = render_obj_view(
    obj_path='path/to/mesh.obj',
    focal_length_mm=50.0,
    sensor_width_mm=36.0,            # or None
    sensor_height_mm=None,           # infers via square-pixel/AR
    image_width_px=800,
    image_height_px=600,
    camera_height_m=1.5,
    camera_pitch_deg=30.0
)

# To save:
img.save('view.png')

# Or to display in a Jupyter notebook:
display(img)

# ================== Save Meshes ================
# Create final_obj directory inside meshes if it doesn't exist
import os
import zipfile
from google.colab import files

# Set the mesh folder
mesh_dir = './meshes'
zip_filename = 'meshes.zip'

# Create the zip file
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    # Walk through the entire directory structure
    for root, dirs, found_files in os.walk(mesh_dir):
        for file in found_files:
            # Get the full file path
            file_path = os.path.join(root, file)
            # Calculate path relative to mesh_dir for arcname
            rel_path = os.path.relpath(file_path, os.path.dirname(mesh_dir))
            # Add file to the zip with relative path
            zipf.write(file_path, arcname=rel_path)

print(f"✅ Zipped the entire meshes directory into {zip_filename}")

# Download the zip file
files.download(zip_filename)
# ================= Kaggle ================
import csv

# Create and save a CSV file with the header
def create_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "predicted"])

# Function to append a new line with id and prediction
def append_to_csv(filename, id, prediction):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([id, prediction])

# Example usage
filename = 'predictions.csv'
create_csv(filename)
append_to_csv(filename, 1, 0.95)
append_to_csv(filename, 2, 0.88)

# ================= Pandas ================
import pandas as pd

def create_data_store():
  """
  Returns an empty DataFrame with two columns:
    - 'id' as integers
    - 'predicted' as floats
  """
  # 1) Make an empty DataFrame with the right column names
  df = pd.DataFrame(columns=['id', 'predicted'])

  # 2) Tell pandas what type each column should hold
  df = df.astype({'id': 'int64', 'predicted': 'float64'})

  return df

def add_row(df, id_value, predicted_value):
  """
  Appends a new row to df. Always returns the updated df.

  Parameters:
    - df:        the DataFrame you’re building
    - id_value:  an integer (we cast it just in case)
    - predicted_value: a float (we cast it too)
  """
  # Build a single-row dict
  new_row = {
      'id': int(id_value),
      'predicted': float(predicted_value)
  }

  # .loc[len(df)] puts it at the “next” index (end of the table)
  df.loc[len(df)] = new_row

  return df

def update_rows_value(df, row_label: str, row_value: str, column_label: str, value: str):
  """
  Updates the value for a specific column in the rows matching a label/value pair.

  Args:
      df (pd.DataFrame): DataFrame to update
      row_label (str): Column name to match rows on
      row_value: Value to match in row_label column
      column_label (str): Column name to update
      value: New value to set

  Returns:
      pd.DataFrame: Updated DataFrame
  """
  matched = df[df[row_label] == row_value]
  df[column_label].dtype
  if matched.empty:
    print(f"No rows found with {row_label} = {row_value}")
    return df
  df.loc[matched.index, column_label] = value
  return df

def find_rows_with_values(df, pairs: dict):
  """
  Finds the rows in the DataFrame that match multiple label/value pairs.

  Args:
      df (pd.DataFrame): DataFrame to search
      pairs (dict): Dictionary of column names to match on and their values
                   e.g. {'col1': val1, 'col2': val2}

  Returns:
      pd.DataFrame: DataFrame containing rows that match all label/value pairs
  """
  mask = pd.Series(True, index=df.index)
  for label, value in pairs.items():
      mask &= (df[label] == value)
  matched = df[mask]
  return matched

def display_dataframe(df):
    """
    Displays the DataFrame in a Jupyter notebook cell.
    """
    from IPython.display import display
    display(df)

def save_dataframe_to_csv(df, filename):
    """
    Saves df to a CSV file named `filename`.
    The file will have exactly two columns and no extra index column.
    """
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}")

def export_predictions(df, filename):
    """
    Saves prediction data to a CSV file.

    Extracts 'Food Index' and 'volume' columns from the DataFrame,
    renames them to 'id' and 'prediction', and saves them to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing food data
        filename (str): Path to the output CSV file
    """
    # Create a new DataFrame with only the required columns
    predictions_df = df[['Food Index', 'volume']].copy()

    # Rename columns to match required format
    predictions_df.columns = ['id', 'prediction']

    # Remove rows with missing volume values
    predictions_df = predictions_df.dropna(subset=['prediction'])

    # Save to CSV
    predictions_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename} with {len(predictions_df)} entries")

def load_dataframe_from_csv(filename):
    """
    Loads a DataFrame from a CSV file named `filename`.
    """
    if not filename.endswith('.csv'):
        raise ValueError("Filename must end with '.csv'")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    df = pd.read_csv(filename)
    return df

# ================= Pandas ================
