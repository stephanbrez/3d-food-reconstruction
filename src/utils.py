import csv
import os
import pandas as pd
from IPython.display import display
from io import StringIO
from typing import Dict, List, Tuple, Optional, Union, Literal, Any


# ═════ Debug Utilities ═════

def debug_print(message: str, debug_info: bool) -> None:
    """
    Print debug message if debug_info is True.

    Parameters
    ----------
    message : str
        Debug message to print
    debug_info : bool
        Whether to actually print the message
    """
    if debug_info:
        print(f"[DEBUG] {message}")


# ═════ Data Setup ═════

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

def get_combo_id(filename: str) -> Optional[int]:
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

def get_item_names_from_combo(item_ids, combo_id):
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

# ═════ Pandas ═════

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
