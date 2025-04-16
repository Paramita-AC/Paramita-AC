import os
from pathlib import Path
from typing import NamedTuple, List, Dict, Union
import pandas as pd
import json
from PIL import Image
import numpy as np

class ImageInfo(NamedTuple):
    well: str
    plate: str
    channel_id: str
    well_image_index: str
    channel_name: str
    read_index: str
    filename: str
    full_path: str
    mask: bool  # True for .npy files, False for .tif files

class FolderMetadata(NamedTuple):
    folder_path: str

def parse_image_filename(filename: str, full_path: str) -> ImageInfo:
    """Parse microscope image filename into structured data."""
    # Split filename into components
    name_parts = filename.split('_')
    
    if len(name_parts) < 6:
        raise ValueError(f"Invalid filename format: {filename}")
    
    # Extract components
    well = name_parts[0]
    plate = name_parts[1]
    channel_id = name_parts[2]
    well_image_index = name_parts[3]
    channel_name = name_parts[4]  # Handle channel names with spaces
    read_index = name_parts[5].split('.')[0]  # Remove file extension
    
    # Determine if file is a mask (.npy) or not
    is_mask = filename.endswith('.npy')
    
    return ImageInfo(
        well=well,
        plate=plate,
        channel_id=channel_id,
        well_image_index=well_image_index,
        channel_name=channel_name,
        read_index=read_index,
        filename=filename,
        full_path=full_path,
        mask=is_mask
    )

def load_folder_metadata(metadata_path: str) -> Dict:
    """Load metadata from a file (supports JSON, can be extended for other formats)."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error parsing metadata file: {metadata_path}")
        return {}
    except FileNotFoundError:
        print(f"Metadata file not found: {metadata_path}")
        return {}

def parse_folder_name(folder_path: str) -> Dict[str, str]:
    """
    Parse folder name to extract date, time, and plate information.
    Example: "250312_121525_Plate 1" -> 
        {
            'folder_date': '2025-03-12',
            'folder_time': '12:15:25',
            'folder_plate': '1'
        }
    """
    try:
        # Get the folder name from the path
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid folder name format: {folder_name}")
        
        # Parse date (assuming format YYMMDD)
        date_str = parts[0]
        year = f"20{date_str[:2]}"  # Assuming 20XX years
        month = date_str[2:4]
        day = date_str[4:6]
        formatted_date = f"{year}-{month}-{day}"
        
        # Parse time (HHMMSS)
        time_str = parts[1]
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:6]
        formatted_time = f"{hour}:{minute}:{second}"
        
        # Parse plate number
        plate_str = parts[2]  # "Plate X"
        plate_num = plate_str.split()[-1]  # Get the number after "Plate"
        
        return {
            'folder_date': formatted_date,
            'folder_time': formatted_time,
            'folder_plate': plate_num
        }
    except Exception as e:
        print(f"Error parsing folder name {folder_path}: {str(e)}")
        return {
            'folder_date': '',
            'folder_time': '',
            'folder_plate': ''
        }

def build_dataset(folder_paths: List[str]) -> pd.DataFrame:
    """
    Build dataset from microscope images in multiple folders with their metadata.
    
    Args:
        folder_paths: List of folder paths containing image data and metadata.json
    
    Returns:
        pandas.DataFrame: Dataset containing image information and folder metadata
    """
    all_image_data = []
    
    for folder_path in folder_paths:
        # Load metadata for this folder
        metadata_path = os.path.join(folder_path, "metadata.json")
        metadata = load_folder_metadata(metadata_path)
        
        # Parse folder name and add to metadata
        folder_info = parse_folder_name(folder_path)
        metadata.update(folder_info)
        
        # Walk through all subdirectories in this folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.tif', '.npy')):
                    full_path = os.path.join(root, file)
                    try:
                        # Get basic image info
                        image_info = parse_image_filename(file, full_path)
                        
                        # Convert ImageInfo to dictionary
                        image_dict = image_info._asdict()
                        
                        # Add metadata to image info
                        image_dict.update(metadata)
                        
                        all_image_data.append(image_dict)
                    except ValueError as e:
                        print(f"Skipping invalid file: {file}, Error: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_image_data)
    return df

def reconstruct_image_path(well: str, plate: str, channel_id: str, 
                         well_image_index: str, channel_name: str, 
                         read_index: str, root_folder: str, file_type: str = 'tif') -> str:
    """Reconstruct full image path from components."""
    if file_type == 'npy':
        filename = f"{well}_{plate}_{channel_id}_{well_image_index}_{channel_name}_{read_index:03d}_seg.npy"
    else:
        filename = f"{well}_{plate}_{channel_id}_{well_image_index}_{channel_name}_{read_index:03d}.tif"
    
    # Search for the file in root_folder and its subdirectories
    for root, _, files in os.walk(root_folder):
        if filename in files:
            return os.path.join(root, filename)
    
    raise FileNotFoundError(f"Image not found: {filename}")

def load_image_by_id(dataset: pd.DataFrame, idx: int) -> np.ndarray:
    """
    Load image using its index from the dataset.
    
    Args:
        dataset: pandas DataFrame containing image information
        idx: index of the image in the dataset
    
    Returns:
        numpy.ndarray: Loaded image as a numpy array
    
    Raises:
        IndexError: If idx is out of bounds
        FileNotFoundError: If image file doesn't exist
    """
    if idx < 0 or idx >= len(dataset):
        raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(dataset)}")
    
    # Get image info from dataset
    row = dataset.iloc[idx]
    
    # Load image based on file extension
    try:
        if row.full_path.endswith('.npy'):
            return np.load(row.full_path)
        else:  # .tif files
            img = Image.open(row.full_path)
            return np.array(img)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load image at {row.full_path}. Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Example of folder paths
    folder_paths = [
        "/home/shared/Cytation C10/250305_171844_dfb_12_march5/250305_174946_Plate 1",
        "/home/shared/Cytation C10/250306_170210_dfb_12_march6/250306_170210_Plate 1",
        "/home/shared/Cytation C10/250310_105206_dfb_12_march10/250310_105206_Plate 1",
        "/home/shared/Cytation C10/250311_191049_dfb_12_march11/250311_191049_Plate 1",
        "/home/shared/Cytation C10/250312_121525_dfb_12_march12/250312_121525_Plate 1"
        # Add more folder paths as needed
    ]
    
    # Build dataset with metadata
    dataset = build_dataset(folder_paths)
    
    # Save to CSV (optional)
    dataset.to_csv("datasets/dfb_dataset.csv", index=False)
    print(f"Dataset shape: {dataset.shape}")
    # Example of loading image by dataset index
    try:
        image = load_image_by_id(dataset, 0)
        print(f"Successfully loaded image with shape: {image.shape}")
        
        # Print metadata for this image
        metadata_cols = [col for col in dataset.columns 
                        if col not in ImageInfo._fields]
        print("\nMetadata for this image:")
        for col in metadata_cols:
            print(f"{col}: {dataset.iloc[0][col]}")
            
    except (IndexError, FileNotFoundError) as e:
        print(e)
