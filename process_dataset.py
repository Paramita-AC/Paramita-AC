import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile as tiff

def load_npy_mask(npy_path):
    """Load mask from .npy file using cellpose format and apply thresholding."""
    data = np.load(npy_path, allow_pickle=True).item()
    mask = data['masks']
    # Apply thresholding to convert to binary mask
    _, mask = cv.threshold(mask.astype(np.uint8), 0, 255, cv.THRESH_BINARY)
    return mask

def print_group_info(group_df, group_idx):
    """Print detailed information about a group."""
    print("\n" + "="*80)
    print(f"Group {group_idx} Information:")
    print(f"Cell Type: {group_df['cell_type'].iloc[0]}")
    print(f"Passage: {group_df['passage'].iloc[0]}")
    print(f"Seeding Date: {group_df['seeding_date'].iloc[0]}")
    print(f"Well: {group_df['well'].iloc[0]}")
    print(f"Plate: {group_df['plate'].iloc[0]}")
    print(f"Image Index: {group_df['well_image_index'].iloc[0]}")
    print(f"Folder Date: {group_df['folder_date'].iloc[0]}")
    print(f"Folder Time: {group_df['folder_time'].iloc[0]}")
    print(f"Folder Plate: {group_df['folder_plate'].iloc[0]}")
    
    print("\nChannels:")
    for _, row in group_df[~group_df['mask']].sort_values('channel_id').iterrows():
        print(f"  Channel {row['channel_id']}: {row['channel_name']} ({row['full_path']})")
    
    mask_df = group_df[group_df['mask']]
    if not mask_df.empty:
        print("\nMask:")
        print(f"  {mask_df.iloc[0]['full_path']}")
    print("="*80)

def display_all_groups(mask_groups, output_dir=None):
    """
    Display all groups in a single figure with multiple rows.
    
    Args:
        mask_groups: DataFrame containing all groups
        output_dir: Optional directory to save the displayed images
    """
    # Get the number of groups
    n_groups = len(mask_groups.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate']))
    
    # Get the maximum number of channels (excluding mask) across all groups
    max_channels = max(len(group[~group['mask']]) for _, group in mask_groups.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate']))
    
    # Calculate grid dimensions
    n_cols = max_channels + 1  # +1 for mask
    n_rows = n_groups
    
    # Create figure with subplots
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
    
    # Process each group
    for group_idx, (group_key, group_df) in enumerate(mask_groups.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate'])):
        # Print group information
        print_group_info(group_df, group_idx)
        
        # Sort by channel_id to ensure consistent order
        group_df = group_df.sort_values('channel_id')
        
        # Display regular images
        for idx, (_, row) in enumerate(group_df[~group_df['mask']].iterrows()):
            img_path = row['full_path']
            if img_path.endswith('.tif'):
                img = tiff.imread(img_path)
            else:
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            
            ax = plt.subplot(n_rows, n_cols, group_idx*n_cols + idx + 1)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Group {group_idx}\nChannel {row['channel_id']}")
            ax.axis('off')
        
        # Display mask if exists
        mask_df = group_df[group_df['mask']]
        if not mask_df.empty:
            mask_path = mask_df.iloc[0]['full_path']
            if mask_path.endswith('.npy'):
                mask = load_npy_mask(mask_path)
            else:
                mask = tiff.imread(mask_path)
                # Apply thresholding to tif masks as well for consistency
                _, mask = cv.threshold(mask.astype(np.uint8), 0, 255, cv.THRESH_BINARY)
            
            ax = plt.subplot(n_rows, n_cols, group_idx*n_cols + n_cols)
            ax.imshow(mask, cmap='gray')
            ax.set_title(f"Group {group_idx}\nMask")
            ax.axis('off')
    
    # Add overall title
    fig.suptitle("All Groups with Channels and Masks", fontsize=16, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or show
    if output_dir:
        output_path = Path(output_dir) / "all_groups.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.show()

def process_dataset(csv_path: str, output_path: str = None, display_groups: bool = False, output_dir: str = None) -> pd.DataFrame:
    """
    Process the dataset by grouping and filtering based on specified criteria.
    
    Args:
        csv_path: Path to the input CSV file (dfb_dataset.csv)
        output_path: Optional path to save the processed dataset
        display_groups: Whether to display images for each group
        output_dir: Directory to save displayed images if display_groups is True
    
    Returns:
        pd.DataFrame: Processed dataset containing only groups with at least one mask
    """
    # Read the dataset
    df = pd.read_csv(csv_path)
    
    # Print initial dataset information
    print("\nInitial Dataset Information:")
    print(f"Total rows: {len(df)}")
    print(f"Number of mask files: {df['mask'].sum()}")
    print("\nUnique values in key columns:")
    for col in ['cell_type', 'passage', 'well', 'plate']:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Group by specified columns
    grouped = df.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate'])
    
    # Print information about all groups before filtering
    print("\nGroups before filtering:")
    print(f"Total number of groups: {len(grouped)}")
    
    # Filter groups that have at least one mask
    mask_groups = grouped.filter(lambda x: x['mask'].any())
    
    # Print information about filtered groups
    print("\nGroups after filtering (with masks):")
    print(f"Number of groups with masks: {len(mask_groups.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate']))}")
    
    # Sort the filtered dataset
    mask_groups = mask_groups.sort_values(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate'])
    
    # Save to CSV if output path is provided
    if output_path:
        mask_groups.to_csv(output_path, index=False)
        print(f"Processed dataset saved to: {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Original dataset size: {len(df)}")
    print(f"Processed dataset size: {len(mask_groups)}")
    
    # Count unique values for each grouping column
    print("\nUnique values in each grouping column:")
    for col in ['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate']:
        unique_count = mask_groups[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Count total number of unique groups
    unique_groups = len(mask_groups.groupby(['cell_type', 'passage', 'seeding_date', 'well', 'plate', 'well_image_index', 'folder_date', 'folder_time', 'folder_plate']))
    print(f"\nTotal number of unique groups: {unique_groups}")
    
    # Print group statistics
    print("\nGroups by cell type:")
    print(mask_groups.groupby('cell_type').size())
    
    print("\nGroups by passage:")
    print(mask_groups.groupby('passage').size())
    
    # Display images for each group if requested
    if display_groups:
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nProcessing groups for display...")
        try:
            display_all_groups(mask_groups, output_dir)
        except Exception as e:
            print(f"Error displaying groups: {str(e)}")
    
    return mask_groups

if __name__ == "__main__":
    # Example usage
    csv_path = "datasets/dfb_dataset.csv"
    output_path = "datasets/dfb_dataset_processed.csv"
    output_dir = "datasets/group_images"
    
    processed_df = process_dataset(csv_path, output_path, display_groups=True, output_dir=output_dir)
    
    # Print example of first few rows
    print("\nFirst few rows of processed dataset:")
    print(processed_df.head())
