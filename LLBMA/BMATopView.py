####################################################################################################
# Imports ##########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openslide
from pathlib import Path
from PIL import Image, ImageDraw

# Within package imports ###########################################################################
from LLBMA.vision.masking import get_white_mask, get_obstructor_mask, get_top_view_mask
from LLBMA.resources.BMAassumptions import *
from LLBMA.vision.processing import read_with_timeout
from LLBMA.vision.bma_particle_detection import (
    get_top_view_preselection_mask,
    get_grid_rep,
)


def extract_top_view(wsi_path, save_dir=None):
    # you can get the stem by removing the last 5 characters from the file name (".ndpi")
    stem = Path(wsi_path).stem[:-5]

    print("Extracting top view")
    # open the wsi in tmp_dir and extract the top view
    wsi = openslide.OpenSlide(wsi_path)
    toplevel = wsi.level_count - 1
    topview = read_with_timeout(
        wsi=wsi,
        location=(0, 0),
        level=toplevel,
        dimensions=wsi.level_dimensions[toplevel],
    )

    # make sure to convert topview tp a PIL image in RGB mode
    if topview.mode != "RGB":
        topview = topview.convert("RGB")

    if save_dir is not None:
        topview.save(os.path.join(save_dir, stem + ".jpg"))
    wsi.close()

    return topview


class TopView:
    """A TopView class object representing all the information needed at the top view of the WSI.

    === Class Attributes ===
    - image : the image of the top view
    - mask : the mask of the top view
    - blue_mask : the blue mask of the top view
    - overlayed_image : the image of the top view with the mask overlayed
    - grid_rep : the grid representation of the top view
    - width : the width of the top view
    - height : the height of the top view
    - downsampling_rate : the downsampling rate of the top view
    - level : the level of the top view in the WSI

    - is_bma : whether the top view is a bone marrow aspirate top view
    - verbose : whether to print out the progress of the top view

    - binary_mask_np : the binary mask of the top view as a numpy array
    """

    def __init__(self, image, downsampling_rate, level, verbose=False, is_bma=True):
        """Initialize a TopView object.
        Image is a PIL image. Check the type of image. If not PIL image, raise ValueError.
        """
        self.verbose = verbose
        self.is_bma = is_bma
        self.downsampling_rate = downsampling_rate
        self.level = level

        self.binary_mask_np = None

        if self.verbose:
            print("Checking the type of image...")
        # check the type of image
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL image.")

        self.image = image

        if self.verbose:
            print("Printing various masks of the top view...")

        # make sure image is converted to cv2 format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        mask, overlayed_image, final_blue_mask = get_top_view_preselection_mask(
            image, verbose=False
        )

        # if the mask is all black then change the mask to all white
        if np.all(mask == 0):
            mask = 255 * np.ones_like(mask)
            print(
                "User Warning: The mask is all black. Changing the mask to all white."
            )

        # now make sure mask, overlayed_image and final_blue_mask are converted to PIL images after converting to RGB
        mask_pil = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        overlayed_image_pil = Image.fromarray(
            cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
        )
        final_blue_mask_pil = Image.fromarray(
            cv2.cvtColor(final_blue_mask, cv2.COLOR_BGR2RGB)
        )

        self.mask = mask_pil
        self.overlayed_image = overlayed_image_pil
        self.blue_mask = final_blue_mask_pil

        grid_rep = get_grid_rep(
            image=image,
            mask=mask,
            final_blue_mask=final_blue_mask,
            overlayed_image=overlayed_image,
        )

        # make sure grid_rep is converted to PIL image
        grid_rep_pil = Image.fromarray(cv2.cvtColor(grid_rep, cv2.COLOR_BGR2RGB))

        self.grid_rep = grid_rep_pil

    def is_peripheral_blood(self):
        """Return True iff the top view is a peripheral blood top view."""
        return True

    def is_in_mask(self, coordinate):
        """Return True iff the coordinate is within the mask area.
        The input coordinate needs to be the level 0 coordinate.
        """

        if self.binary_mask_np is None:
            # make sure to get a cv2 format of the mask as a binary numpy array
            mask_np = cv2.cvtColor(np.array(self.mask), cv2.COLOR_RGB2GRAY)

            # make sure to convert mask_np to a binary mask
            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

            self.binary_mask_np = mask_np

        # Adjust coordinates by downsampling factor
        TL_x_adj, TL_y_adj, BR_x_adj, BR_y_adj = [
            int(coord / topview_downsampling_factor) for coord in coordinate
        ]

        # Check if the box is within the mask area
        # Ensuring the coordinates are within the mask dimensions
        TL_x_adj, TL_y_adj = max(0, TL_x_adj), max(0, TL_y_adj)
        BR_x_adj, BR_y_adj = min(self.binary_mask_np.shape[1], BR_x_adj), min(
            self.binary_mask_np.shape[0], BR_y_adj
        )

        if np.any(self.binary_mask_np[TL_y_adj:BR_y_adj, TL_x_adj:BR_x_adj]):
            return True

        return False

    def filter_coordinates_with_mask(self, coordinates):
        """Filters out coordinates not in the binary mask area.

        Args:
            coordinates (list of tuples): List of (TL_x, TL_y, BR_x, BR_y) boxes.

        Returns:
            list of tuples: Filtered list of coordinates.
        """

        if self.binary_mask_np is None:
            # make sure to get a cv2 format of the mask as a binary numpy array
            mask_np = cv2.cvtColor(np.array(self.mask), cv2.COLOR_RGB2GRAY)

            # make sure to convert mask_np to a binary mask
            _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

            self.binary_mask_np = mask_np

        filtered_coordinates = []

        for box in coordinates:
            # Adjust coordinates by downsampling factor
            TL_x_adj, TL_y_adj, BR_x_adj, BR_y_adj = [
                int(
                    coord / (topview_downsampling_factor // search_view_downsample_rate)
                )
                for coord in box
            ]

            # Check if the box is within the mask area
            # Ensuring the coordinates are within the mask dimensions
            TL_x_adj, TL_y_adj = max(0, TL_x_adj), max(0, TL_y_adj)
            BR_x_adj, BR_y_adj = min(self.binary_mask_np.shape[1], BR_x_adj), min(
                self.binary_mask_np.shape[0], BR_y_adj
            )

            if np.any(self.binary_mask_np[TL_y_adj:BR_y_adj, TL_x_adj:BR_x_adj]):
                # If any part of the box is within the mask, keep it
                filtered_coordinates.append(box)

        assert len(filtered_coordinates) > 0, "No coordinates are within the mask area."

        return filtered_coordinates

    def save_images(self, save_dir):
        """Save the image, mask, overlayed image, blue_mask and grid representation of the top view in save_dir."""

        self.image.save(os.path.join(save_dir, "top_view_image.png"))
        self.mask.save(os.path.join(save_dir, "top_view_mask.png"))
        self.overlayed_image.save(
            os.path.join(save_dir, "top_view_overlayed_image.png")
        )
        self.blue_mask.save(os.path.join(save_dir, "top_view_blue_mask.png"))
        self.grid_rep.save(os.path.join(save_dir, "top_view_grid_rep.png"))


class SpecimenError(ValueError):
    """Exception raised when the specimen is not the correct type for the operation."""

    pass


class RelativeBlueSignalTooWeakError(ValueError):
    """Exception raised when the blue signal is too weak."""

    def __init__(self, message):
        """Initialize a BlueSignalTooWeakError object."""

        super().__init__(message)

    def __str__(self):
        """Return the error message."""

        return self.args[0]


class TopViewError(ValueError):
    """Exception raised when the top view is not the correct type for the operation."""

    def __init__(self, message):
        """Initialize a TopViewError object."""

        super().__init__(message)

def get_top_view_from_dzi(dzi_path, level):
    """
    Get the top view from the DZI file by stitching together all image tiles at the specified level.
    Handles variable tile sizes and calculates dimensions through iterative summation.
    
    Args:
        dzi_path (str): Path to the DZI file/directory
        level (int): The zoom level to extract
        
    Returns:
        PIL.Image: Stitched image containing the complete view at the specified level
    """
    
    # Construct the path to the level directory
    level_dir = os.path.join(dzi_path, str(level))
    
    # Check if the directory exists
    if not os.path.exists(level_dir):
        raise ValueError(f"Level {level} does not exist in the DZI path {dzi_path}")
    
    # Get all the image files in the directory
    image_files = [f for f in os.listdir(level_dir) if f.endswith('.jpg')]
    
    if not image_files:
        raise ValueError(f"No image files found in {level_dir}")
    
    # Extract row and column indices using regex
    pattern = r'(\d+)_(\d+)\.jpg'
    indices = []
    for img_file in image_files:
        match = re.match(pattern, img_file)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            indices.append((i, j, img_file))
    
    # Determine the grid dimensions
    max_i = max(idx[0] for idx in indices)
    max_j = max(idx[1] for idx in indices)
    
    # Load all images and store their dimensions
    tiles = {}
    for i, j, img_file in indices:
        img_path = os.path.join(level_dir, img_file)
        img = Image.open(img_path)
        tiles[(i, j)] = {
            'image': img,
            'width': img.width,
            'height': img.height
        }
    
    # Calculate row heights and column widths
    row_heights = {}
    col_widths = {}
    
    # Initialize with zeros
    for i in range(max_i + 1):
        col_widths[i] = 0
    for j in range(max_j + 1):
        row_heights[j] = 0
    
    # Find the maximum height for each row and width for each column
    for (i, j), tile_info in tiles.items():
        col_widths[i] = max(col_widths[i], tile_info['width'])
        row_heights[j] = max(row_heights[j], tile_info['height'])
    
    # Calculate the total width and height by summing
    total_width = sum(col_widths.values())
    total_height = sum(row_heights.values())
    
    # Create a new image with the calculated size
    stitched_img = Image.new('RGB', (total_width, total_height))
    draw = ImageDraw.Draw(stitched_img)
    
    # Calculate the starting position for each tile
    x_positions = {0: 0}
    for i in range(1, max_i + 1):
        x_positions[i] = x_positions[i-1] + col_widths[i-1]
    
    y_positions = {0: 0}
    for j in range(1, max_j + 1):
        y_positions[j] = y_positions[j-1] + row_heights[j-1]
    
    # Place each tile in the correct position
    for (i, j), tile_info in tiles.items():
        x = x_positions[i]
        y = y_positions[j]
        
        # Paste the image at the calculated position
        stitched_img.paste(tile_info['image'], (x, y))
    
    return stitched_img

if __name__ == "__main__":
    dzi_path = "/media/hdd2/neo/test_v5_dzi/test"
    save_path = "/home/neo/Documents/neo/LL-BMA-v5/test_image.jpg"
    level = 11
    top_view = get_top_view_from_dzi(dzi_path, level)

    top_view.save(save_path)