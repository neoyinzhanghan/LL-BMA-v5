import io
import os
import ray
import time
import openslide
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from LLBMA.BMAFocusRegion import FocusRegion
from LLBMA.resources.BMAassumptions import *

def ensure_dir_exists(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

@ray.remote
class WSICropManagerWithFRCreation:
    def __init__(self, wsi_path, topview) -> None:
        self.wsi_path = wsi_path
        self.wsi = None
        self.topview = topview

    def open_slide(self):
        self.wsi = openslide.OpenSlide(self.wsi_path)

    def close_slide(self):
        if self.wsi:
            self.wsi.close()
            self.wsi = None

    def get_level_0_dimensions(self):
        if self.wsi is None:
            self.open_slide()
        return self.wsi.dimensions

    def get_level_N_dimensions(self, wsi_level):
        if self.wsi is None:
            self.open_slide()
        return self.wsi.level_dimensions[wsi_level]

    def get_tile_coordinate_level_pairs(self, tile_size=512, wsi_level=0):
        if self.wsi is None:
            self.open_slide()

        width, height = self.get_level_N_dimensions(wsi_level)
        coordinates = []

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                coordinates.append(
                    (
                        (x, y, min(x + tile_size, width), min(y + tile_size, height)),
                        wsi_level,
                    )
                )

        return coordinates

    def crop(self, coords, wsi_level=0):
        if self.wsi is None:
            self.open_slide()

        coords_level_0 = (
            coords[0] * (2**wsi_level),
            coords[1] * (2**wsi_level),
            coords[2] * (2**wsi_level),
            coords[3] * (2**wsi_level),
        )

        image = self.wsi.read_region(
            (coords_level_0[0], coords_level_0[1]),
            wsi_level,
            (coords[2] - coords[0], coords[3] - coords[1]),
        )

        return image.convert("RGB")

    def async_process_tile_batch(self, tile_coords_level_pairs, output_dir, crop_size=512):
        tiles_info = []
        focus_regions = []
        
        for tile_coord_level_pair in tile_coords_level_pairs:
            tile_coord, wsi_level = tile_coord_level_pair
            image = self.crop(tile_coord, wsi_level=wsi_level)
            
            # Calculate DZI level (inverse of WSI level)
            dzi_level = 18 - wsi_level
            
            # Calculate tile coordinates
            col = tile_coord[0] // crop_size
            row = tile_coord[1] // crop_size
            
            # Create the level directory path
            level_dir = os.path.join(output_dir, str(dzi_level))
            ensure_dir_exists(level_dir)
            
            # Save the tile as JPEG
            tile_path = os.path.join(level_dir, f"{col}_{row}.jpg")
            image.save(tile_path, "JPEG", quality=90)
            
            tiles_info.append((dzi_level, col, row, tile_path))
            
            # Handle focus regions if at level 0
            if wsi_level == 0 and self.topview.is_in_mask(tile_coord):
                if (tile_coord[2] - tile_coord[0] == tile_coord[3] - tile_coord[1]):
                    downsampled_image = image.resize(
                        (
                            focus_regions_size // (2**search_view_level),
                            focus_regions_size // (2**search_view_level),
                        )
                    )
                    
                    focus_region = FocusRegion(
                        downsampled_coordinate=tile_coord,
                        downsampled_image=downsampled_image,
                    )
                    # focus_region.get_image(image) # TODO V5, we are removing this step to prevent RAM usage
                    focus_region.get_dzi_high_mag_image_path(tile_path)
                    focus_regions.append(focus_region)
        
        return tiles_info, focus_regions

def create_dzi_metadata(output_dir, image_width, image_height, tile_size=512):
    """Create DZI metadata file."""
    metadata = {
        "Image": {
            "xmlns": "http://schemas.microsoft.com/deepzoom/2008",
            "Format": "jpg",
            "Overlap": "0",
            "TileSize": str(tile_size),
            "Size": {
                "Width": str(image_width),
                "Height": str(image_height)
            }
        }
    }
    
    metadata_path = os.path.join(output_dir, "dzi_metadata.xml")
    with open(metadata_path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="jpg" Overlap="0" TileSize="{tile_size}">
    <Size Width="{image_width}" Height="{image_height}"/>
</Image>""")

def process_lower_levels(wsi_path, output_dir, tile_size=512):
    """Process lower resolution levels (11 to 0) from level 7 WSI image."""
    wsi = openslide.OpenSlide(wsi_path)
    level_7_dimensions = wsi.level_dimensions[7]
    image = wsi.read_region((0, 0), 7, level_7_dimensions).convert("RGB")

    for depth in range(10, -1, -1):
        current_image = image.resize(
            (
                max(image.width // (2 ** (10 - depth + 1)), 1),
                max(image.height // (2 ** (10 - depth + 1)), 1),
            )
        )
        
        level_dir = os.path.join(output_dir, str(depth))
        ensure_dir_exists(level_dir)
        
        for y in range(0, current_image.height, tile_size):
            for x in range(0, current_image.width, tile_size):
                right = min(x + tile_size, current_image.width)
                bottom = min(y + tile_size, current_image.height)
                
                patch = current_image.crop((x, y, right, bottom)).convert("RGB")
                tile_path = os.path.join(level_dir, f"{x//tile_size}_{y//tile_size}.jpg")
                patch.save(tile_path, "JPEG", quality=90)

def dzsave_with_FR_creation(
    wsi_path,
    output_dir,
    topview,
    tile_size=512,
    num_cpus=32,
    region_cropping_batch_size=512,
):
    """Create a DeepZoom image pyramid from a WSI, saving as JPEG files."""
    ensure_dir_exists(output_dir)
    
    wsi = openslide.OpenSlide(wsi_path)
    width, height = wsi.dimensions
    
    # Create DZI metadata
    create_dzi_metadata(output_dir, width, height, tile_size)
    
    print(f"Processing WSI of dimensions: {width}x{height}")
    start_time = time.time()
    
    # Initialize Ray workers
    manager = WSICropManagerWithFRCreation.remote(wsi_path, topview)
    task_managers = [
        WSICropManagerWithFRCreation.remote(wsi_path, topview)
        for _ in range(num_cpus)
    ]
    
    # Get all tile coordinates for levels 0-7
    focus_regions_coordinates = []
    for level in range(0, 8):
        focus_regions_coordinates.extend(
            ray.get(
                manager.get_tile_coordinate_level_pairs.remote(
                    tile_size=tile_size, wsi_level=level
                )
            )
        )
    
    # Process tiles in batches
    batches = [focus_regions_coordinates[i:i + region_cropping_batch_size] 
               for i in range(0, len(focus_regions_coordinates), region_cropping_batch_size)]
    
    tasks = {}
    focus_regions = []
    
    for i, batch in enumerate(batches):
        manager = task_managers[i % num_cpus]
        task = manager.async_process_tile_batch.remote(batch, output_dir, tile_size)
        tasks[task] = batch
    
    # Process tasks and collect results
    with tqdm(total=len(focus_regions_coordinates), desc="Processing tiles") as pbar:
        while tasks:
            done_ids, _ = ray.wait(list(tasks.keys()))
            for done_id in done_ids:
                try:
                    tiles_info, new_focus_regions = ray.get(done_id)
                    focus_regions.extend(new_focus_regions)
                    pbar.update(len(tasks[done_id]))
                except ray.exceptions.RayTaskError as e:
                    print(f"Task failed with error: {e}")
                del tasks[done_id]
    
    # Process lower resolution levels
    print("Processing lower resolution levels")
    process_lower_levels(wsi_path, output_dir, tile_size)
    
    time_taken = time.time() - start_time
    return time_taken, focus_regions

if __name__ == "__main__":
    slide_path = "/media/hdd3/neo/brenda_tmp/H18-9786;S10;MSKM - 2023-06-21 21.41.10.ndpi"
    output_dir = "/media/hdd3/neo/brenda_tmp/H18-9786;S10;MSKM - 2023-06-21 21.41.10_dzi"
    
    start_time = time.time()
    print(f"Processing slide at {slide_path}")
    time_taken, focus_regions = dzsave_with_FR_creation(
        wsi_path=slide_path,
        output_dir=output_dir,
        topview=None,  # You'll need to provide the topview object
        tile_size=512,
        num_cpus=32,
        region_cropping_batch_size=512,
    )
    
    print(f"Finished processing slide in {time_taken:.2f} seconds")