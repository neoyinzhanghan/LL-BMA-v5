# Standard library imports
import os
import io
import time
import base64
from xml.etree import ElementTree as ET

# Third-party imports
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from LLBMA.resources.BMAassumptions import *
from LLBMA.BMAFocusRegion import FocusRegion
from LLBMA.tiling.universal_wsi import UniversalWSI
def image_to_jpeg_string(image):
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="JPEG")
        jpeg_string = buffer.getvalue()
    finally:
        buffer.close()
    return jpeg_string

def jpeg_string_to_image(jpeg_string):
    buffer = io.BytesIO(jpeg_string)
    image = Image.open(buffer)
    image.load()
    buffer.close()
    return image

def encode_image_to_base64(jpeg_string):
    return base64.b64encode(jpeg_string)

def decode_image_from_base64(encoded_string):
    return base64.b64decode(encoded_string)

class WSITileDataset(Dataset):
    def __init__(self, wsi_path, tile_size, mpp_needed, save_dir, max_level=18):
        self.wsi_path = wsi_path
        self.uwsi = UniversalWSI(wsi_path)
        self.level_0_dimensions = self.uwsi.get_level_0_dimensions()
        self.level_0_mpp = self.uwsi.get_level_0_mpp()
        self.mpp_needed = mpp_needed
        self.scaling_factor = self.mpp_needed / self.level_0_mpp
        self.level_0_tile_size = int(tile_size * self.scaling_factor)
        self.tile_size = tile_size
        self.save_dir = save_dir
        self.level_save_dir = os.path.join(save_dir, str(max_level))
        self.width_for_mpp_needed = int(self.level_0_dimensions[0] * self.level_0_mpp / self.mpp_needed)
        self.height_for_mpp_needed = int(self.level_0_dimensions[1] * self.level_0_mpp / self.mpp_needed)
        if os.path.exists(self.level_save_dir):
            print(f"UserWarning: {self.level_save_dir} already exists, this leads to overwriting tiles")
        self.num_tiles_x = self.width_for_mpp_needed // self.tile_size + 1
        self.num_tiles_y = self.height_for_mpp_needed // self.tile_size + 1
        os.makedirs(self.level_save_dir, exist_ok=True)
        
        # Calculate the tile coordinates
        self.tile_coordinates = [(x, y) for x in range(self.num_tiles_x) for y in range(self.num_tiles_y)]
    def __len__(self):
        return len(self.tile_coordinates)
    def __getitem__(self, idx):
        # Write DZI metadata XML file if it doesn't exist yet
        if idx == 0:
            dzi_metadata = f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" Format="jpg" Overlap="0" TileSize="{self.tile_size}">
    <Size Width="{self.width_for_mpp_needed}" Height="{self.height_for_mpp_needed}"/>
</Image>"""
            with open(os.path.join(self.save_dir, "dzi_metadata.xml"), "w") as f:
                f.write(dzi_metadata)
        tile_coordinate = self.tile_coordinates[idx]

        pyramid_base_coordinate = (
            tile_coordinate[0] * self.tile_size,
            tile_coordinate[1] * self.tile_size,
        )
        pyramid_base_coordinate_BR = (
            min(int(pyramid_base_coordinate[0] + self.tile_size), self.width_for_mpp_needed),    # NOTE this is the maximum x and y coordinate that can be used to read the tile
            min(int(pyramid_base_coordinate[1] + self.tile_size), self.height_for_mpp_needed),
        )
        scaled_tile_coordinate = (
            min(int(tile_coordinate[0] * self.level_0_tile_size), self.level_0_dimensions[0]),
            min(int(tile_coordinate[1] * self.level_0_tile_size), self.level_0_dimensions[1]),
        )
        scaled_BR_tile_coordinate = (
            min(int((tile_coordinate[0] + 1) * self.level_0_tile_size), self.level_0_dimensions[0]),
            min(int((tile_coordinate[1] + 1) * self.level_0_tile_size), self.level_0_dimensions[1]),
        )
        scaled_coordinate = (
            scaled_tile_coordinate[0],
            scaled_tile_coordinate[1],
            scaled_BR_tile_coordinate[0],
            scaled_BR_tile_coordinate[1],
        )
        actual_tile_size = scaled_BR_tile_coordinate[0] - scaled_tile_coordinate[0], scaled_BR_tile_coordinate[1] - scaled_tile_coordinate[1]
        tile = self.uwsi.read_ground_region(scaled_tile_coordinate, actual_tile_size)
        # save the tile using x_y naming convention
        
        # calculate the ground level tile size using the pyramid base coordinate
        ground_level_tile_size_x = pyramid_base_coordinate_BR[0] - pyramid_base_coordinate[0]
        ground_level_tile_size_y = pyramid_base_coordinate_BR[1] - pyramid_base_coordinate[1]
        ground_level_tile_size = (ground_level_tile_size_x, ground_level_tile_size_y)
        
        try:
            # resize the tile to the ground level tile size
            tile = tile.resize(ground_level_tile_size)
        except Exception as e:
            print(f"Error resizing tile: {e}")
            print(f"Tile size: {tile.size}")
            print(f"Ground level tile size: {ground_level_tile_size}")
            print(f"Pyramid base coordinate: {pyramid_base_coordinate}")
            print(f"Pyramid base coordinate BR: {pyramid_base_coordinate_BR}")
            print(f"Scaled tile coordinate: {scaled_tile_coordinate}")
            print(f"Scaled BR tile coordinate: {scaled_BR_tile_coordinate}")
            print(f"Actual tile size: {actual_tile_size}")
            print(f"Tile coordinate: {tile_coordinate}")
            print(f"Tile size: {self.tile_size}")
            print(f"Width for mpp needed: {self.width_for_mpp_needed}")
            print(f"Height for mpp needed: {self.height_for_mpp_needed}")
            print(f"Level 0 dimensions: {self.level_0_dimensions}")
            print(f"Level 0 mpp: {self.level_0_mpp}")
            print(f"Mpp needed: {self.mpp_needed}")
            print(f"Scaling factor: {self.scaling_factor}")
            print(f"Level 0 tile size: {self.level_0_tile_size}")
            import sys
            sys.exit()
            raise e
        
        tile_path = os.path.join(self.level_save_dir, f"{tile_coordinate[0]}_{tile_coordinate[1]}.jpg")
        tile.save(tile_path)
        downsampled_image = tile.resize(
            (
                focus_regions_size // (2**search_view_level),
                focus_regions_size // (2**search_view_level),
            )
        )
        resampled_coordinate = (
            pyramid_base_coordinate[0],
            pyramid_base_coordinate[1],
            pyramid_base_coordinate_BR[0],
            pyramid_base_coordinate_BR[1],
        )
        fr = FocusRegion(resampled_coordinate=resampled_coordinate, level_0_coordinate=scaled_coordinate, downsampled_image=downsampled_image)
        fr.get_dzi_high_mag_image_path(tile_path)
        return fr
    
# define a null collate function which just returns None
def null_collate(batch):
    return None

def list_collate(batch):
    return batch

def tile_pyramid_base_dzi(wsi_path, save_dir, tile_size, mpp_needed, batch_size=32, num_workers=tiling_num_workers, max_level=18):
    if os.path.exists(save_dir):
        print(f"UserWarning: {save_dir} already exists, this leads to overwriting tiles")
    os.makedirs(save_dir, exist_ok=True)
    
    focus_regions = []
    dataset = WSITileDataset(wsi_path, tile_size, mpp_needed, save_dir, max_level)
    # create dataloader with 16 workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=list_collate)
    for frs in tqdm(dataloader, desc="Saving Tiles", total=len(dataloader)):
        focus_regions.extend(frs)
        
    return focus_regions

def calculate_max_x_y(pyramid_root_dir, current_level):
    current_level_dir = os.path.join(pyramid_root_dir, str(current_level))
    xs, ys = [], []
    for base_file in os.listdir(current_level_dir):
        if not base_file.endswith('.jpg'):
            continue
        base_file_name = base_file.replace('.jpg', '')
        xs.append(int(base_file_name.split('_')[0]))
        ys.append(int(base_file_name.split('_')[1]))
    return max(xs), max(ys)

def get_tile_image_from_dzi(file_dir_path, level, x, y):
    level_dir_path = os.path.join(file_dir_path, f'{level}')
    tile_file_path = os.path.join(level_dir_path, f'{x}_{y}.jpg')
    if os.path.exists(tile_file_path):
        return Image.open(tile_file_path)
    else:
        # print(f"Tile file path {tile_file_path} does not exist")
        return None

def combine_images(top_left_tile_image, top_right_tile_image, bottom_left_tile_image, bottom_right_tile_image):
    if top_left_tile_image is None:
        raise ValueError("Empty Tile Error: Top left tile image not found")
    elif top_right_tile_image is None and bottom_left_tile_image is None and bottom_right_tile_image is not None:
        raise ValueError("ChessBoard Error: Top right tile image and bottom left tile image not found, yet bottom right tile image found, this should not happen!")
    elif top_right_tile_image is None and bottom_left_tile_image is None:
        # return half_sized top_left_tile_image
        return top_left_tile_image.resize((max(top_left_tile_image.width // 2, 1), max(top_left_tile_image.height // 2, 1)))
    elif top_right_tile_image is None and bottom_left_tile_image is not None and bottom_right_tile_image is not None:
        raise ValueError("L-Shape Error: Top right tile image not found, bottom left is found, and bottom right tile image is found, this should not happen!")
    elif top_right_tile_image is None and bottom_left_tile_image is not None and bottom_right_tile_image is None:
        # stich together top_left_tile_image and bottom_left_tile_image
        # create a new image with the width of top_left_tile_image and the height of top_left_tile_image + bottom_left_tile_image
        new_image = Image.new('RGB', (top_left_tile_image.width, top_left_tile_image.height + bottom_left_tile_image.height))
        # paste top_left_tile_image in the top left corner
        new_image.paste(top_left_tile_image, (0, 0))
        # paste bottom_left_tile_image in the bottom left corner
        new_image.paste(bottom_left_tile_image, (0, top_left_tile_image.height))
        
        # resize the new image to the size of the bottom_left_tile_image
        new_image = new_image.resize((max(new_image.width // 2, 1), max(new_image.height // 2, 1)))
        return new_image
    elif top_right_tile_image is not None and bottom_left_tile_image is None and bottom_right_tile_image is not None:
        raise ValueError("L-Shape Error: Top right tile image is found, bottom left tile image is not found, yet bottom right tile image is found, this should not happen!")
    elif top_right_tile_image is not None and bottom_left_tile_image is None and bottom_right_tile_image is None:
        # stich together top_left_tile_image and top_right_tile_image
        # create a new image with the width of top_left_tile_image and the height of top_right_tile_image
        new_image = Image.new('RGB', (top_left_tile_image.width + top_right_tile_image.width, top_left_tile_image.height))
        # paste top_left_tile_image in the top left corner
        new_image.paste(top_left_tile_image, (0, 0))
        # paste top_right_tile_image in the top right corner
        new_image.paste(top_right_tile_image, (top_left_tile_image.width, 0))
        
        # resize the new image to the size of the top_right_tile_image
        new_image = new_image.resize((max(new_image.width // 2, 1), max(new_image.height // 2, 1)))
        return new_image
    elif bottom_right_tile_image is None:
        raise ValueError("L-Shape Error: All except the bottom right tile image not found, this should not happen!")
    else:
        # Check that adjacent edges have matching sizes
        if top_left_tile_image.width != bottom_left_tile_image.width:
            raise ValueError(f"Left edge size mismatch: Top left width ({top_left_tile_image.width}) != Bottom left width ({bottom_left_tile_image.width})")
        if top_right_tile_image.width != bottom_right_tile_image.width:
            raise ValueError(f"Right edge size mismatch: Top right width ({top_right_tile_image.width}) != Bottom right width ({bottom_right_tile_image.width})")
        if top_left_tile_image.height != top_right_tile_image.height:
            raise ValueError(f"Top edge size mismatch: Top left height ({top_left_tile_image.height}) != Top right height ({top_right_tile_image.height})")
        if bottom_left_tile_image.height != bottom_right_tile_image.height:
            raise ValueError(f"Bottom edge size mismatch: Bottom left height ({bottom_left_tile_image.height}) != Bottom right height ({bottom_right_tile_image.height})")

        # Create new image with correct dimensions from the tiles
        new_width = top_left_tile_image.width + top_right_tile_image.width
        new_height = top_left_tile_image.height + bottom_left_tile_image.height
        new_image = Image.new('RGB', (new_width, new_height))

        # Paste tiles in correct positions
        new_image.paste(top_left_tile_image, (0, 0))
        new_image.paste(top_right_tile_image, (top_left_tile_image.width, 0))
        new_image.paste(bottom_left_tile_image, (0, top_left_tile_image.height))
        new_image.paste(bottom_right_tile_image, (top_left_tile_image.width, top_left_tile_image.height))
        
        # Resize the combined image by half
        new_image = new_image.resize((max(new_width // 2, 1), max(new_height // 2, 1)))
        return new_image

    raise ValueError("Empty Return Error: This should not happen! Something is wrong with the tile images function!")
            
class WSIPyramidDataset(Dataset):
    def __init__(self, pyramid_root_dir, current_level, tile_size):
        self.tile_size = tile_size
        self.pyramid_root_dir = pyramid_root_dir
        self.current_level = current_level
        self.next_level = current_level - 1
        self.current_level_dir = os.path.join(pyramid_root_dir, str(self.current_level))
        self.next_level_dir = os.path.join(pyramid_root_dir, str(self.next_level))
        self.max_x, self.max_y = calculate_max_x_y(self.pyramid_root_dir, self.current_level)
        if (self.max_x + 1) % 2 == 0:
            self.num_tiles_x = (self.max_x + 1) // 2
        else:
            self.num_tiles_x = (self.max_x + 1) // 2 + 1
        if (self.max_y + 1) % 2 == 0:
            self.num_tiles_y = (self.max_y + 1) // 2
        else:
            self.num_tiles_y = (self.max_y + 1) // 2 + 1
        # self.tiles_coordinates = [(x, y) for x in range(self.num_tiles_x) for y in range(self.num_tiles_y)]
        self.tiles_coordinates = [(x, y) for y in range(self.num_tiles_y) for x in range(self.num_tiles_x)]
        # check if the next level directory exists
        if os.path.exists(self.next_level_dir):
            print(f"UserWarning: Next level {self.next_level_dir} already exists, skipping creation of next level!")
        os.makedirs(self.next_level_dir, exist_ok=True)

    def __len__(self):
        return len(self.tiles_coordinates)
    
    def __getitem__(self, idx):
        tile_coordinate = self.tiles_coordinates[idx]
        top_left_tile_coordinate_from_previous_level = (tile_coordinate[0] * 2, tile_coordinate[1] * 2)
        top_right_tile_coordinate_from_previous_level = (tile_coordinate[0] * 2 + 1, tile_coordinate[1] * 2)
        bottom_left_tile_coordinate_from_previous_level = (tile_coordinate[0] * 2, tile_coordinate[1] * 2 + 1)
        bottom_right_tile_coordinate_from_previous_level = (tile_coordinate[0] * 2 + 1, tile_coordinate[1] * 2 + 1)

        top_left_tile_image = get_tile_image_from_dzi(self.pyramid_root_dir, self.current_level, top_left_tile_coordinate_from_previous_level[0], top_left_tile_coordinate_from_previous_level[1])
        top_right_tile_image = get_tile_image_from_dzi(self.pyramid_root_dir, self.current_level, top_right_tile_coordinate_from_previous_level[0], top_right_tile_coordinate_from_previous_level[1])
        bottom_left_tile_image = get_tile_image_from_dzi(self.pyramid_root_dir, self.current_level, bottom_left_tile_coordinate_from_previous_level[0], bottom_left_tile_coordinate_from_previous_level[1])
        bottom_right_tile_image = get_tile_image_from_dzi(self.pyramid_root_dir, self.current_level, bottom_right_tile_coordinate_from_previous_level[0], bottom_right_tile_coordinate_from_previous_level[1])
        
        try:
            combined_image = combine_images(top_left_tile_image, top_right_tile_image, bottom_left_tile_image, bottom_right_tile_image)
        except ValueError as e:
            print(f"ValueError: {e}")
            print(f"Tile coordinate: {tile_coordinate}")
            raise e
        
        # save the combined image in the next level directory
        if not os.path.exists(self.next_level_dir):
            os.makedirs(self.next_level_dir)
        combined_image.save(os.path.join(self.next_level_dir, f'{tile_coordinate[0]}_{tile_coordinate[1]}.jpg'))
        
        return None
        
def create_pyramid_next_level_dzi(pyramid_base_dir, tile_size, current_level, num_workers, batch_size):
    if current_level <= 0:
        raise ValueError(f"Current level {current_level} must be greater than 0")
    dataset = WSIPyramidDataset(pyramid_base_dir, current_level, tile_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=null_collate)
    for batch in tqdm(dataloader, desc="Creating Pyramid Next Level", total=len(dataloader)):
        pass
    
def universal_dzsave_dzi(wsi_path, save_dir, tile_size, mpp_needed, batch_size=tiling_batch_size, num_workers=tiling_num_workers, num_workers_pyramid=tiling_num_workers, max_level=18):
    start_time = time.time()
    focus_regions = tile_pyramid_base_dzi(wsi_path, save_dir, tile_size, mpp_needed, batch_size, num_workers, max_level)
    
    current_level = max_level
    pyramid_base_dir = save_dir    
    while current_level > 0:
        if current_level > 16:
            print(f"Creating pyramid level {current_level-1} from level {current_level}")
            create_pyramid_next_level_dzi(pyramid_base_dir, tile_size, current_level, num_workers_pyramid, batch_size)
        else:
            print(f"Creating pyramid level {current_level-1} from level {current_level}")
            create_pyramid_next_level_dzi(pyramid_base_dir, tile_size, current_level, 1, 1)
            
        current_level -= 1    
        
    time_taken = time.time() - start_time
    
    return time_taken, focus_regions
        
def tile_pyramid_base_dzi_with_FR_creation(wsi_path, save_dir, tile_size, mpp_needed, batch_size=32, num_workers=tiling_num_workers, max_level=18):
    if os.path.exists(save_dir):
        print(f"UserWarning: {save_dir} already exists, this leads to overwriting tiles")
    os.makedirs(save_dir, exist_ok=True)
    dataset = WSITileDataset(wsi_path, tile_size, mpp_needed, save_dir, max_level)
    # create dataloader with 16 workers
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=null_collate)
    for batch in tqdm(dataloader, desc="Saving Tiles", total=len(dataloader)):
        pass
        
def extract_from_xml(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Define the namespace
    namespace = {'ns': 'http://schemas.microsoft.com/deepzoom/2008'}
    
    # Extract values using namespace
    image_width = int(root.find('ns:Size', namespace).get('Width'))
    image_height = int(root.find('ns:Size', namespace).get('Height'))
    tile_size = int(root.get('TileSize'))
    overlap = int(root.get('Overlap'))
    format = root.get('Format')
    
    return image_width, image_height, tile_size, overlap, format
