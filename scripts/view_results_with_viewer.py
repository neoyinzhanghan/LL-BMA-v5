import os
import argparse
import json
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, send_file
from werkzeug.serving import run_simple
from LLBMA.resources.BMAassumptions import cellnames_dict, differential_group_dict_display
from io import BytesIO

app = Flask(__name__)

# Global variables to store paths
RESULT_FOLDER = None
DZI_PATH = None
# Path to the logo
LOGO_PATH = '/home/neo/Documents/neo/LL-BMA-v5/LLBMA/resources/logo_76.png'
# Cache for focus regions dataframe
FOCUS_REGIONS_DF = None

def _add_yellow_boundary(pil_image):
    """Add a yellow boundary to the image (for debugging)"""
    # Get the current size of the image
    width, height = pil_image.size

    # If the image is too small to have an 8-pixel boundary, just make it yellow
    if width <= 16 or height <= 16:
        return Image.new(
            "RGB", (width, height), (255, 255, 0)
        )  # Return a completely yellow image

    # Load the image data into a list of pixels
    pixels = pil_image.load()

    # Apply a yellow boundary of 8 pixels on each side (top, bottom, left, right)
    for y in range(8):  # Top and bottom boundaries
        for x in range(width):
            pixels[x, y] = (255, 255, 0)  # Top row
            pixels[x, height - y - 1] = (255, 255, 0)  # Bottom row

    for x in range(8):  # Left and right boundaries
        for y in range(height):
            pixels[x, y] = (255, 255, 0)  # Left column
            pixels[width - x - 1, y] = (255, 255, 0)  # Right column

    return pil_image


def _add_green_boundary(pil_image):
    """Add a green boundary to the image (for focus regions)"""
    # Get the current size of the image
    width, height = pil_image.size

    # If the image is too small to have an 8-pixel boundary, just make it green
    if width <= 16 or height <= 16:
        return Image.new(
            "RGB", (width, height), (0, 255, 0)
        )  # Return a completely green image

    # Load the image data into a list of pixels
    pixels = pil_image.load()

    # Apply a green boundary of 8 pixels on each side (top, bottom, left, right)
    for y in range(8):  # Top and bottom boundaries
        for x in range(width):
            pixels[x, y] = (0, 255, 0)  # Top row
            pixels[x, height - y - 1] = (0, 255, 0)  # Bottom row

    for x in range(8):  # Left and right boundaries
        for y in range(height):
            pixels[x, y] = (0, 255, 0)  # Left column
            pixels[width - x - 1, y] = (0, 255, 0)  # Right column

    return pil_image


def string_to_tuple(input_str):
    """Convert a string representation of a tuple to an actual tuple"""
    # Remove the parentheses and split by commas
    return tuple(map(int, input_str.strip("()").split(", ")))


def extract_slide_name_from_dzi_path():
    """Extract slide name from DZI path for use in results lookup"""
    if not DZI_PATH:
        return None
    
    # Get the base directory name
    base_name = os.path.basename(os.path.normpath(DZI_PATH))
    
    # Handle different naming patterns
    if "_dzi" in base_name:
        # Format: slidename_dzi
        return base_name.split("_dzi")[0] + ".h5"
    else:
        # Fallback: just use the directory name
        return base_name + ".h5"


def get_LLBMA_processing_status(slide_h5_name):
    """Check if a slide has been processed by LLBMA"""
    # Extract the slide name without extension
    wsi_name_no_ext = slide_h5_name.split(".h5")[0]
    
    # Check if result subdir exists in the input RESULT_FOLDER
    subdir = os.path.join(RESULT_FOLDER, wsi_name_no_ext)
    subdir_error = os.path.join(RESULT_FOLDER, f"ERROR_{wsi_name_no_ext}")
    
    if os.path.exists(subdir):
        return "Processed"
    elif os.path.exists(subdir_error):
        return "Error"
    else:
        return "Not Processed"


def get_annotated_focus_region_indices_and_coordinates(slide_h5_name):
    """Get focus regions information from the provided result folder"""
    
    df_dict = {
        "high_mag_score": [],
        "idx": [],
        "row": [],
        "col": [],
        "coordinate": [],
        "image_path": [],
    }
    
    for level in range(19):
        df_dict[f"x_{level}"] = []
        df_dict[f"y_{level}"] = []
    
    status = get_LLBMA_processing_status(slide_h5_name)
    if status == "Error" or status == "Not Processed":
        raise ValueError(
            f"Cannot get annotated focus regions for {slide_h5_name}. Status: {status}"
        )
    
    # Get the subdir for the slide_h5_name within the input RESULT_FOLDER
    wsi_name_no_ext = slide_h5_name.split(".h5")[0]
    subdir = os.path.join(RESULT_FOLDER, wsi_name_no_ext)
    
    # Get the high_mag_focus_regions_info.csv file from the selected_focus_regions subdir
    high_mag_focus_regions_info_path = os.path.join(
        subdir, "selected_focus_regions", "high_mag_focus_regions_info.csv"
    )
    
    if not os.path.exists(high_mag_focus_regions_info_path):
        print(f"Warning: Could not find focus regions info at {high_mag_focus_regions_info_path}")
        return pd.DataFrame(df_dict)
    
    # Read the high_mag_focus_regions_info.csv file
    high_mag_focus_regions_info_df = pd.read_csv(high_mag_focus_regions_info_path)
    
    for i, df_row in high_mag_focus_regions_info_df.iterrows():
        high_mag_score = df_row["adequate_confidence_score_high_mag"]
        
        # Round the high_mag_score to 3 decimal places
        high_mag_score = round(high_mag_score, 3)
        idx = df_row["idx"]
        coordinate_string = df_row["coordinate"]
        coordinate = df_row["coordinate"]
        
        TLx, TLy, BRx, BRy = string_to_tuple(coordinate_string)
        
        row = TLx // 512
        col = TLy // 512
        
        # Use the input RESULT_FOLDER to construct the image path
        image_path = os.path.join(
            subdir,
            "selected_focus_regions",
            "high_mag_annotated",
            f"{idx}.jpg",
        )
        
        df_dict["high_mag_score"].append(high_mag_score)
        df_dict["idx"].append(idx)
        df_dict["row"].append(row)
        df_dict["col"].append(col)
        df_dict["coordinate"].append(coordinate)
        df_dict["image_path"].append(image_path)
        
        for level in range(19):
            downsample_level = 18 - level
            downsample_factor = 2**downsample_level
            
            df_dict[f"x_{level}"].append(TLx / downsample_factor)
            df_dict[f"y_{level}"].append(TLy / downsample_factor)
    
    return pd.DataFrame(df_dict)


def get_annotated_tile(
    tile_image, tile_row, tile_col, tile_level, focus_regions_df, debug_mode=True
):
    """Apply annotations to a tile based on focus regions"""
    print(f"get_annotated_tile: level={tile_level}, row={tile_row}, col={tile_col}, debug_mode={debug_mode}")
    
    # If no focus regions or annotations disabled, return the original tile
    if focus_regions_df is None or len(focus_regions_df) == 0:
        print("No focus regions or annotations disabled")
        if debug_mode:
            print("Adding yellow boundary (debug mode)")
            tile_image = _add_yellow_boundary(tile_image)
        return tile_image

    # For very zoomed out levels (0-9), don't modify the tile
    if tile_level <= 9:
        print(f"Low zoom level ({tile_level}), not modifying beyond debug")
        if debug_mode:
            print("Adding yellow boundary (debug mode)")
            tile_image = _add_yellow_boundary(tile_image)
        return tile_image

    # For medium zoom levels (10-14), highlight focus regions with green rectangles
    elif tile_level < 15:
        print(f"Medium zoom level ({tile_level}), highlighting focus regions")
        found = False
        # Iterate over focus regions
        for idx, df_row in focus_regions_df.iterrows():
            level_x, level_y = df_row[f"x_{tile_level}"], df_row[f"y_{tile_level}"]

            # Calculate position within the tile
            region_translation_x, region_translation_y = int(tile_row * 512), int(
                tile_col * 512
            )
            rel_level_x, rel_level_y = (
                int(level_x - region_translation_x),
                int(level_y - region_translation_y),
            )

            # Calculate the size of the focus region at this zoom level
            region_level_width, region_level_height = int(
                512 // 2 ** (18 - tile_level)
            ), int(512 // 2 ** (18 - tile_level))

            # Check if the focus region is visible in this tile
            if 0 <= rel_level_x < 512 and 0 <= rel_level_y < 512:
                print(f"Found focus region in tile: region_idx={idx}, rel_pos=({rel_level_x},{rel_level_y})")
                # Create a green rectangle
                tile_array = np.array(tile_image)
                tile_array[
                    rel_level_y : rel_level_y + region_level_height,
                    rel_level_x : rel_level_x + region_level_width,
                ] = [0, 255, 0]
                tile_image = Image.fromarray(tile_array)
                found = True

        # Add border if needed
        if not found and debug_mode:
            print("No focus regions found in tile, adding yellow boundary (debug mode)")
            tile_image = _add_yellow_boundary(tile_image)
        elif found:
            print("Focus regions found, adding green boundary")
            tile_image = _add_green_boundary(tile_image)
            
        return tile_image

    # For high zoom levels (15-17), overlay thumbnails of the annotated images
    elif tile_level < 18:
        print(f"High zoom level ({tile_level}), overlaying annotated thumbnails")
        found = False
        # Iterate over focus regions
        for idx, df_row in focus_regions_df.iterrows():
            level_x, level_y = df_row[f"x_{tile_level}"], df_row[f"y_{tile_level}"]

            # Calculate position within the tile
            region_translation_x, region_translation_y = int(tile_row * 512), int(
                tile_col * 512
            )
            rel_level_x, rel_level_y = (
                int(level_x - region_translation_x),
                int(level_y - region_translation_y),
            )

            # Calculate the size of the focus region at this zoom level
            region_level_width, region_level_height = int(
                512 // 2 ** (18 - tile_level)
            ), int(512 // 2 ** (18 - tile_level))

            # Check if the focus region is visible in this tile
            if 0 <= rel_level_x < 512 and 0 <= rel_level_y < 512:
                # Get the path to the annotated image
                image_path = df_row["image_path"]
                print(f"Found focus region in tile: region_idx={idx}, image_path={image_path}")

                if os.path.exists(image_path):
                    # Load and process the tile
                    tile_array = np.array(tile_image)
                    
                    # Open the annotated image
                    try:
                        image = Image.open(image_path)
                        # Resize to fit the region
                        image = image.resize((region_level_width, region_level_height))
                        # Overlay on the tile
                        image_array = np.array(image)
                        tile_array[
                            rel_level_y : rel_level_y + region_level_height,
                            rel_level_x : rel_level_x + region_level_width,
                        ] = image_array
                        tile_image = Image.fromarray(tile_array)
                        found = True
                        print(f"Successfully overlaid annotated image")
                    except Exception as e:
                        print(f"Error processing annotated image {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Annotated image file not found: {image_path}")

        # Add border if needed
        if not found and debug_mode:
            print("No focus regions found in tile, adding yellow boundary (debug mode)")
            tile_image = _add_yellow_boundary(tile_image)
        elif found:
            print("Focus regions found, adding green boundary")
            tile_image = _add_green_boundary(tile_image)
            
        return tile_image

    # For maximum zoom level (18), replace with the annotated image if available
    else:
        print(f"Max zoom level ({tile_level}), checking for full replacement")
        found = False
        # Iterate over focus regions
        for idx, df_row in focus_regions_df.iterrows():
            img_row, img_col = df_row["row"], df_row["col"]

            # Check if this tile matches a focus region
            if tile_row == img_row and tile_col == img_col:
                image_path = df_row["image_path"]
                print(f"Found exact match for focus region: region_idx={idx}, image_path={image_path}")

                if os.path.exists(image_path):
                    try:
                        # Replace the tile with the annotated image
                        tile_image = Image.open(image_path)
                        found = True
                        print(f"Successfully replaced tile with annotated image")
                    except Exception as e:
                        print(f"Error loading annotated image {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Annotated image file not found: {image_path}")

        # Add border if needed
        if found:
            print("Focus region found, adding green boundary")
            tile_image = _add_green_boundary(tile_image)
        else:
            print("No focus regions found in tile, adding yellow boundary (debug mode)")
            tile_image = _add_yellow_boundary(tile_image)
            
        return tile_image

def get_image_paths():
    """
    Scan the result folder structure and return organized image paths
    """
    result = {
        'regions': {
            'unannotated': [],
            'annotated': []
        },
        'cells': {}
    }
    
    # Get region images (unannotated)
    unannotated_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_unannotated')
    if os.path.exists(unannotated_dir):
        for file in sorted(os.listdir(unannotated_dir)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                result['regions']['unannotated'].append(file)
    
    # Get region images (annotated)
    annotated_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_annotated')
    if os.path.exists(annotated_dir):
        for file in sorted(os.listdir(annotated_dir)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                result['regions']['annotated'].append(file)
    
    # Get cell images from subdirectories
    cells_dir = os.path.join(RESULT_FOLDER, 'selected_cells')
    if os.path.exists(cells_dir):
        for subdir in sorted(os.listdir(cells_dir)):
            subdir_path = os.path.join(cells_dir, subdir)
            if os.path.isdir(subdir_path):
                cell_class = f"{subdir} - {cellnames_dict.get(subdir, 'Unknown')}"
                result['cells'][cell_class] = []
                for file in sorted(os.listdir(subdir_path)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        result['cells'][cell_class].append(file)
    
    return result

def get_differential_counts():
    """
    Calculate differential counts based on the cell types found in the result folder
    """
    differential_counts = {
        "total": 0,
        "total_for_differential": 0,
        "regions_count": 0,
        "groups": {}
    }
    
    # Initialize all groups with zero counts
    for group_name in differential_group_dict_display.keys():
        differential_counts["groups"][group_name] = 0
    
    # Count regions
    regions_dir = os.path.join(RESULT_FOLDER, 'selected_focus_regions', 'high_mag_unannotated')
    if os.path.exists(regions_dir):
        differential_counts["regions_count"] = len([f for f in os.listdir(regions_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    
    # Count cells in each directory
    cells_dir = os.path.join(RESULT_FOLDER, 'selected_cells')
    if os.path.exists(cells_dir):
        for subdir in os.listdir(cells_dir):
            subdir_path = os.path.join(cells_dir, subdir)
            if os.path.isdir(subdir_path):
                # Count images in this cell type directory
                cell_count = len([f for f in os.listdir(subdir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
                
                # Add to total count
                differential_counts["total"] += cell_count
                
                # Find which group this cell type belongs to
                for group_name, cell_types in differential_group_dict_display.items():
                    if subdir in cell_types:
                        differential_counts["groups"][group_name] += cell_count
                        # Add to total for differential if not "Skipped Cells & Artifacts"
                        if group_name != "Skipped Cells & Artifacts":
                            differential_counts["total_for_differential"] += cell_count
                        break
    
    # Calculate percentages
    differential_percentages = {"groups": {}}
    total_for_differential = differential_counts["total_for_differential"]
    total_cells = differential_counts["total"]
    
    for group_name, count in differential_counts["groups"].items():
        if group_name == "Skipped Cells & Artifacts" and total_cells > 0:
            # Calculate percentage of skipped cells based on total cells
            percentage = (count / total_cells) * 100
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": round(percentage, 1),
                "is_na": False
            }
        elif total_for_differential > 0:
            percentage = (count / total_for_differential) * 100
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": round(percentage, 1),
                "is_na": False
            }
        else:
            differential_percentages["groups"][group_name] = {
                "count": count,
                "percentage": 0,
                "is_na": False
            }
    
    # Add total counts to the result
    differential_percentages["total"] = differential_counts["total"]
    differential_percentages["total_for_differential"] = differential_counts["total_for_differential"]
    differential_percentages["regions_count"] = differential_counts["regions_count"]
    
    return differential_percentages

def get_dimensions(dzi_dir):
    """Get dimensions from DZI metadata file."""
    metadata_path = os.path.join(dzi_dir, "dzi_metadata.xml")
    try:
        tree = ET.parse(metadata_path)
        root = tree.getroot()
        size = root.find("{http://schemas.microsoft.com/deepzoom/2008}Size")
        width = int(size.get("Width"))
        height = int(size.get("Height"))
        return width, height
    except Exception as e:
        print(f"Error reading DZI metadata: {e}")
        return 10000, 10000  # Default values if metadata can't be read

def retrieve_tile_dzi(slide_path, level, row, col):
    """Retrieve a tile from the DZI directory structure."""
    tile_path = os.path.join(slide_path, str(level), f"{row}_{col}.jpg")
    try:
        if os.path.exists(tile_path):
            return Image.open(tile_path)
        else:
            # Return a blank tile if the requested tile doesn't exist
            return Image.new('RGB', (512, 512), 'white')
    except Exception as e:
        print(f"Error retrieving tile at {tile_path}: {e}")
        # Return a blank tile in case of error
        return Image.new('RGB', (512, 512), 'white')

@app.route('/get_dimensions', methods=["GET"])
def get_dimensions_api():
    """API endpoint to get slide dimensions"""
    if not DZI_PATH or not os.path.exists(DZI_PATH):
        return jsonify({"width": 10000, "height": 10000})
    
    width, height = get_dimensions(DZI_PATH)
    return jsonify({"width": width, "height": height})

@app.route('/check_state', methods=["GET"])
def check_state():
    """Return the current state of annotations and debug mode"""
    return jsonify({
        "focus_regions_count": len(FOCUS_REGIONS_DF) if FOCUS_REGIONS_DF is not None else 0,
        "focus_regions_columns": list(FOCUS_REGIONS_DF.columns) if FOCUS_REGIONS_DF is not None else []
    })

@app.route('/tile_api', methods=["GET"])
def tile_api():
    """API endpoint to serve DZI tiles"""
    if not DZI_PATH or not os.path.exists(DZI_PATH):
        # Return a blank tile if DZI path is not set
        blank_tile = Image.new('RGB', (512, 512), 'white')
        img_io = io.BytesIO()
        blank_tile.save(img_io, format="JPEG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    
    level = int(request.args.get("level"))
    row = int(request.args.get("x"))
    col = int(request.args.get("y"))
    
    # Get the raw tile
    tile = retrieve_tile_dzi(DZI_PATH, level, row, col)
    
    img_io = io.BytesIO()
    tile.save(img_io, format="JPEG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")

@app.route('/annotated_tile_api', methods=["GET"])
def annotated_tile_api():
    """API endpoint to serve annotated DZI tiles"""
    if not DZI_PATH or not os.path.exists(DZI_PATH):
        # Return a blank tile if DZI path is not set
        blank_tile = Image.new('RGB', (512, 512), 'white')
        img_io = io.BytesIO()
        blank_tile.save(img_io, format="JPEG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")
    
    level = int(request.args.get("level"))
    row = int(request.args.get("x"))
    col = int(request.args.get("y"))
    
    # Get the raw tile
    tile = retrieve_tile_dzi(DZI_PATH, level, row, col)
    
    # Handle annotations if needed
    if get_LLBMA_processing_status(RESULT_FOLDER) == "Processed":
        df = get_annotated_focus_region_indices_and_coordinates(RESULT_FOLDER)
        tile = get_annotated_tile(
            tile_image=tile,
            tile_row=row,
            tile_col=col,
            tile_level=level,
            focus_regions_df=df,
            debug_mode=False,
        )

    img_io = io.BytesIO()
    tile.save(img_io, format="JPEG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/jpeg")


@app.route('/')
def index():
    if not RESULT_FOLDER or not os.path.exists(RESULT_FOLDER):
        return "Error: Result folder not found or not set."
    
    image_paths = get_image_paths()
    differential_data = get_differential_counts()
    has_wsi = DZI_PATH is not None and os.path.exists(DZI_PATH)
    
    return render_template('index.html', 
                          image_paths=image_paths, 
                          result_folder=RESULT_FOLDER,
                          differential_data=differential_data,
                          has_wsi=has_wsi)

@app.route('/logo')
def get_logo():
    """Serve the logo file"""
    if os.path.exists(LOGO_PATH):
        directory, filename = os.path.split(LOGO_PATH)
        return send_from_directory(directory, filename)
    return ""

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon (using the logo)"""
    if os.path.exists(LOGO_PATH):
        directory, filename = os.path.split(LOGO_PATH)
        return send_from_directory(directory, filename)
    return ""

def create_templates():
    """Create the templates directory and index.html file"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>LeukoLocator - Image Viewer</title>
    <link rel="icon" href="/favicon.ico" type="image/x-icon">
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/openseadragon.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
        
        :root {
            --primary: #6741B2;
            --primary-dark: #270F7E;
            --accent: #BE98B3;
            --secondary: #7574C4;
            --light-bg: #D8D7E5;
        }
        
        * {
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Nunito', 'Avenir', 'Helvetica Neue', sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f8fa;
            color: #333;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        .header {
            background-color: rgba(216, 215, 229, 0.5);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            width: 50px;
            height: 50px;
            margin-right: 15px;
        }
        
        .app-title {
            font-size: 24px;
            font-weight: 700;
            margin: 0;
        }
        
        .main-wrapper {
            display: flex;
            flex: 1;
        }
        
        .sidebar {
            width: 300px;
            background-color: white;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
            padding: 20px;
            overflow-y: auto;
            transition: all 0.3s ease;
            position: sticky;
            top: 80px;
            height: calc(100vh - 80px);
            z-index: 90;
        }
        
        .sidebar-collapsed {
            width: 40px;
            padding: 20px 10px;
            overflow: hidden;
        }
        
        .sidebar-collapsed h2, 
        .sidebar-collapsed .summary-stats, 
        .sidebar-collapsed .differential-container {
            opacity: 0;
            visibility: hidden;
        }
        
        .sidebar-toggle {
            position: absolute;
            right: 10px;
            top: 20px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 4px;
            width: 30px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 95;
            transition: transform 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }
        
        .sidebar-collapsed .sidebar-toggle {
            transform: rotate(180deg);
            left: 5px;
        }
        
        .container {
            flex: 1;
            padding: 20px;
            transition: all 0.3s ease;
            width: calc(100% - 300px);
        }
        
        .container-expanded {
            width: calc(100% - 40px);
        }
        
        h1, h2, h3 {
            color: var(--primary-dark);
        }
        
        .image-container {
            margin-bottom: 40px;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .image-item {
            margin-bottom: 20px;
            max-width: 300px;
            cursor: pointer;
        }
        
        .region-image {
            width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .cell-image {
            width: 96px;
            height: 96px;
            object-fit: cover;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .cell-item {
            margin-bottom: 20px;
            max-width: 100px;
        }
        
        button {
            padding: 8px 15px;
            background-color: var(--secondary);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
            font-family: 'Nunito', 'Avenir', sans-serif;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        
        button:hover {
            background-color: var(--primary);
        }
        
        .control-panel {
            background-color: var(--light-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .section-title {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--light-bg);
            padding-bottom: 10px;
        }
        
        .section-title h2 {
            margin: 0;
        }
        
        #loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(39, 15, 126, 0.9);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .loading-text {
            margin-bottom: 20px;
            font-size: 24px;
            color: white;
            font-weight: 600;
        }
        
        .logo-container {
            margin-bottom: 30px;
        }
        
        .spinning-logo {
            width: 100px;
            height: 100px;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .progress-container {
            width: 70%;
            max-width: 500px;
            background-color: var(--light-bg);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 30px;
            width: 0%;
            background-color: var(--accent);
            text-align: center;
            line-height: 30px;
            color: white;
            font-weight: 600;
            transition: width 0.3s;
        }
        
        #main-content {
            display: none;
        }
        
        .toggle-all-btn {
            background-color: var(--primary);
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .toggle-all-btn:hover {
            background-color: var(--primary-dark);
        }
        
        .differential-container {
            margin-bottom: 20px;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        
        .differential-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        
        .differential-item:last-child {
            border-bottom: none;
        }
        
        .differential-name {
            font-weight: 600;
        }
        
        /* Styling for skipped cells text */
        .skipped-text {
            color: #e74c3c;
        }
        
        .differential-value {
            display: flex;
            align-items: center;
        }
        
        .differential-count {
            margin-right: 10px;
            color: var(--primary-dark);
            font-weight: 600;
        }
        
        .differential-percentage {
            color: var(--secondary);
        }
        
        .bar-container {
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .bar {
            height: 100%;
            background-color: var(--primary);
            border-radius: 10px;
        }
        
        /* Special styling for skipped cells bar */
        .bar-skipped {
            background-color: #e74c3c; /* Red color for skipped cells */
        }
        
        .bar-label {
            font-size: 11px;
            color: #666;
            margin-top: 3px;
            font-style: italic;
        }
        
        .summary-stats {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            transition: opacity 0.3s ease, visibility 0.3s ease;
        }
        
        .summary-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
        }
        
        .summary-label {
            font-weight: 600;
        }
        
        .summary-value {
            color: var(--primary-dark);
            font-weight: 700;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }
        
        .modal-content {
            background-color: #fefefe;
            margin: 15% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 600px;
            border-radius: 8px;
        }
        
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover,
        .close:focus {
            color: black;
            text-decoration: none;
        }
        
        .path-display {
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            word-break: break-all;
            font-family: monospace;
        }
        
        .copy-btn {
            background-color: var(--primary);
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        
        .copy-btn:hover {
            background-color: var(--primary-dark);
        }
        
        .copy-message {
            color: green;
            margin-top: 10px;
            display: none;
        }
        
        /* Whole Slide Image viewer styles */
        #wsi-container {
            width: 100%;
            height: 600px;
            margin-bottom: 20px;
        }
        
        .wsi-viewer {
            width: 100%;
            height: 100%;
            background-color: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        
        /* New styles for WSI controls */
        .wsi-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
        }
        
        .wsi-toggle-btn {
            background-color: var(--primary);
            color: white;
            padding: 8px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
        }
        
        .wsi-toggle-btn:hover {
            background-color: var(--primary-dark);
        }
        
        .wsi-toggle-btn.active {
            background-color: var(--accent);
        }
        
        /* Switch styles */
        .switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        
        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 34px;
        }
        
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: #2ecc71;
        }
        
        input:focus + .slider {
            box-shadow: 0 0 1px #2ecc71;
        }
        
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        
        .switch-label {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .sidebar-note {
            font-size: 12px;
            font-style: italic;
            color: #666;
            margin-top: 15px;
            padding-top: 10px;
            border-top: 1px dashed #ccc;
        }
    </style>
</head>
<body>
    <!-- Loading overlay -->
    <div id="loading-overlay">
        <div class="logo-container">
            <img src="/logo" class="spinning-logo" alt="LeukoLocator Logo">
        </div>
        <div class="loading-text">Loading LeukoLocator...</div>
        <div class="progress-container">
            <div id="progress-bar" class="progress-bar">0%</div>
        </div>
    </div>

    <!-- Path display modal -->
    <div id="path-modal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <h2>Image Path</h2>
            <div id="path-display" class="path-display"></div>
            <button id="copy-path-btn" class="copy-btn">Copy to Clipboard</button>
            <div id="copy-message" class="copy-message">Path copied to clipboard!</div>
        </div>
    </div>

    <!-- Main content (hidden initially) -->
    <div id="main-content">
        <header class="header">
            <img src="/logo" alt="LeukoLocator Logo" class="logo">
            <h1 class="app-title">LeukoLocator</h1>
        </header>
        
        <div class="main-wrapper">
            <!-- Sidebar with differential counts -->
            <div id="sidebar" class="sidebar">
                <button id="sidebar-toggle" class="sidebar-toggle">◀</button>
                <h2>Analysis Summary</h2>
                
                <!-- Summary statistics -->
                <div class="summary-stats">
                    <div class="summary-item">
                        <span class="summary-label">Total Regions:</span>
                        <span class="summary-value">{{ differential_data.regions_count }}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Objects Detected:</span>
                        <span class="summary-value">{{ differential_data.total }}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Cells in Differential:</span>
                        <span class="summary-value">{{ differential_data.total_for_differential }}</span>
                    </div>
                </div>
                
                <h2>Differential Counts</h2>
                
                <div class="differential-container">
                    {% for group_name, data in differential_data.groups.items() %}
                    {% if group_name == 'Skipped Cells & Artifacts' %}
                    <div class="differential-item">
                        <div class="skipped-text" style="width: 100%;">
                            Found {{ data.count }} Skipped cells & artifacts, {{ data.percentage }}% of all objects
                        </div>
                    </div>
                    {% else %}
                    <div class="differential-item">
                        <div class="differential-name">{{ group_name|capitalize }}</div>
                        <div class="differential-value">
                            <span class="differential-count">{{ data.count }}</span>
                            <span class="differential-percentage">
                                {% if data.is_na %}
                                    ({{ data.percentage }})
                                {% else %}
                                    ({{ data.percentage }}%)
                                {% endif %}
                            </span>
                        </div>
                    </div>
                    {% if not data.is_na %}
                    <div class="bar-container">
                        <div class="bar" style="width: {{ data.percentage }}%;"></div>
                    </div>
                    {% endif %}
                    {% endif %}
                    {% endfor %}
                    
                    <!-- Note about skipped cells -->
                    <div class="sidebar-note">
                        * Skipped cells & artifacts are not included in the differential count calculations
                    </div>
                </div>
            </div>
            
            <!-- Main content container -->
            <div id="content-container" class="container">
                <h1>Slide Analysis Results</h1>
                
                <!-- Whole Slide Image Section -->
                {% if has_wsi %}
                <div class="image-container">
                    <div class="section-title">
                        <h2>Whole Slide Image</h2>
                    </div>
                    
                    <!-- WSI Controls - Added annotation toggle switch -->
                    <div class="control-panel">
                        <h3 style="margin-top: 0;">Slide View Controls:</h3>
                        <div class="wsi-controls">
                            <div class="switch-label">
                                <span>Show Annotations:</span>
                                <label class="switch">
                                    <input type="checkbox" id="annotation-toggle" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                    
                    <div id="wsi-container">
                        <div id="openseadragon-viewer" class="wsi-viewer"></div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Display Regions Section -->
                <div class="image-container">
                    <div class="section-title">
                        <h2>Focus Regions</h2>
                    </div>
                    
                    <div class="control-panel">
                        <h3 style="margin-top: 0;">Annotation Controls:</h3>
                        <p><strong>Keyboard Shortcut:</strong> Press 'A' key to toggle between annotated and unannotated versions of all images</p>
                        <p><strong>Individual Images:</strong> Use the toggle button below each image to switch that specific image</p>
                        <p><strong>View Path:</strong> Click on any image to view and copy its full path</p>
                        <button id="toggle-all-btn" class="toggle-all-btn">Toggle Region Annotations</button>
                    </div>
                    
                    <div class="image-gallery">
                        {% for image in image_paths.regions.unannotated %}
                            {% if image in image_paths.regions.annotated %}
                            <div class="image-item" id="region-{{ loop.index }}">
                                <img src="/get_image/selected_focus_regions/high_mag_unannotated/{{ image }}" 
                                     data-unannotated="/get_image/selected_focus_regions/high_mag_unannotated/{{ image }}"
                                     data-annotated="/get_image/selected_focus_regions/high_mag_annotated/{{ image }}"
                                     data-state="unannotated"
                                     data-path="{{ result_folder }}/selected_focus_regions/high_mag_unannotated/{{ image }}"
                                     data-annotated-path="{{ result_folder }}/selected_focus_regions/high_mag_annotated/{{ image }}"
                                     class="preload-image region-image"
                                     onclick="showImagePath(this)">
                                <button onclick="toggleAnnotation('region-{{ loop.index }}')">Toggle Slide Annotation</button>
                            </div>
                            {% endif %}
                        {% endfor %}
                    </div>
                </div>
                
                <!-- Display Cells Section -->
                {% for cell_type, images in image_paths.cells.items() %}
                <div class="image-container">
                    <div class="section-title">
                        <h2>{{ cell_type }}</h2>
                    </div>
                    <div class="image-gallery">
                        {% for image in images %}
                        <div class="image-item cell-item">
                            <img src="/get_image/selected_cells/{{ cell_type.split(' - ')[0] }}/{{ image }}" 
                                 class="preload-image cell-image"
                                 data-path="{{ result_folder }}/selected_cells/{{ cell_type.split(' - ')[0] }}/{{ image }}"
                                 onclick="showImagePath(this)">
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Get all images that need to be preloaded
            const imagesToPreload = document.querySelectorAll('.preload-image');
            const totalImages = imagesToPreload.length;
            const progressBar = document.getElementById('progress-bar');
            const loadingOverlay = document.getElementById('loading-overlay');
            const mainContent = document.getElementById('main-content');
            
            let loadedImages = 0;
            
            // Initialize OpenSeadragon if whole slide image is available
            {% if has_wsi %}
            initializeViewer(true); // Initialize with annotations on by default
            {% endif %}
            
            // No images to preload case
            if (totalImages === 0) {
                progressBar.style.width = '100%';
                progressBar.textContent = '100%';
                
                // Hide loading overlay and show content after a short delay
                setTimeout(function() {
                    loadingOverlay.style.display = 'none';
                    mainContent.style.display = 'block';
                }, 500);
                return;
            }
            
            // Preload each image
            imagesToPreload.forEach(function(img) {
                // Create a new image object to preload
                const preloadImg = new Image();
                
                preloadImg.onload = function() {
                    loadedImages++;
                    
                    // Update progress bar
                    const percentComplete = Math.round((loadedImages / totalImages) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    
                    // If all images are loaded
                    if (loadedImages === totalImages) {
                        // Hide loading overlay and show content after a short delay
                        setTimeout(function() {
                            loadingOverlay.style.display = 'none';
                            mainContent.style.display = 'block';
                        }, 500);
                    }
                };
                
                preloadImg.onerror = function() {
                    loadedImages++;
                    console.error('Failed to load image:', img.src);
                    
                    // Update progress bar even on error
                    const percentComplete = Math.round((loadedImages / totalImages) * 100);
                    progressBar.style.width = percentComplete + '%';
                    progressBar.textContent = percentComplete + '%';
                    
                    // If all images are loaded (or failed)
                    if (loadedImages === totalImages) {
                        // Hide loading overlay and show content after a short delay
                        setTimeout(function() {
                            loadingOverlay.style.display = 'none';
                            mainContent.style.display = 'block';
                        }, 500);
                    }
                };
                
                // Start loading the image
                preloadImg.src = img.src;
                
                // For annotated images, also preload the annotated version
                if (img.hasAttribute('data-annotated')) {
                    const preloadAnnotated = new Image();
                    preloadAnnotated.src = img.getAttribute('data-annotated');
                }
            });
            
            // Set up modal close button
            document.querySelector('.close').addEventListener('click', function() {
                document.getElementById('path-modal').style.display = 'none';
            });
            
            // Close modal when clicking outside of it
            window.addEventListener('click', function(event) {
                const modal = document.getElementById('path-modal');
                if (event.target === modal) {
                    modal.style.display = 'none';
                }
            });
            
            // Set up copy button
            document.getElementById('copy-path-btn').addEventListener('click', function() {
                const pathText = document.getElementById('path-display').textContent;
                navigator.clipboard.writeText(pathText).then(function() {
                    const copyMessage = document.getElementById('copy-message');
                    copyMessage.style.display = 'block';
                    setTimeout(function() {
                        copyMessage.style.display = 'none';
                    }, 2000);
                });
            });
            
            // Toggle sidebar
            document.getElementById('sidebar-toggle').addEventListener('click', function() {
                const sidebar = document.getElementById('sidebar');
                const container = document.getElementById('content-container');
                const toggleButton = document.getElementById('sidebar-toggle');
                
                if (sidebar.classList.contains('sidebar-collapsed')) {
                    sidebar.classList.remove('sidebar-collapsed');
                    container.classList.remove('container-expanded');
                    toggleButton.textContent = '◀';
                } else {
                    sidebar.classList.add('sidebar-collapsed');
                    container.classList.add('container-expanded');
                    toggleButton.textContent = '▶';
                }
            });
            
            // Toggle All button functionality
            document.getElementById('toggle-all-btn').addEventListener('click', function() {
                const visibleImages = document.querySelectorAll('.image-item img[data-state]');
                
                // Check if all images are already annotated
                let allAnnotated = true;
                visibleImages.forEach(img => {
                    if (img.getAttribute('data-state') !== 'annotated') {
                        allAnnotated = false;
                    }
                });
                
                // If all are annotated, turn off all annotations
                // If some or none are annotated, turn on all annotations
                visibleImages.forEach(img => {
                    const itemId = img.closest('.image-item').id;
                    const currentState = img.getAttribute('data-state');
                    
                    if (allAnnotated) {
                        // Turn off annotations if all are annotated
                        if (currentState !== 'unannotated') {
                            toggleAnnotation(itemId);
                        }
                    } else {
                        // Turn on annotations if some are not annotated
                        if (currentState !== 'annotated') {
                            toggleAnnotation(itemId);
                        }
                    }
                });
            });
            
            // Add keyboard shortcut (A) to toggle annotation for the currently visible images
            document.addEventListener('keydown', function(event) {
                if (event.key.toLowerCase() === 'a') {
                    const visibleImages = document.querySelectorAll('.image-item img[data-state]');
                    visibleImages.forEach(img => {
                        const itemId = img.closest('.image-item').id;
                        toggleAnnotation(itemId);
                    });
                }
            });
            
            // Setup annotation toggle for WSI viewer
            {% if has_wsi %}
            const annotationToggle = document.getElementById('annotation-toggle');
            annotationToggle.addEventListener('change', function() {
                // Save current viewer position
                const viewerPosition = saveViewerPosition();
                
                // Reinitialize viewer with new annotation setting
                initializeViewer(this.checked, viewerPosition);
            });
            {% endif %}
        });

        // Variables to store viewer instance and current state
        let viewer;
        let viewerInitialized = false;

        // Function to save current viewer position
        function saveViewerPosition() {
            if (!viewer) return null;
            
            const viewportCenter = viewer.viewport.getCenter();
            const zoom = viewer.viewport.getZoom();
            
            return {
                x: viewportCenter.x,
                y: viewportCenter.y,
                zoom: zoom
            };
        }

        // Function to initialize OpenSeadragon viewer
        function initializeViewer(showAnnotations = true, restorePosition = null) {
            fetch('/get_dimensions')
                .then(response => response.json())
                .then(dimensions => {
                    const width = dimensions.width;
                    const height = dimensions.height;
                    
                    // If viewer exists, destroy it first
                    if (viewer) {
                        viewer.destroy();
                    }
                    
                    // Create the viewer with the appropriate tile source
                    viewer = OpenSeadragon({
                        id: "openseadragon-viewer",
                        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/2.4.2/images/",
                        tileSources: {
                            width: width,
                            height: height,
                            tileSize: 512,
                            maxLevel: 18,
                            getTileUrl: function(level, x, y) {
                                return showAnnotations ? 
                                    `/annotated_tile_api?level=${level}&x=${x}&y=${y}` : 
                                    `/tile_api?level=${level}&x=${x}&y=${y}`;
                            }
                        },
                        showNavigator: true,
                        navigatorPosition: "BOTTOM_RIGHT",
                        minZoomLevel: 0.5,
                        zoomPerScroll: 1.5,
                    });
                    
                    // Restore position if provided
                    if (restorePosition) {
                        viewer.addOnceHandler('open', function() {
                            viewer.viewport.panTo(new OpenSeadragon.Point(restorePosition.x, restorePosition.y), true);
                            viewer.viewport.zoomTo(restorePosition.zoom, null, true);
                        });
                    }
                    
                    viewerInitialized = true;
                })
                .catch(error => console.error("Error fetching slide dimensions:", error));
        }

        // Function to toggle between annotated and unannotated images
        function toggleAnnotation(itemId) {
            const imgElement = document.querySelector(`#${itemId} img`);
            const currentState = imgElement.getAttribute('data-state');
            
            if (currentState === 'unannotated') {
                imgElement.src = imgElement.getAttribute('data-annotated');
                imgElement.setAttribute('data-state', 'annotated');
            } else {
                imgElement.src = imgElement.getAttribute('data-unannotated');
                imgElement.setAttribute('data-state', 'unannotated');
            }
        }
        
        // Function to show image path in modal
        function showImagePath(imgElement) {
            event.stopPropagation(); // Prevent triggering other click events
            
            const modal = document.getElementById('path-modal');
            const pathDisplay = document.getElementById('path-display');
            
            // Get the appropriate path based on current state
            let path;
            if (imgElement.hasAttribute('data-state') && imgElement.getAttribute('data-state') === 'annotated') {
                path = imgElement.getAttribute('data-annotated-path');
            } else {
                path = imgElement.getAttribute('data-path');
            }
            
            // Set the path text and display the modal
            pathDisplay.textContent = path;
            modal.style.display = 'block';
        }
    </script>
</body>
</html>
""")

def main():
    parser = argparse.ArgumentParser(description='Flask app to display cell, region, and whole slide images.')
    parser.add_argument('result_folder', type=str, help='Path to the result folder')
    parser.add_argument('--dzi_path', type=str, help='Path to DZI folder for whole slide image viewing (optional)', default=None)
    args = parser.parse_args()
    
    global RESULT_FOLDER, DZI_PATH, FOCUS_REGIONS_DF
    RESULT_FOLDER = os.path.abspath(args.result_folder)
    
    if not os.path.exists(RESULT_FOLDER):
        print(f"Error: Result folder '{RESULT_FOLDER}' does not exist.")
        return
    
    print(f"Using result folder: {RESULT_FOLDER}")
    
    # Set DZI path if provided
    if args.dzi_path:
        DZI_PATH = os.path.abspath(args.dzi_path)
        if not os.path.exists(DZI_PATH):
            print(f"Warning: DZI path '{DZI_PATH}' does not exist.")
            DZI_PATH = None
        else:
            print(f"DZI path set to '{DZI_PATH}'")
            
            # Try to load focus regions for the slide
            try:
                print(f"Attempting to load focus regions for slide: {DZI_PATH}")
                print(f"From result folder: {RESULT_FOLDER}")
                FOCUS_REGIONS_DF = get_annotated_focus_region_indices_and_coordinates(RESULT_FOLDER)
                print(f"Loaded {len(FOCUS_REGIONS_DF)} focus regions")
                print(f"Focus regions columns: {list(FOCUS_REGIONS_DF.columns)}")
                print(f"First focus region: {FOCUS_REGIONS_DF.iloc[0].to_dict() if len(FOCUS_REGIONS_DF) > 0 else 'None'}")
            except Exception as e:
                print(f"Error loading focus regions: {e}")
                import traceback
                traceback.print_exc()
                FOCUS_REGIONS_DF = None
    else:
        print("No DZI path specified. Whole slide image viewer will not be available.")
    
    # Check for required subfolders
    required_dirs = [
        os.path.join(RESULT_FOLDER, 'selected_cells'),
        os.path.join(RESULT_FOLDER, 'selected_focus_regions')
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"Warning: Required directory '{directory}' does not exist.")
    
    # Create templates directory and files
    create_templates()
    
    # Define the route for serving images
    @app.route('/get_image/<path:image_path>')
    def get_image(image_path):
        directory, filename = os.path.split(image_path)
        return send_from_directory(os.path.join(RESULT_FOLDER, directory), filename)
    
    # Run the app
    print(f"Starting server... Navigate to http://127.0.0.1:5000/ in your browser")
    app.run(debug=True, host='127.0.0.1', port=5000)

if __name__ == '__main__':
    main()