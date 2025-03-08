import os
import numpy as np
import pandas as pd
from PIL import Image

def _add_yellow_boundary(pil_image):  # TODO this function is for debugging only
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
    # Remove the parentheses and split by commas
    return tuple(map(int, input_str.strip("()").split(", ")))


def get_LLBMA_processing_status(result_dir):
    if os.path.exists(result_dir):
        return "Processed"
    else:
        return "Not Processed"


def get_annotated_focus_region_indices_and_coordinates(result_dir):
    """Get a list of tuples (high_mag_score, idx, row, col, coordinate, image_path) for the annotated focus regions."""

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

    if (
        get_LLBMA_processing_status(result_dir) == "Error"
        or get_LLBMA_processing_status(result_dir) == "Not Processed"
    ):
        raise ValueError(
            f"Cannot get annotated focus regions for {result_dir}. Status: {get_LLBMA_processing_status(result_dir)}"
        )
    # get the high_mag_focus_regions_info.csv file from the selected_focus_regions subdir
    high_mag_focus_regions_info_path = os.path.join(
        result_dir, "selected_focus_regions", "high_mag_focus_regions_info.csv"
    )

    # read the high_mag_focus_regions_info.csv file
    high_mag_focus_regions_info_df = pd.read_csv(high_mag_focus_regions_info_path)

    for i, df_row in high_mag_focus_regions_info_df.iterrows():
        high_mag_score = df_row["adequate_confidence_score_high_mag"]

        # round the high_mag_score to 3 decimal places
        high_mag_score = round(high_mag_score, 3)
        idx = df_row["idx"]
        coordinate_string = df_row["coordinate"]
        coordinate = df_row["coordinate"]

        TLx, TLy, BRx, BRy = string_to_tuple(coordinate_string)

        row = TLx // 512
        col = TLy // 512

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
    tile_image, tile_row, tile_col, tile_level, focus_regions_df, debug_mode=False
):

    if tile_level <= 9:
        if debug_mode:
            tile_image = _add_yellow_boundary(tile_image)
        return tile_image

    elif tile_level < 15:

        found = False
        # iterate over the rows of the focus_regions_df
        for idx, df_row in focus_regions_df.iterrows():
            level_x, level_y = df_row[f"x_{tile_level}"], df_row[f"y_{tile_level}"]

            region_translation_x, region_translation_y = int(tile_row * 512), int(
                tile_col * 512
            )
            rel_level_x, rel_level_y = (
                int(level_x - region_translation_x),
                int(level_y - region_translation_y),
            )

            region_level_width, region_level_height = int(
                512 // 2 ** (18 - tile_level)
            ), int(512 // 2 ** (18 - tile_level))

            if 0 <= rel_level_x < 512 and 0 <= rel_level_y < 512:
                # set the corresponding pixels in the tile_image to red (should be a square of width  equal to region_level_width, and height equal to region_level_width)
                # with topleft corner at (rel_level_x, rel_level_y)
                tile_array = np.array(tile_image)
                tile_array[
                    rel_level_y : rel_level_y + region_level_height,
                    rel_level_x : rel_level_x + region_level_width,
                ] = [0, 255, 0]
                tile_image = Image.fromarray(tile_array)

                found = True

        if not found:
            if debug_mode:
                tile_image = _add_yellow_boundary(tile_image)
        if found:
            tile_image = _add_green_boundary(tile_image)
        return tile_image

    elif tile_level < 18:
        found = False

        # iterate over the rows of the focus_regions_df
        for idx, df_row in focus_regions_df.iterrows():
            level_x, level_y = df_row[f"x_{tile_level}"], df_row[f"y_{tile_level}"]

            region_translation_x, region_translation_y = int(tile_row * 512), int(
                tile_col * 512
            )
            rel_level_x, rel_level_y = (
                int(level_x - region_translation_x),
                int(level_y - region_translation_y),
            )

            region_level_width, region_level_height = int(
                512 // 2 ** (18 - tile_level)
            ), int(512 // 2 ** (18 - tile_level))

            if 0 <= rel_level_x < 512 and 0 <= rel_level_y < 512:
                # set the corresponding pixels in the tile_image to red (should be a square of width  equal to region_level_width, and height equal to region_level_width)
                # with topleft corner at (rel_level_x, rel_level_y)
                tile_array = np.array(tile_image)

                image_path = df_row["image_path"]

                # open the image
                image = Image.open(image_path)

                # resize the image to region_level_width x region_level_height
                image = image.resize((region_level_width, region_level_height))

                # convert the image to an array
                image_array = np.array(image)

                tile_array[
                    rel_level_y : rel_level_y + region_level_height,
                    rel_level_x : rel_level_x + region_level_width,
                ] = image_array

                tile_image = Image.fromarray(tile_array)

                found = True

        if not found:
            if debug_mode:
                tile_image = _add_yellow_boundary(tile_image)
        if found:
            tile_image = _add_green_boundary(tile_image)
        return tile_image

    else:
        found = False
        # iterate over the rows of the df
        for idx, df_row in focus_regions_df.iterrows():
            img_row, img_col = df_row["row"], df_row["col"]

            if tile_row == img_row and tile_col == img_col:
                image_path = df_row["image_path"]

                # open the image
                tile_image = Image.open(image_path)

                found = True

        if found:
            tile_image = _add_green_boundary(tile_image)
        else:
            if debug_mode:
                tile_image = _add_yellow_boundary(tile_image)
        return tile_image