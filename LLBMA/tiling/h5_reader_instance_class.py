import io
import ray
import h5py
import base64
import numpy as np
from PIL import Image
from LLBMA.resources.BMAassumptions import (
    focus_regions_size,
    max_dzsave_level,
    search_view_level,
)


def image_to_jpeg_string(image):
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()
    try:
        # Save the image in JPEG format to the buffer
        image.save(buffer, format="JPEG")
        jpeg_string = buffer.getvalue()  # Get the byte data
    finally:
        buffer.close()  # Explicitly close the buffer to free memory

    return jpeg_string


def jpeg_string_to_image(jpeg_string):
    # Create an in-memory bytes buffer from the byte string
    buffer = io.BytesIO(jpeg_string)

    # Open the image from the buffer and keep the buffer open
    image = Image.open(buffer)

    # Load the image data into memory so that it doesn't depend on the buffer anymore
    image.load()

    return image


def encode_image_to_base64(jpeg_string):
    return base64.b64encode(jpeg_string)


def decode_image_from_base64(encoded_string):
    return base64.b64decode(encoded_string)


def retrieve_tile_h5(h5_path, level, row, col):
    with h5py.File(h5_path, "r") as f:
        try:
            jpeg_string = f[str(level)][row, col]
            jpeg_string = decode_image_from_base64(jpeg_string)
            image = jpeg_string_to_image(jpeg_string)

        except Exception as e:
            print(
                f"Error: {e} occurred while retrieving tile at level: {level}, row: {row}, col: {col} from {h5_path}"
            )
            jpeg_string = f[str(level)][row, col]
            print(f"jpeg_string: {jpeg_string}")
            jpeg_string = decode_image_from_base64(jpeg_string)
            print(f"jpeg_string base 64 decoded: {jpeg_string}")
            raise e
        return image


# class h5_reader:
#     """
#     A singleton-like class for reading tiles from an H5 file.
#     Ensures only one instance exists per unique h5_path.

#     === Attributes ===
#     - h5_path : the path to the H5 file
#     - f : the H5 file object, None if not opened
#     """

#     _instances = {}

#     def __new__(cls, h5_path):
#         # Check if an instance with the same h5_path already exists
#         if h5_path in cls._instances:
#             return cls._instances[h5_path]
#         # If not, create a new instance and store it in the dictionary
#         instance = super(h5_reader, cls).__new__(cls)
#         cls._instances[h5_path] = instance
#         return instance

#     def __init__(self, h5_path):
#         self.h5_path = h5_path
#         self.f = None

#     def open(self):
#         """Open the H5 file."""
#         if self.f is None:
#             self.f = h5py.File(self.h5_path, "r")

#     def close(self):
#         """Close the H5 file if it's open."""
#         if self.f is not None:
#             self.f.close()
#             self.f = None

#     def get_level_0_width(self):
#         return self.f["level_0_width"][0]

#     def get_level_0_height(self):
#         return self.f["level_0_height"][0]

#     def get_max_level(self):
#         return self.f["num_levels"][0] - 1

#     def get_patch_size(self):
#         return self.f["patch_size"][0]

#     def get_overlap(self):
#         return self.f["overlap"][0]

#     def retrieve_tile(self, level, row, col):
#         """
#         Retrieve a tile at the specified level, row, and column.

#         Parameters:
#             level (int): The zoom level of the tile.
#             row (int): The row index of the tile.
#             col (int): The column index of the tile.

#         Returns:
#             The tile image as an object.
#         """
#         if self.f is None:
#             raise RuntimeError(
#                 "H5 file is not open. Please call open() before retrieving tiles."
#             )

#         try:
#             jpeg_string = self.f[str(level)][row, col]
#             jpeg_string = decode_image_from_base64(jpeg_string)
#             image = jpeg_string_to_image(jpeg_string)
#         except Exception as e:
#             print(
#                 f"Error: {e} occurred while retrieving tile at level: {level}, row: {row}, col: {col} from {self.h5_path}"
#             )
#             jpeg_string = self.f[str(level)][row, col]
#             print(f"jpeg_string: {jpeg_string}")
#             jpeg_string = decode_image_from_base64(jpeg_string)
#             print(f"jpeg_string base 64 decoded: {jpeg_string}")
#             raise e

#         return image

#     def read_region_level_0(self, TL_x, TL_y):
#         """
#         Retrieve a region of tiles at the specified level and top left corner.

#         Parameters:
#             level (int): The zoom level of the tile.
#             TL_x (int): The x coordinate of the top left corner of the region.
#             TL_y (int): The y coordinate of the top left corner of the region.

#         Returns:
#             The region of tiles as a numpy array.
#         """

#         if self.f is None:
#             raise RuntimeError(
#                 "H5 file is not open. Please call open() before retrieving tiles."
#             )

#         row, col = TL_x // focus_regions_size, TL_y // focus_regions_size
#         level = max_dzsave_level
#         image = self.retrieve_tile(level, row, col)

#         return image

#     def read_region_search_view_level(self, TL_x, TL_y):
#         """
#         Retrieve a region of tiles at the search view level and top left corner.

#         Parameters:
#             level (int): The zoom level of the tile.
#             TL_x (int): The x coordinate of the top left corner of the region at level 0.
#             TL_y (int): The y coordinate of the top left corner of the region at level 0.

#         Returns:
#             The region of tiles as a numpy array.
#         """

#         if self.f is None:
#             raise RuntimeError(
#                 "H5 file is not open. Please call open() before retrieving tiles."
#             )

#         row, col = TL_x // (focus_regions_size * (2**search_view_level)), TL_y // (
#             focus_regions_size * (2**search_view_level)
#         )

#         level = max_dzsave_level - search_view_level
#         image = self.retrieve_tile(level, row, col)

#         search_view_focus_region_size = focus_regions_size // (2**search_view_level)

#         remainder_x = (
#             TL_x % (focus_regions_size * (2**search_view_level))
#         ) // 2**search_view_level
#         remainder_y = (
#             TL_y % (focus_regions_size * (2**search_view_level))
#         ) // 2**search_view_level

#         offset_x = remainder_x // search_view_focus_region_size
#         offset_y = remainder_y // search_view_focus_region_size

#         downsampled_image = image.crop(
#             (
#                 offset_x * search_view_focus_region_size,
#                 offset_y * search_view_focus_region_size,
#                 (offset_x + 1) * search_view_focus_region_size,
#                 (offset_y + 1) * search_view_focus_region_size,
#             )
#         )

#         return downsampled_image


# @ray.remote
class h5_reader:
    """
    A class for reading tiles from an H5 file.

    === Attributes ===
    - h5_path : the path to the H5 file
    - f : the H5 file object, None if not opened
    """

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.f = None

    def open(self):
        """Open the H5 file."""
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r", swmr=True)

    def close(self):
        """Close the H5 file if it's open."""
        if self.f is not None:
            self.f.close()
            self.f = None

    def get_level_0_width(self):
        return self.f["level_0_width"][0]

    def get_level_0_height(self):
        return self.f["level_0_height"][0]

    def get_max_level(self):
        return self.f["num_levels"][0] - 1

    def get_patch_size(self):
        return self.f["patch_size"][0]

    def get_overlap(self):
        return self.f["overlap"][0]

    def retrieve_tile(self, level, row, col):
        """
        Retrieve a tile at the specified level, row, and column.

        Parameters:
            level (int): The zoom level of the tile.
            row (int): The row index of the tile.
            col (int): The column index of the tile.

        Returns:
            The tile image as an object.
        """
        if self.f is None:
            raise RuntimeError(
                "H5 file is not open. Please call open() before retrieving tiles."
            )

        try:
            jpeg_string = self.f[str(level)][row, col]
            jpeg_string = decode_image_from_base64(jpeg_string)
            image = jpeg_string_to_image(jpeg_string)
        except Exception as e:
            print(
                f"Error: {e} occurred while retrieving tile at level: {level}, row: {row}, col: {col} from {self.h5_path}"
            )
            jpeg_string = self.f[str(level)][row, col]
            print(f"jpeg_string: {jpeg_string}")
            jpeg_string = decode_image_from_base64(jpeg_string)
            print(f"jpeg_string base 64 decoded: {jpeg_string}")
            raise e

        return image

    def read_region_level_0(self, TL_x, TL_y):
        """
        Retrieve a region of tiles at the specified level and top left corner.

        Parameters:
            level (int): The zoom level of the tile.
            TL_x (int): The x coordinate of the top left corner of the region.
            TL_y (int): The y coordinate of the top left corner of the region.

        Returns:
            The region of tiles as a numpy array.
        """

        if self.f is None:
            raise RuntimeError(
                "H5 file is not open. Please call open() before retrieving tiles."
            )

        row, col = TL_x // focus_regions_size, TL_y // focus_regions_size
        level = max_dzsave_level
        image = self.retrieve_tile(level, row, col)

        return image

    def read_region_search_view_level(self, TL_x, TL_y):
        """
        Retrieve a region of tiles at the search view level and top left corner.

        Parameters:
            level (int): The zoom level of the tile.
            TL_x (int): The x coordinate of the top left corner of the region at level 0.
            TL_y (int): The y coordinate of the top left corner of the region at level 0.

        Returns:
            The region of tiles as a numpy array.
        """

        if self.f is None:
            raise RuntimeError(
                "H5 file is not open. Please call open() before retrieving tiles."
            )

        row, col = TL_x // (focus_regions_size * (2**search_view_level)), TL_y // (
            focus_regions_size * (2**search_view_level)
        )

        level = max_dzsave_level - search_view_level
        image = self.retrieve_tile(level, row, col)

        search_view_focus_region_size = focus_regions_size // (2**search_view_level)

        remainder_x = (
            TL_x % (focus_regions_size * (2**search_view_level))
        ) // 2**search_view_level
        remainder_y = (
            TL_y % (focus_regions_size * (2**search_view_level))
        ) // 2**search_view_level

        offset_x = remainder_x // search_view_focus_region_size
        offset_y = remainder_y // search_view_focus_region_size

        downsampled_image = image.crop(
            (
                offset_x * search_view_focus_region_size,
                offset_y * search_view_focus_region_size,
                (offset_x + 1) * search_view_focus_region_size,
                (offset_y + 1) * search_view_focus_region_size,
            )
        )

        return downsampled_image


if __name__ == "__main__":
    h5_path = (
        "/home/greg/Documents/neo/cp-lab-wsi-upload/wsi-and-heatmaps/bma_test_slide.h5"
    )
    h5 = h5_reader(h5_path)

    save_dir = "/media/hdd3/neo"

    h5.open()

    # randomly select a level_0 tile coordinate
    level_0_width = h5.get_level_0_width()
    level_0_height = h5.get_level_0_height()

    TL_x = (
        np.random.randint(0, level_0_width // focus_regions_size) * focus_regions_size
    )

    TL_y = (
        np.random.randint(0, level_0_height // focus_regions_size) * focus_regions_size
    )

    image = h5.read_region_level_0(TL_x, TL_y)

    downsampled_image = h5.read_region_search_view_level(TL_x, TL_y)

    # save the image to disk
    image.save(f"{save_dir}/LLBMA_test_FR_level_0_tile.jpg")
    downsampled_image.save(f"{save_dir}/LLBMA_test_FR_search_view_tile.jpg")
