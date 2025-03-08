####################################################################################################
# Imports ##########################################################################################
####################################################################################################

# Outside imports ##################################################################################`
import openslide
import ray

# Within package imports ###########################################################################
from LLBMA.vision.image_quality import VoL
from LLBMA.BMAFocusRegion import FocusRegion
from LLBMA.resources.BMAassumptions import (
    search_view_level,
    search_view_downsample_rate,
    search_view_focus_regions_size,
    snap_shot_size,
)
from LLBMA.tiling.h5_reader_instance_class import h5_reader
from PIL import Image, ImageOps


# @ray.remote(num_cpus=num_cpus_per_cropper)
@ray.remote
class WSICropManager:
    """A class representing a manager that crops WSIs.

    === Class Attributes ===
    - wsi_path : the path to the WSI
    - wsi : the WSI
    """

    def __init__(self, wsi_path) -> None:
        self.wsi_path = wsi_path
        self.wsi = None

    def open_slide(self):
        """Open the WSI."""

        self.wsi = openslide.OpenSlide(self.wsi_path)

    def close_slide(self):
        """Close the WSI."""

        self.wsi.close()

        self.wsi = None

    def crop(self, coords, level=0, downsample_rate=1):
        """Crop the WSI at the lowest level of magnification."""

        if self.wsi is None:
            self.open_slide()

        level_0_coords = (
            coords[0] * downsample_rate,
            coords[1] * downsample_rate,
            coords[2] * downsample_rate,
            coords[3] * downsample_rate,
        )

        image = self.wsi.read_region(
            level_0_coords, level, (coords[2] - coords[0], coords[3] - coords[1])
        )

        image = image.convert("RGB")

        return image

    def async_get_bma_focus_region_batch(self, focus_region_coords):
        """Return a list of focus regions."""

        focus_regions = []
        for focus_region_coord in focus_region_coords:

            image = self.crop(
                focus_region_coord,
                level=search_view_level,
                downsample_rate=search_view_downsample_rate,
            )

            focus_region = FocusRegion(
                downsampled_coordinate=focus_region_coord, downsampled_image=image
            )
            focus_regions.append(focus_region)

        for focus_region in focus_regions:
            wsi = openslide.OpenSlide(self.wsi_path)

            pad_size = snap_shot_size // 2

            padded_coordinate = (
                focus_region.resampled_coordinate[0] - pad_size,
                focus_region.resampled_coordinate[1] - pad_size,
                focus_region.resampled_coordinate[2] + pad_size,
                focus_region.resampled_coordinate[3] + pad_size,
            )
            padded_image = wsi.read_region(
                padded_coordinate,
                0,
                (
                    focus_region.resampled_coordinate[2]
                    - focus_region.resampled_coordinate[0]
                    + pad_size * 2,
                    focus_region.resampled_coordinate[3]
                    - focus_region.resampled_coordinate[1]
                    + pad_size * 2,
                ),
            )

            original_width = focus_region.resampled_coordinate[2] - focus_region.resampled_coordinate[0]
            original_height = focus_region.resampled_coordinate[3] - focus_region.resampled_coordinate[1]

            unpadded_image = padded_image.crop(
                (
                    pad_size,
                    pad_size,
                    pad_size + original_width,
                    pad_size + original_height,
                )
            )

            focus_region.get_image(unpadded_image, padded_image)

        return focus_regions


# @ray.remote(num_cpus=num_cpus_per_cropper)
@ray.remote
class WSIH5FocusRegionCreationManager:
    """A class representing a manager that create focus regions from h5 dzsaved source.

    === Class Attributes ===
    - h5_path : the path to the h5 file
    - h5_reader : the h5 reader
    """

    def __init__(self, h5_path) -> None:
        self.h5_path = h5_path
        self.h5_reader = h5_reader(h5_path)
        self.h5_reader.open()

    def async_get_bma_focus_region_batch(self, focus_region_coords):
        """Return a list of focus regions."""

        focus_regions = []
        for focus_region_coord in focus_region_coords:

            image = self.h5_reader.read_region_level_0(
                TL_x=focus_region_coord[0],
                TL_y=focus_region_coord[1],
            )

            # downsampling the image by a factor of 2 ** search_view_level to match the search view level focus region size
            downsampled_image = self.h5_reader.read_region_search_view_level(
                TL_x=focus_region_coord[0],
                TL_y=focus_region_coord[1],
            )

            # downsampled_image = image.resize(
            #     (
            #         search_view_focus_regions_size,
            #         search_view_focus_regions_size,
            #     )
            # )

            focus_region = FocusRegion(
                downsampled_coordinate=focus_region_coord,
                downsampled_image=downsampled_image,
            )

            focus_region.get_image(image)

            focus_regions.append(focus_region)

        return focus_regions


# class WSIH5FocusRegionCreationManagerSerial:
#     """A class representing a manager that create focus regions from h5 dzsaved source.

#     === Class Attributes ===
#     - h5_path : the path to the h5 file
#     - h5_reader : the h5 reader
#     """

#     def __init__(self, h5_path) -> None:
#         self.h5_path = h5_path
#         self.h5_reader = h5_reader(h5_path)
#         self.h5_reader.open()

#     def get_bma_focus_region(self, focus_region_coord):
#         """Return a list of focus regions."""

#         image = self.h5_reader.read_region_level_0(
#             TL_x=focus_region_coord[0],
#             TL_y=focus_region_coord[1],
#         )

#         # Define the padding size (half of `snap_shot_size`)
#         padding_size = snap_shot_size // 2

#         # Add padding around the image with black pixels
#         # padded_image = ImageOps.expand(image, border=padding_size, fill="black")
#         padded_image = image

#         # downsampling the image by a factor of 2 ** search_view_level to match the search view level focus region size
#         # downsampled_image = self.h5_reader.read_region_search_view_level(
#         #     TL_x=focus_region_coord[0],
#         #     TL_y=focus_region_coord[1],
#         # )

#         downsampled_image = image.resize(
#             (
#                 search_view_focus_regions_size,
#                 search_view_focus_regions_size,
#             )
#         )
#         focus_region = FocusRegion(
#             downsampled_coordinate=focus_region_coord,
#             downsampled_image=downsampled_image,
#         )

#         focus_region.get_image(image, padded_image)

#         return focus_region


# @ray.remote
# class WSIH5FocusRegionCreationManager:
#     """A class representing a manager that creates focus regions from h5 dzsaved source."""

#     def __init__(self, h5_path) -> None:
#         self.h5_path = h5_path
#         self.h5_reader = h5_reader.remote(h5_path)
#         self.h5_reader.open.remote()

#     def async_get_bma_focus_region_batch(self, focus_region_coords):
#         """Return a list of focus regions."""
#         focus_regions = []
#         for focus_region_coord in focus_region_coords:

#             # Get the region level 0 image
#             image_ref = self.h5_reader.read_region_level_0.remote(
#                 TL_x=focus_region_coord[0],
#                 TL_y=focus_region_coord[1],
#             )
#             image = ray.get(image_ref)  # Fetch the result

#             # Define the padding size (half of `snap_shot_size`)
#             padding_size = snap_shot_size // 2

#             # Add padding around the image with black pixels
#             padded_image = ImageOps.expand(image, border=padding_size, fill="black")

#             # Get the downsampled image
#             downsampled_image_ref = self.h5_reader.read_region_search_view_level.remote(
#                 TL_x=focus_region_coord[0],
#                 TL_y=focus_region_coord[1],
#             )
#             downsampled_image = ray.get(downsampled_image_ref)  # Fetch the result

#             focus_region = FocusRegion(
#                 downsampled_coordinate=focus_region_coord,
#                 downsampled_image=downsampled_image,
#             )

#             focus_region.get_image(image, padded_image)

#             focus_regions.append(focus_region)

#         return focus_regions
