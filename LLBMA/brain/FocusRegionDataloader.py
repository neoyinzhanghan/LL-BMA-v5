import torch
import openslide
from PIL import Image
from LLBMA.vision.image_quality import VoL
from LLBMA.resources.BMAassumptions import (
    high_mag_region_clf_batch_size,
    num_focus_region_dataloader_workers,
    focus_regions_size,
)
from torchvision import transforms


class HighMagFocusRegionDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for high magnification focus regions.

    Attributes:
    - focus_regions (list): A list of focus region objects. Each focus region object must have an `image` attribute.
    """

    def __init__(self, focus_regions):
        """
        Initialize the dataset.

        Args:
        - focus_regions (list): A list of focus region objects with an `image` attribute.
        """
        self.focus_regions = focus_regions

    def __len__(self):
        """
        Return the number of focus regions.

        Returns:
        - int: The length of the dataset.
        """
        return len(self.focus_regions)

    def __getitem__(self, index):
        """
        Retrieve a focus region and its corresponding image tensor.

        Args:
        - index (int): The index of the focus region.

        Returns:
        - tuple: A tuple containing the focus region object and its image as a tensor.
        """
        focus_region = self.focus_regions[index]
        dzi_high_mag_image_path = focus_region.dzi_high_mag_image_path
        image = Image.open(dzi_high_mag_image_path)
        
        vol = VoL(image)
        focus_region.VoL_high_mag = vol
        # Ensure the image is valid before transforming
        if image is None:
            raise ValueError(f"Focus region at index {index} has no image.")

        # Apply ToTensor transformation
        tensor_image = transforms.ToTensor()(image)

        return focus_region, tensor_image


def custom_collate_function(batch):
    """A custom collate function that returns the focus regions and the images as a batch."""

    # print(f"The length of the batch is {len(batch)}")
    # print(
    #     f"The first item in the batch is {batch[0]}, of type {type(batch[0])} and length {len(batch[0])}"
    # )
    focus_regions = [item[0] for item in batch]
    images_tensors = [item[1] for item in batch]

    assert len(focus_regions) == len(images_tensors), "The number of focus regions and images tensors must be the same"

    filtered_focus_regions = []
    filtered_images_tensors = []

    for i in range(len(focus_regions)):
        # assert that the image tensor is a 3 channel image with size focus_regions_size x focus_regions_size
        assert images_tensors[i].shape[0] == 3, "The image tensor must be a 3 channel image"
        
        if images_tensors[i].shape[1] == focus_regions_size and images_tensors[i].shape[2] == focus_regions_size:
            filtered_focus_regions.append(focus_regions[i])
            filtered_images_tensors.append(images_tensors[i])
    
    if len(filtered_images_tensors) == 0:
        print("Warning: No valid images tensors found in the batch")
        # create a empty tensor of shape (0, 3, focus_regions_size, focus_regions_size)
        filtered_images_batch = torch.zeros(0, 3, focus_regions_size, focus_regions_size)
        return filtered_focus_regions, filtered_images_batch
    
    filtered_images_batch = torch.stack(filtered_images_tensors)

    return filtered_focus_regions, filtered_images_batch


def get_high_mag_focus_region_dataloader(
    focus_regions,
    batch_size=high_mag_region_clf_batch_size,
    num_workers=num_focus_region_dataloader_workers,
):
    """Return a dataloader of high magnification focus regions."""
    high_mag_focus_region_dataset = HighMagFocusRegionDataset(focus_regions)

    high_mag_focus_region_dataloader = torch.utils.data.DataLoader(
        high_mag_focus_region_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_function,
    )

    return high_mag_focus_region_dataloader


# def get_alternating_high_mag_focus_region_dataloader(
#     focus_regions,
#     wsi_path,
#     num_data_loaders=num_region_clf_managers,
#     batch_size=region_clf_batch_size,
#     num_workers=num_croppers,
# ):

#     list_of_lists_of_focus_regions = [[] for _ in range(num_data_loaders)]
#     for i, focus_region in enumerate(focus_regions):
#         list_idx = i % num_data_loaders
#         list_of_lists_of_focus_regions[list_idx].append(focus_region)

#     dataloaders = []

#     for focus_regions in list_of_lists_of_focus_regions:
#         dataloader = get_high_mag_focus_region_dataloader(
#             focus_regions, wsi_path, batch_size, num_workers
#         )
#         dataloaders.append(dataloader)

#     return dataloaders
