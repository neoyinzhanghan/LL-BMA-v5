from PIL import Image
import cv2
import numpy as np

from LLBMA.vision.bma_particle_detection import (
    get_top_view_preselection_mask,
    get_grid_rep,
)

image_path = "/media/hdd2/neo/test_v5/fd63ec0b-c7cc-469c-bf5f-19608a35f42b/top_view_image.png"
image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode

# Convert PIL image to cv2 format as required by get_top_view_preselection_mask
image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

mask, overlayed_image, final_blue_mask = get_top_view_preselection_mask(image_cv2, verbose=True)