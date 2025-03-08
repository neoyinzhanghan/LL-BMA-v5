import os
import openslide
import pandas as pd
from tqdm import tqdm

slide_path = "/media/hdd3/neo/error_slides_ndpi/test_slide.ndpi"
cells_info_csv = (
    "/media/hdd3/neo/test_error_results_dir_tmp/test_slide/cells/cells_info.csv"
)
save_dir = "/media/hdd3/neo/test_error_results_dir_tmp/recrop_tmp"

# open the cells_info.csv file and randomly subset 1000 rows
cells_info = pd.read_csv(cells_info_csv)
cells_info_subset = cells_info.sample(1000)

for i, row in tqdm(cells_info_subset.iterrows(), total=cells_info_subset.shape[0]):
    TL_x, TL_y, BR_x, BR_y = row["TL_x"], row["TL_y"], row["BR_x"], row["BR_y"]
    slide = openslide.OpenSlide(slide_path)

    # crop the region from the slide
    cell = slide.read_region((TL_x, TL_y), 0, (BR_x - TL_x, BR_y - TL_y))

    # if RGBA, convert to RGB
    if cell.mode == "RGBA":
        cell = cell.convert("RGB")

    # save the cell
    cell.save(os.path.join(save_dir, f"cell_{i}.png"))
