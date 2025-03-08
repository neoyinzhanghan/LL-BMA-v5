import os
import shutil
import random
from tqdm import tqdm

result_dir = "/media/hdd3/neo/test_error_results_dir_050"
ERROR_save_dir = "/media/hdd3/neo/tmp_ERROR"
non_ERROR_save_dir = "/media/hdd3/neo/tmp_non_ERROR"

os.makedirs(ERROR_save_dir, exist_ok=True)
os.makedirs(non_ERROR_save_dir, exist_ok=True)

os.makedirs(os.path.join(ERROR_save_dir, "above_050_regions"), exist_ok=True)
os.makedirs(os.path.join(ERROR_save_dir, "below_050_regions"), exist_ok=True)
os.makedirs(os.path.join(non_ERROR_save_dir, "above_050_regions"), exist_ok=True)
os.makedirs(os.path.join(non_ERROR_save_dir, "below_050_regions"), exist_ok=True)

os.makedirs(os.path.join(ERROR_save_dir, "ERROR_grid_rep"), exist_ok=True)
os.makedirs(os.path.join(ERROR_save_dir, "non_ERROR_grid_rep"), exist_ok=True)
os.makedirs(os.path.join(non_ERROR_save_dir, "ERROR_grid_rep"), exist_ok=True)
os.makedirs(os.path.join(non_ERROR_save_dir, "non_ERROR_grid_rep"), exist_ok=True)

# get the list of subdirectories in the result directory
subdirs = [
    os.path.join(result_dir, subdir)
    for subdir in os.listdir(result_dir)
    if os.path.isdir(os.path.join(result_dir, subdir))
]

img_idx = 0

for subdir in tqdm(subdirs, desc="Processing Result Dirs"):
    # first get the list of all .jpeg files in the subdir/focus_region_debug_hoarding folder
    focus_regions_debug_hoarding_dir = os.path.join(
        subdir, "focus_regions_debug_hoarding"
    )

    # if not os.path.exists(focus_regions_debug_hoarding_dir):
    #     continue

    jpeg_paths = [
        os.path.join(focus_regions_debug_hoarding_dir, file)
        for file in os.listdir(focus_regions_debug_hoarding_dir)
        if file.endswith(".jpeg")
    ]

    num_good = 0

    for jpeg_path in jpeg_paths:
        jpeg_name = os.path.basename(jpeg_path)
        name_no_ext = os.path.splitext(jpeg_name)[0]

        score = int(name_no_ext)
        if score >= 500000:
            num_good += 1

    if num_good >= 20:
        is_ERROR = False
    else:
        is_ERROR = True

    for jpeg_path in tqdm(jpeg_paths, desc="Processing JPEGs"):
        # Apply the 5% probability for copying

        jpeg_name = os.path.basename(jpeg_path)
        name_no_ext = os.path.splitext(jpeg_name)[0]

        score = int(name_no_ext)
        if random.random() > 0.05 and score < 500000:
            continue
        elif random.random() > 0.50 and score >= 500000:
            continue

        if score >= 500000:
            if is_ERROR:
                save_dir = os.path.join(ERROR_save_dir, "above_050_regions")
            else:
                save_dir = os.path.join(non_ERROR_save_dir, "above_050_regions")
        else:
            if is_ERROR:
                save_dir = os.path.join(ERROR_save_dir, "below_050_regions")
            else:
                save_dir = os.path.join(non_ERROR_save_dir, "below_050_regions")

        shutil.copy(jpeg_path, os.path.join(save_dir, jpeg_name))
        img_idx += 1

    if is_ERROR:
        shutil.copy(
            os.path.join(subdir, "top_view_grid_rep.png"),
            os.path.join(ERROR_save_dir, "ERROR_grid_rep", f"{img_idx}.png"),
        )
    else:
        shutil.copy(
            os.path.join(subdir, "top_view_grid_rep.png"),
            os.path.join(non_ERROR_save_dir, "non_ERROR_grid_rep", f"{img_idx}.png"),
        )
