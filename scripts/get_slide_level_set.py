import os
from tqdm import tqdm

result_dir = "/media/hdd3/neo/test_error_results_dir_025"
save_dir = "/media/hdd3/neo/RC_level_sets_H17-4780;S10;MSKK - 2023-09-11 09.17.56"

levels = [
    0,
    100000,
    150000,
    200000,
    250000,
    300000,
    350000,
    400000,
    450000,
    500000,
    550000,
    600000,
    650000,
    700000,
    750000,
    800000,
    850000,
    900000,
    950000,
    1000000,
]

stride = 50000

# if the save_dir already exists, delete it
if os.path.exists(save_dir):
    os.system(f"rm -rf {save_dir}")

os.makedirs(save_dir, exist_ok=True)
for level in levels:
    os.makedirs(os.path.join(save_dir, str(level)), exist_ok=True)


def find_level(jpeg_path):
    jpeg_name = os.path.basename(jpeg_path)
    name_no_ext = os.path.splitext(jpeg_name)[0]

    score = int(name_no_ext.split("_")[0])  # get the score from the jpeg name
    for i in range(len(levels)):
        if i == 0:
            continue
        elif score < levels[i]:
            return levels[i - 1]

    print(f"Score {score} is greater than 1000000")
    print(f"JPEG file at {jpeg_path} is faulty")
    import sys

    sys.exit(1)


# first get the list of all subdirs in the result directory
subdirs = [
    os.path.join(result_dir, subdir)
    for subdir in os.listdir(result_dir)
    if os.path.isdir(os.path.join(result_dir, subdir))
]

# only keeps the subdirs that have the focus_regions_debug_hoarding folder
subdirs = [
    subdir
    for subdir in subdirs
    if os.path.exists(os.path.join(subdir, "focus_regions_debug_hoarding"))
]

jpeg_paths = []
# get the list of all .jpeg files in the subdir/focus_region_debug_hoarding folders
for subdir in tqdm(subdirs, desc="Gathering jpegs from Result Dirs"):

    if "H17-4780;S10;MSKK - 2023-09-11 09.17.56" not in subdir:
        continue

    focus_regions_debug_hoarding_dir = os.path.join(
        subdir, "focus_regions_debug_hoarding"
    )

    jpeg_paths.extend(
        [
            os.path.join(focus_regions_debug_hoarding_dir, file)
            for file in os.listdir(focus_regions_debug_hoarding_dir)
            if file.endswith(".jpeg")
        ]
    )

pseudo_idx = 0

for jpeg_path in tqdm(jpeg_paths, desc="Processing JPEGs"):
    level = find_level(jpeg_path)
    level_dir = os.path.join(save_dir, str(level))

    new_name = f"{pseudo_idx}.jpeg"
    new_path = os.path.join(level_dir, new_name)

    # create a symlink to the jpeg in the appropriate level directory
    os.symlink(jpeg_path, new_path)

    pseudo_idx += 1
