import os
from tqdm import tqdm

result_dir = "/media/hdd3/neo/test_error_results_dir_025"

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

num_errors = 0
num_non_errors = 0

# get the list of all .jpeg files in the subdir/focus_region_debug_hoarding folders
for subdir in tqdm(subdirs, desc="Gathering jpegs from Result Dirs"):
    focus_regions_debug_hoarding_dir = os.path.join(
        subdir, "focus_regions_debug_hoarding"
    )

    jpeg_paths = [
        os.path.join(focus_regions_debug_hoarding_dir, file)
        for file in os.listdir(focus_regions_debug_hoarding_dir)
        if file.endswith(".jpeg")
    ]

    num_good = 0

    for jpeg_path in jpeg_paths:
        jpeg_name = os.path.basename(jpeg_path)
        name_no_ext = os.path.splitext(jpeg_name)[0]

        score = int(name_no_ext.split("_")[0])
        if score >= 250000:
            num_good += 1

    if num_good >= 20:
        num_non_errors += 1
    else:
        num_errors += 1

print(f"Number of errors: {num_errors}")
print(f"Number of non-errors: {num_non_errors}")
