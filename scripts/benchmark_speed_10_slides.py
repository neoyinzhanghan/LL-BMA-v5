import os
import time
import shutil
import pandas as pd
from tqdm import tqdm
from LLBMA.front_end.api import analyse_bma

slides_dir = "/media/hdd3/neo/error_slides_ndpi"
h5_dir = "/media/hdd3/neo/error_slides_h5"
dump_dir = "/media/hdd3/neo/test_error_results_dir_runtime_025_10_slides"

os.makedirs(dump_dir, exist_ok=True)

slide_paths = [
    os.path.join(slides_dir, file)
    for file in os.listdir(slides_dir)
    if file.endswith(".ndpi")
]

subdirs = [
    subdir
    for subdir in os.listdir(dump_dir)
    if os.path.isdir(os.path.join(dump_dir, subdir))
]

runtime_profile = {
    "slide_path": [],
    "runtime": [],
}

os.makedirs(dump_dir, exist_ok=True)

num_ran = 0

for slide_path in tqdm(slide_paths, desc="Processing slides"):

    if num_ran == 10:
        break

    slide_name = os.path.basename(slide_path)
    # remove the .ndpi extension
    slide_name = os.path.splitext(slide_name)[0]

    # if slide_name in subdirs:
    #     print(f"Skipping {slide_name} as it has already been processed")
    #     continue

    # if os.path.exists(dump_dir, slide_name) and not os.path.exists(
    #     dump_dir, slide_name, "focus_regions_debug_hoarding"
    # ):
    #     # deliete the directory
    #     shutil.rmtree(os.path.join(dump_dir, slide_name))

    # error_slide_name = "ERROR_" + slide_name
    # if os.path.exists(dump_dir, error_slide_name) and not os.path.exists(
    #     dump_dir, error_slide_name, "focus_regions_debug_hoarding"
    # ):
    #     # deliete the directory
    #     shutil.rmtree(os.path.join(dump_dir, error_slide_name))

    start_time = time.time()

    pretiled_h5_path = os.path.join(
        h5_dir, os.path.basename(slide_path).replace(".ndpi", ".h5")
    )

    # Run the heme_analyze function
    analyse_bma(
        slide_path=slide_path,
        dump_dir=dump_dir,
        hoarding=True,
        continue_on_error=True,
        do_extract_features=False,
        check_specimen_clf=False,
        pretiled_h5_path=None,
    )

    runtime_profile["slide_path"].append(slide_path)
    runtime_profile["runtime"].append(time.time() - start_time)

    num_ran += 1

runtime_df = pd.DataFrame(runtime_profile)
runtime_df.to_csv(os.path.join(dump_dir, "runtime_profile.csv"))

print(f"Process completed for {len(slide_paths)} slides, results saved to {dump_dir}")
print(f"Runtime data saved to {os.path.join(dump_dir, 'runtime_profile.csv')}")
print(f"Average runtime: {runtime_df['runtime'].mean():.2f} seconds per slide")
