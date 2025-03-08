import os
import time
import shutil
from LLBMA.front_end.api import analyse_bma

# slide_path = "/media/hdd3/neo/cornell_sample_bmas/fd63ec0b-c7cc-469c-bf5f-19608a35f42b.svs" # here is an MSK slide
# slide_path = "/media/hdd3/neo/cornell_sample_bmas/95de90f6-2250-4c5f-98b0-2d301aff4ecf.svs" # here is an MSK slide
slide_path = "/media/hdd3/neo/cornell_sample_bmas/4348ce96-9d80-4f4e-aa11-5ba494410326.svs" # here is an MSK slide
# slide_path = "/home/neo/Documents/neo/neo_is_slide_tiling_god/ANONJ4JJKJ17J_1_2.dcm" # here is an example of the dcm slide from Martin from Norway
# slide_path = "/media/hdd3/neo/Immunologic_Molecular_Analysis_Of_Lymphoid_Neoplasia/I24-2605/0072e87f-3274-47d1-ab71-bc73dea8a9e0.svs" # here is example slide from Cornell
dump_dir = "/media/hdd2/neo/test_v5"
tiling_dump_dir = "/media/hdd2/neo/test_v5_dzi"
tiling_format = "dzi"

# if os.path.exists(os.path.join(dump_dir, "test")):
#     shutil.rmtree(os.path.join(dump_dir, "test"))
# if os.path.exists(os.path.join(tiling_dump_dir, "test")):
#     shutil.rmtree(os.path.join(tiling_dump_dir, "test"))

if __name__ == "__main__":

    # # if the dump directory does not exist, create it
    # if not os.path.exists(dump_dir):
    #     os.makedirs(dump_dir)
    # else:
    #     # remove the dump directory if it already exists and recreate it
    #     os.system(f"rm -rf {dump_dir}")
    #     os.makedirs(dump_dir)

    start_time = time.time()

    # Run the heme_analyze function
    analyse_bma(
        slide_path=slide_path,
        dump_dir=dump_dir,
        hoarding=True,
        extra_hoarding=False,
        continue_on_error=False,
        do_extract_features=False,
        check_specimen_clf=False,
        tiling_dump_dir=tiling_dump_dir,
        tiling_format=tiling_format,
    )

    print(f"Time taken: {time.time() - start_time:.2f} seconds to process {slide_path}")
