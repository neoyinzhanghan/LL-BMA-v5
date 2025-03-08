import os
import time
import shutil

slide_name = "H18-9786;S10;MSKM - 2023-06-21 21.41.10"
slide_source_dir = "/pesgisipth/NDPI/"

slide_path = os.path.join(slide_source_dir, slide_name + ".ndpi")

tmp_save_dir = "/media/hdd3/neo/brenda_tmp"

os.makedirs(tmp_save_dir, exist_ok=True)

starttime = time.time()
print(f"Copying slide {slide_name} to {tmp_save_dir}")
# rsync the slide to the tmp_save_dir
shutil.copy(slide_path, tmp_save_dir)
print(f"Slide {slide_name} copied to {tmp_save_dir}")
print(f"Time taken: {time.time() - starttime} seconds")
