import h5py
from LLBMA.tiling.dzsave_h5 import retrieve_tile_h5

dzsave_h5_path = "/media/hdd3/neo/test_error_results_dir_tmp/test_slide/slide.h5"

# open the h5 file
with h5py.File(dzsave_h5_path, "r") as f:

    print(f.keys())

    for key in f.keys():
        print(f"{key}: {f[key].shape}")

    # get the "0" dataset
    dataset = f["11"]

    # print the dimensions of the dataset
    print(dataset.shape)

    # print the the [0,0] element of the dataset
    # print(dataset[3, 1])

tile = retrieve_tile_h5(dzsave_h5_path, 11, 3, 1)

print(type(tile))  # should be a PIL.JpegImagePlugin.JpegImageFile
print(tile.size)  # should be (256, 256)F

# save the image to a file at "/media/hdd3/neo/test_error_results_dir_tmp/test_slide/test_image.jpg"
tile.save("/media/hdd3/neo/test_error_results_dir_tmp/test_slide/test_image.jpg")
