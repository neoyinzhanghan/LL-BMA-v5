import os
import pandas as pd

classes_to_remove = ["U1", "PL2", "PL3", "ER5", "ER6", "U4", "B1", "B2"]

def get_number_of_regions_and_cells(result_dir_path):

    focus_regions_save_subdir = os.path.join(
        result_dir_path, "focus_regions", "high_mag_unannotated"
    )
    cells_save_subdir = os.path.join(result_dir_path, "cells")

    cells_info_path = os.path.join(cells_save_subdir, "cells_info.csv")
    cells_info_df = pd.read_csv(cells_info_path)

    # num_focus_regions_passed is equal to the number of .jpg files in focus_regions_save_subdir
    num_focus_regions_passed = len(
        [
            name
            for name in os.listdir(focus_regions_save_subdir)
            if name.endswith(".jpg")
        ]
    )

    num_unannotated_focus_regions = 0

    for focus_region_name in os.listdir(focus_regions_save_subdir):
        if focus_region_name.endswith(".jpg"):
            idx = int(focus_region_name.split(".jpg")[0])

            if idx not in cells_info_df["focus_region_idx"].values:
                num_unannotated_focus_regions += 1

    # find all the further subdirectories in cells_save_subdir
    cell_subdirs = [
        subdir
        for subdir in os.listdir(cells_save_subdir)
        if os.path.isdir(os.path.join(cells_save_subdir, subdir))
    ]

    # remove the cell_subdirs that are in classes_to_remove
    cell_subdirs = [
        subdir for subdir in cell_subdirs if subdir not in classes_to_remove
    ]

    # num_cells_passed is equal to the number of .jpg files in the cell_subdirs
    num_cells_passed = 0

    for cell_subdir in cell_subdirs:
        if os.path.exists(os.path.join(cells_save_subdir, cell_subdir)):
            num_cells_passed += len(
                [
                    name
                    for name in os.listdir(os.path.join(cells_save_subdir, cell_subdir))
                    if name.endswith(".jpg")
                ]
            )

    num_removed_cells = 0

    for cell_subdir in classes_to_remove:
        if os.path.exists(os.path.join(cells_save_subdir, cell_subdir)):
            num_removed_cells += len(
                [
                    name
                    for name in os.listdir(os.path.join(cells_save_subdir, cell_subdir))
                    if name.endswith(".jpg")
                ]
            )

    return (
        num_focus_regions_passed,
        num_unannotated_focus_regions,
        num_cells_passed,
        num_removed_cells,
    )
