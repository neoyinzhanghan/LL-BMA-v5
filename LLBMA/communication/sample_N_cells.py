import os
import shutil
import pandas as pd
from tqdm import tqdm

classes_to_remove = ["U1", "PL2", "PL3", "ER5", "ER6", "U4", "B1", "B2"]


def sample_N_cells(
    result_dir_path, N=200, verbose=False
):

    high_mag_region_result_path = os.path.join(
        result_dir_path, "focus_regions", "high_mag_focus_regions_info.csv"
    )

    low_mag_region_result_path = os.path.join(
        result_dir_path, "focus_regions", "focus_regions_info.csv"
    )

    cell_info_path = os.path.join(result_dir_path, "cells", "cells_info.csv")

    cell_save_subdir = "selected_cells"
    focus_regions_save_subdir = "selected_focus_regions"

    if os.path.exists(os.path.join(result_dir_path, cell_save_subdir)):
        # delete result_dir_path/cell_save_subdir
        shutil.rmtree(os.path.join(result_dir_path, cell_save_subdir))

    if os.path.exists(os.path.join(result_dir_path, focus_regions_save_subdir)):
        # delete result_dir_path/focus_regions_save_subdir
        shutil.rmtree(os.path.join(result_dir_path, focus_regions_save_subdir))

    os.makedirs(os.path.join(result_dir_path, cell_save_subdir), exist_ok=True)
    os.makedirs(os.path.join(result_dir_path, focus_regions_save_subdir), exist_ok=True)
    os.makedirs(
        os.path.join(result_dir_path, focus_regions_save_subdir, "high_mag_annotated"),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            result_dir_path, focus_regions_save_subdir, "high_mag_unannotated"
        ),
        exist_ok=True,
    )

    # open all these paths as dataframes
    high_mag_region_result_df = pd.read_csv(high_mag_region_result_path)
    low_mag_region_result_df = pd.read_csv(low_mag_region_result_path)
    cell_info_df = pd.read_csv(cell_info_path)

    if verbose:
        # print the columns for all these dataframes
        print("High Mag Region Result Columns:")
        print(high_mag_region_result_df.columns)

        print("\nLow Mag Region Result Columns:")
        print(low_mag_region_result_df.columns)

        print("\nCell Info Columns:")
        print(cell_info_df.columns)

    # sort the high_mag_region_result_df by the confidence score in descending order
    high_mag_region_result_df = high_mag_region_result_df.sort_values(
        by="adequate_confidence_score_high_mag", ascending=False
    )

    regions_to_keep = []
    num_cells = 0
    num_cells_pre_removal = 0

    # iterate through the rows of the high_mag_region_result_df
    for idx, row in high_mag_region_result_df.iterrows():
        if verbose: 
            print(f"High mag score: {row["adequate_confidence_score_high_mag"]}")
        focus_region_idx = row["idx"]

        # look for all the rows in the the cells df that have the focus_region_idx column equal to the focus_region_idx
        cell_info_df_filtered = cell_info_df.loc[
            cell_info_df["focus_region_idx"] == focus_region_idx
        ]

        before_removal = cell_info_df_filtered.shape[0]

        if verbose:
            # print the number of cells in this focus region
            print(
                f"Number of cells in focus region {focus_region_idx}: {cell_info_df_filtered.shape[0]}"
            )

        # remove all the rows where label is in classes_to_remove
        cell_info_df_filtered = cell_info_df_filtered.loc[
            ~cell_info_df_filtered["label"].isin(classes_to_remove)
        ]

        if verbose:
            print(
                f"Number of cells in focus region {focus_region_idx} after removing classes to remove: {cell_info_df_filtered.shape[0]}"
            )

        if cell_info_df_filtered.shape[0] > 0:
            regions_to_keep.append(focus_region_idx)

            num_cells += cell_info_df_filtered.shape[0]
            num_cells_pre_removal += before_removal

        if num_cells > N:
            break

    print(f"Number of regions to keep: {len(regions_to_keep)}")
    print(f"Number of cells: {num_cells}")
    print(f"Number of cells before removal: {num_cells_pre_removal}")

    # find all the cells whose focus_region_idx that are in the regions_to_keep
    cell_info_df_selected = cell_info_df.loc[
        cell_info_df["focus_region_idx"].isin(regions_to_keep)
    ]
    
    low_mag_region_result_df = low_mag_region_result_df.loc[
        low_mag_region_result_df["idx"].isin(regions_to_keep)   
    ]

    high_mag_region_result_df = high_mag_region_result_df.loc[
        high_mag_region_result_df["idx"].isin(regions_to_keep)
    ]
    
    assert cell_info_df_selected.shape[0] == num_cells_pre_removal, f"Number of cells in cell_info_df_selected {cell_info_df_selected.shape[0]} does not match num_cells {num_cells}"

    for focus_region_id in tqdm(regions_to_keep, desc="Copying Regions"):
        filename = f"{focus_region_id}.jpg"

        # copy the high mag annotated focus region symbolic link
        src = os.path.join(
            result_dir_path,
            "focus_regions",
            "high_mag_annotated",
            filename,
        )

        dst = os.path.join(
            result_dir_path,
            focus_regions_save_subdir,
            "high_mag_annotated",
            filename,
        )

        os.symlink(src, dst)

        # copy the high mag unannotated focus region symbolic link
        src = os.path.join(
            result_dir_path,
            "focus_regions",
            "high_mag_unannotated",
            filename,
        )

        dst = os.path.join(
            result_dir_path,
            focus_regions_save_subdir,
            "high_mag_unannotated",
            filename,
        )

        os.symlink(src, dst)

    # iterate over the rows of the cell_info_df_selected
    for idx, row in tqdm(
        cell_info_df_selected.iterrows(),
        desc="Copying Cells",
        total=num_cells_pre_removal,
    ):
        
        if verbose:
            print(
                f"Copying cell named {row['name']} at {row['local_idx']} from {row['focus_region_idx']}"
            )

        cellname = row["name"]
        label = row["label"]

        name_label = cellname.split("-")[0]

        assert (
            name_label == label
        ), f"Name label {name_label} does not match label {label}"

        os.makedirs(
            os.path.join(result_dir_path, cell_save_subdir, label), exist_ok=True
        )

        src = os.path.join(result_dir_path, "cells", label, cellname)
        dst = os.path.join(result_dir_path, cell_save_subdir, label, cellname)

        os.symlink(src, dst)

    # save the selected data frames in the respective directories
    cell_info_df_selected.to_csv(
        os.path.join(result_dir_path, cell_save_subdir, "cells_info.csv"),
        index=False,
    )

    high_mag_region_result_df.to_csv(
        os.path.join(
            result_dir_path, focus_regions_save_subdir, "high_mag_focus_regions_info.csv"
        ),
        index=False,
    )

    low_mag_region_result_df.to_csv(
        os.path.join(
            result_dir_path, focus_regions_save_subdir, "focus_regions_info.csv"
        ),
        index=False,
    )

if __name__ == "__main__":

    results_root_dir = "/media/hdd2/neo/SameDayLLBMAResults"
    # find all the subdirectories of the results_root_dir
    subdirs = [
        subdir
        for subdir in os.listdir(results_root_dir)
        if os.path.isdir(os.path.join(results_root_dir, subdir))
    ]

    # print the number of subdirectories that start with ERROR_
    error_dirs = [subdir for subdir in subdirs if subdir.startswith("ERROR_")]
    non_error_dirs = [subdir for subdir in subdirs if not subdir.startswith("ERROR_")]  

    print(f"Number of error directories: {len(error_dirs)}")
    print(f"Number of non-error directories: {len(non_error_dirs)}")
    print(f"Number of total directories: {len(subdirs)}")

    for non_error_dir in non_error_dirs:
        sample_N_cells(os.path.join(results_root_dir, non_error_dir))


    # test_result_dir = "/media/hdd2/neo/test_slide_result_dir"
    # sample_N_cells(test_result_dir, verbose=True)