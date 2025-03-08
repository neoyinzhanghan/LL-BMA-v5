import os
import random
import shutil
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from LLBMA.resources.BMAassumptions import *

save_dir = "/media/hdd3/neo/CC_level_sets"
dump_dir = "/media/greg/534773e3-83ea-468f-a40d-46c913378014/neo/results_dir"
result_df_tracker = (
    "/home/greg/Documents/neo/LL-Eval/pipeline_nonerror_aggregate_df.csv"
)

# if save_dir already exists, delete it
if os.path.exists(save_dir):
    os.system(f"rm -rf {save_dir}")

os.makedirs(save_dir, exist_ok=True)

for cell_type in cellnames:
    os.makedirs(os.path.join(save_dir, cell_type), exist_ok=True)

increment = 0.05


def level_to_str(level: float):
    level = round(level, 2)
    return str(level).replace(".", "_")


def get_levels():
    levels = []
    level = 0.0
    while round(level, 2) <= 1.0:
        levels.append(level)
        level += increment

    levels_str = [level_to_str(level) for level in levels]
    return levels, levels_str


# get the levels and levels_str
levels, levels_str = get_levels()
all_levels, all_level_str = get_levels()

# print(levels)
# print(levels_str)
# print(all_levels)
# print(all_level_str)


def find_levels(score, levels=all_levels):
    for i in range(len(levels)):
        if i == 0:
            continue
        elif score <= levels[i]:
            # print(f"Score {score} is less than or equal to {levels[i]}")
            return levels[i - 1]
        # print(f"Score {score} is greater than {levels[i]}")

    print(f"Score {score} is greater than 1.0")
    print(f"Score {score} is faulty")
    return levels[0]


for cell_type in cellnames:
    for level_str in levels_str:
        os.makedirs(os.path.join(save_dir, cell_type, level_str), exist_ok=True)

# get all the paths in the result directory
# open the result_df_tracker
result_df = pd.read_csv(result_df_tracker)

# get the result_dir_name column as a list of strings
result_dirs = result_df["result_dir_name"].tolist()

subdirs = [os.path.join(dump_dir, subdir) for subdir in result_dirs]

# only keep the ones that do not start with ERROR
subdirs = [
    subdir for subdir in subdirs if not os.path.basename(subdir).startswith("ERROR")
]

num_per_subdir = 100

pseudo_idx = 0

for subdir in tqdm(subdirs, desc="Processing Result Dirs"):
    # check if the cells folder exists
    cells_dir = os.path.join(subdir, "cells")

    # check how many jpgs are in the cells folder
    jpgs = [
        os.path.join(root, file)
        for root, _, files in os.walk(cells_dir)
        for file in files
        if file.endswith(".jpg") and "YOLO" not in file
    ]

    if len(jpgs) < num_per_subdir:
        print(f"Skipping {subdir} because it has less than {num_per_subdir} jpgs")
        continue

    # randomly select num_per_subdir jpgs
    jpgs = random.sample(jpgs, num_per_subdir)

    cell_df_path = os.path.join(subdir, "cells", "cells_info.csv")

    cell_df = pd.read_csv(cell_df_path)

    for jpg in jpgs:
        jpg_name = os.path.basename(jpg)
        cell_df_row = cell_df[cell_df["name"] == jpg_name]

        for cell_type in cellnames:
            cell_type_prob = float(cell_df_row[cell_type].values[0])

            level = find_levels(cell_type_prob)
            level_str = level_to_str(level)

            new_name = f"{pseudo_idx}.jpg"
            new_path = os.path.join(save_dir, cell_type, level_str, new_name)

            # create a symlink to the jpg in the appropriate level directory
            shutil.copy(jpg, new_path)
            pseudo_idx += 1

for cell_type in cellnames:
    level_counts = {}
    level_counts_no_0_0 = {}

    for level_str in levels_str:
        level_dir = os.path.join(save_dir, cell_type, level_str)

        if os.path.exists(level_dir):
            num_jpgs = len(
                [
                    file
                    for file in os.listdir(level_dir)
                    if os.path.isfile(os.path.join(level_dir, file))
                ]
            )
        else:
            num_jpgs = 0

        level_counts[level_str] = num_jpgs
        if level_str != "0_0":
            level_counts_no_0_0[level_str] = num_jpgs

    # Sort the levels for consistent plotting
    sorted_levels = sorted(level_counts.keys())
    sorted_counts = [level_counts[level] for level in sorted_levels]

    sorted_levels_no_0_0 = sorted(level_counts_no_0_0.keys())
    sorted_counts_no_0_0 = [
        level_counts_no_0_0[level] for level in sorted_levels_no_0_0
    ]

    # Ensure the directory for saving plots exists
    plot_dir = os.path.join(save_dir, cell_type)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot with 0_0
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_levels, sorted_counts, color="blue")
    plt.title(f"Number of JPGs per Level for {cell_type} (Including 0_0)")
    plt.xlabel("Levels")
    plt.ylabel("Number of JPGs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "num_cells.png")
    plt.savefig(plot_path)
    plt.close()

    # Plot without 0_0
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_levels_no_0_0, sorted_counts_no_0_0, color="green")
    plt.title(f"Number of JPGs per Level for {cell_type} (Excluding 0_0)")
    plt.xlabel("Levels")
    plt.ylabel("Number of JPGs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path_no_0_0 = os.path.join(plot_dir, "num_cells_no_0_0.png")
    plt.savefig(plot_path_no_0_0)
    plt.close()

    print(f"Plots saved for {cell_type}: {plot_path}, {plot_path_no_0_0}")
