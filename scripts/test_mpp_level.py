import os
import csv
from openslide import OpenSlide
from tqdm import tqdm


def extract_slide_info(slide_path):
    try:
        slide = OpenSlide(slide_path)
        level0_mpp = float(slide.properties.get("openslide.mpp-x", "Unknown"))
        level3_downsample = (
            slide.level_downsamples[3]
            if len(slide.level_downsamples) > 3
            else "Unknown"
        )
        level7_downsample = (
            slide.level_downsamples[7]
            if len(slide.level_downsamples) > 7
            else "Unknown"
        )
        return level0_mpp, level3_downsample, level7_downsample
    except Exception as e:
        print(f"Error processing {slide_path}: {e}")
        return "Unknown", "Unknown", "Unknown"


def scan_directory(data_dir):
    slide_info = []
    slide_files = []
    # Gather all slide files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(
                (".svs", ".tiff", ".ndpi")
            ):  # add other slide formats as necessary
                slide_files.append(os.path.join(root, file))

    # Process slide files with tqdm progress bar
    for slide_path in tqdm(slide_files, desc="Processing slides"):
        level0_mpp, level3_downsample, level7_downsample = extract_slide_info(
            slide_path
        )
        slide_info.append(
            [
                os.path.basename(slide_path),
                level0_mpp,
                level3_downsample,
                level7_downsample,
            ]
        )
    return slide_info


def save_to_csv(slide_info, output_path):
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Slide Name", "Level 0 MPP", "Level 3 Downsample", "Level 7 Downsample"]
        )
        for row in slide_info:
            writer.writerow(row)


def main(data_dir):
    slide_info = scan_directory(data_dir)
    output_path = os.path.join(data_dir, "mpp_info.csv")
    save_to_csv(slide_info, output_path)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    data_dir = (
        "/media/hdd1/neo/BMA_AML"  # Update this path to your specific data directory
    )
    main(data_dir)
