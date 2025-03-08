import os
import ray
import cv2
import json
import pandas as pd
import numpy as np
from LLBMA.brain.BMAHighMagRegionChecker import (
    BMAHighMagRegionCheckerBatched,
    BMAHighMagRegionCheckerBatchedSerial,
)
from LLBMA.resources.BMAassumptions import (
    num_region_clf_managers,
    high_mag_region_clf_ckpt_path,
    # min_num_focus_regions,
    high_mag_region_clf_threshold,
    # max_num_focus_regions,
    topview_downsampling_factor,
)
from LLBMA.brain.FocusRegionDataloader import (
    get_high_mag_focus_region_dataloader,
)
from LLBMA.communication.visualization import save_hist_KDE_rug_plot
from tqdm import tqdm
from PIL import Image
from ray.exceptions import RayTaskError


def sort_focus_regions_based_on_low_mag_score(focus_regions):
    """Sort the focus regions based on the confidence score at low magnification."""
    return sorted(
        focus_regions, key=lambda x: x.adequate_confidence_score, reverse=True
    )


class BMAHighMagRegionCheckTracker:
    """A class that keeps track of focus regions that made it past the low magnification checks.
    This class keeps track of the high magnification quality control of these regions.

    === Class Attributes ===
    - focus_regions: a list of focus regions that made it past the low magnification checks
    - info_df: a pandas DataFrame that stores the information of the focus regions
    - save_dir: the directory where the results will be saved

    """

    def __init__(self, focus_regions, save_dir) -> None:
        sorted_focus_regions = sort_focus_regions_based_on_low_mag_score(focus_regions)

        self.save_dir = save_dir

        dataloader = get_high_mag_focus_region_dataloader(sorted_focus_regions)
        
        new_focus_regions = []
        
        high_mag_checker = BMAHighMagRegionCheckerBatchedSerial(model_ckpt_path=high_mag_region_clf_ckpt_path)

        for batch in tqdm(dataloader, desc="Checking high magnification focus regions...", total=len(dataloader)):
            new_focus_regions.extend(high_mag_checker.check_high_mag_score(batch)[0])

        # ray.shutdown()
        # ray.init(
        #     ignore_reinit_error=True,
        #     # object_store_memory=10 * 1024 * 1024 * 1024,  # 10GB
        # )

        # high_mag_checkers = [
        #     BMAHighMagRegionCheckerBatched.remote(
        #         model_ckpt_path=high_mag_region_clf_ckpt_path,
        #     )
        #     for _ in range(num_region_clf_managers)
        # ]

        # tasks = {}
        # new_focus_regions = []

        # # Assign all batches to workers
        # for i, batch in enumerate(dataloader):
        #     # Assign each batch to a worker
        #     worker_idx = i % len(high_mag_checkers)  # Round-robin assignment
        #     high_mag_checker = high_mag_checkers[worker_idx]
        #     task_id = high_mag_checker.async_check_high_mag_score.remote(batch)
        #     tasks[task_id] = high_mag_checker  # Use ObjectRef as the key

        # # Track progress with tqdm
        # with tqdm(
        #     total=len(focus_regions),
        #     desc="Getting high magnification focus regions diagnostics...",
        # ) as pbar:
        #     try:
        #         # Process completed tasks
        #         while tasks:
        #             done, _ = ray.wait(list(tasks.keys()), timeout=0.1)
        #             for done_task_id in done:
        #                 try:
        #                     # Retrieve results from Ray task
        #                     focus_regions, _ = ray.get(done_task_id)
        #                     new_focus_regions.extend(focus_regions)

        #                     # Update progress bar
        #                     pbar.update(len(focus_regions))
        #                 except RayTaskError as e:
        #                     print(e)
        #                     raise e
        #                 finally:
        #                     # Remove completed task from the dictionary
        #                     del tasks[done_task_id]  # Use ObjectRef as the key

        #     except Exception as e:
        #         print(f"Error occurred: {e}")

        # ray.shutdown()

        self.focus_regions = new_focus_regions

        print(
            f"Number of focus regions after high magnification check: {len(self.focus_regions)}"
        )

        # populate the info_df with the information of the focus regions
        info_dct = {
            "idx": [],
            "VoL_high_mag": [],
            "adequate_confidence_score_high_mag": [],
            "selected_high_mag": [],
            "coordinate": [],
        }

        for focus_region in self.focus_regions:
            info_dct["idx"].append(focus_region.idx)
            info_dct["VoL_high_mag"].append(focus_region.VoL_high_mag)
            info_dct["adequate_confidence_score_high_mag"].append(
                focus_region.adequate_confidence_score_high_mag
            )
            info_dct["selected_high_mag"].append(
                focus_region.adequate_confidence_score_high_mag
                > high_mag_region_clf_threshold
            )
            info_dct["coordinate"].append(focus_region.resampled_coordinate)

        # create a pandas DataFrame to store the information of the focus regions
        # it should have the following columns:
        # --idx: the index of the focus region
        # --VoL_high_mag: the volume of the focus region at high magnification
        # --adequate_confidence_score_high_mag: the confidence score of the focus region at high magnification

        self.info_df = pd.DataFrame(info_dct)

        print(
            f"Number of selected focus regions after high magnification check: {len(self.info_df[self.info_df['selected_high_mag']])}"
        )

        # save the df in the /media/hdd3/test_high_mag_focus_regions_info.csv
        # self.info_df.to_csv("/media/hdd3/test_high_mag_focus_regions_info.csv")

    def get_good_focus_regions(self):
        """The criterion for a good focus region is that it has an adequate confidence score at high magnification:
        - VoL_high_mag > 7
        - adequate_confidence_score_high_mag > high_mag_region_clf_threshold
        """

        good_focus_regions = []

        for focus_region in self.focus_regions:
            if (
                focus_region.VoL_high_mag > 8
                and focus_region.adequate_confidence_score_high_mag
                > high_mag_region_clf_threshold
            ):
                good_focus_regions.append(focus_region)
                
        for focus_region in tqdm(self.focus_regions, desc="Getting good focus regions images..."):
            focus_region.get_image_from_dzi_high_mag_image_path()
            
        # if len(good_focus_regions) < min_num_focus_regions:
        #     raise HighMagCheckFailedError(
        #         f"Only {len(good_focus_regions)} good focus regions remain after the high magnification check, and the minimum number of focus regions required is {min_num_focus_regions}."
        #     )

        return good_focus_regions

    def save_results(self, save_dir):

        # save the df in the save_dir/focus_regions/high_mag_focus_regions_info.csv
        self.info_df.to_csv(
            f"{save_dir}/focus_regions/high_mag_focus_regions_info.csv", index=False
        )

    def save_high_mag_score_plot(self):

        plot_save_path = os.path.join(
            self.save_dir, "focus_regions", "high_mag_score_plot.png"
        )

        save_hist_KDE_rug_plot(
            self.info_df,
            column_name="adequate_confidence_score_high_mag",
            save_path=plot_save_path,
            title="High Magnification Confidence Score Distribution",
            lines=[high_mag_region_clf_threshold],
        )

    def save_confidence_heatmap(self, topview_img_pil, save_dir):
        # Convert the PIL image to OpenCV format (BGR)
        topview_img_cv = cv2.cvtColor(np.array(topview_img_pil), cv2.COLOR_RGB2BGR)

        # Create a blank image (heatmap) with the same dimensions as topview_img, but with 3 channels for RGB colors
        heatmap = np.zeros((*topview_img_cv.shape[:2], 3), dtype=np.uint8)

        # Iterate through the patches
        for index, row in self.info_df.iterrows():
            # Extract the bounding box and confidence score
            TL_x, TL_y, BR_x, BR_y = row["coordinate"]
            confidence_score = row["adequate_confidence_score_high_mag"]

            # Adjust the coordinates for the downsampling factor
            TL_x_adj = int(TL_x / topview_downsampling_factor)
            TL_y_adj = int(TL_y / topview_downsampling_factor)
            BR_x_adj = int(BR_x / topview_downsampling_factor)
            BR_y_adj = int(BR_y / topview_downsampling_factor)

            # Calculate color based on confidence_score, red for 0, green for 1
            red_intensity = (1 - confidence_score) * 255
            green_intensity = confidence_score * 255
            color = [0, green_intensity, red_intensity]  # BGR format for OpenCV

            # Assign the color to the corresponding region in the heatmap
            heatmap[TL_y_adj:BR_y_adj, TL_x_adj:BR_x_adj] = color

        # Since the heatmap is already in BGR format, we don't need to apply a colormap
        heatmap_colored = heatmap

        # Overlay the heatmap on the original topview image
        overlay_img_cv = cv2.addWeighted(topview_img_cv, 0.7, heatmap_colored, 0.3, 0)

        # Convert back to PIL image in RGB format
        overlay_img_pil = Image.fromarray(
            cv2.cvtColor(overlay_img_cv, cv2.COLOR_BGR2RGB)
        )

        # save the overlay_img_pil in save_dir
        overlay_img_pil.save(os.path.join(save_dir, "confidence_heatmap_high_mag.png"))

    def save_selected_focus_region_locations(self, topview_img_pil, save_dir):
        # Convert the PIL image to OpenCV format (BGR)
        topview_img_cv = cv2.cvtColor(np.array(topview_img_pil), cv2.COLOR_RGB2BGR)

        # Create a blank image (heatmap) with the same dimensions as topview_img, but with 3 channels for RGB colors
        heatmap = np.zeros((*topview_img_cv.shape[:2], 3), dtype=np.uint8)

        # Iterate through the patches
        for index, row in self.info_df.iterrows():
            # Extract the bounding box and confidence score
            TL_x, TL_y, BR_x, BR_y = row["coordinate"]
            selected = row["selected_high_mag"]

            # turn the selected from boolean to int, 0 for False, 1 for True
            selected = int(selected)

            # Adjust the coordinates for the downsampling factor
            TL_x_adj = int(TL_x / topview_downsampling_factor)
            TL_y_adj = int(TL_y / topview_downsampling_factor)
            BR_x_adj = int(BR_x / topview_downsampling_factor)
            BR_y_adj = int(BR_y / topview_downsampling_factor)

            # Calculate color based on confidence_score, red for 0, green for 1
            red_intensity = (1 - selected) * 255
            green_intensity = selected * 255
            color = [0, green_intensity, red_intensity]  # BGR format for OpenCV

            # Assign the color to the corresponding region in the heatmap
            heatmap[TL_y_adj:BR_y_adj, TL_x_adj:BR_x_adj] = color

        # Since the heatmap is already in BGR format, we don't need to apply a colormap
        heatmap_colored = heatmap

        # Overlay the heatmap on the original topview image
        overlay_img_cv = cv2.addWeighted(topview_img_cv, 0.7, heatmap_colored, 0.3, 0)

        # Convert back to PIL image in RGB format
        overlay_img_pil = Image.fromarray(
            cv2.cvtColor(overlay_img_cv, cv2.COLOR_BGR2RGB)
        )

        # save the overlay_img_pil in save_dir
        overlay_img_pil.save(
            os.path.join(save_dir, "top_view_with_selected_focus_regions_locations.png")
        )


class HighMagCheckFailedError(Exception):
    """This error is raised when not enough good focus regions remain after the high magnification check."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
