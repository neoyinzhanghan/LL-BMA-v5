####################################################################################################
# Imports ###########################################################################################
####################################################################################################

# Outside imports ##################################################################################
import ray
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
import contextlib
import sys
from ultralytics import YOLO

# Within package imports ###########################################################################
from LLBMA.resources.BMAassumptions import *
from LLBMA.brain.metrics import bb_intersection_over_union_fixed_dims
from LLBMA.vision.processing import zero_pad
from LLBMA.WBCCandidate import WBCCandidate
from LLBMA.vision.image_quality import VoL
from LLBMA.brain.utils import *


def _remove_wbc_df_duplicates(df):
    filtered_df = pd.DataFrame(
        columns=["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"]
    )

    for i in range(len(df)):

        found_duplicate = False
        for j in range(len(filtered_df)):
            iou = bb_intersection_over_union_fixed_dims(
                df.iloc[i][["TL_x", "TL_y", "BR_x", "BR_y"]],
                filtered_df.iloc[j][["TL_x", "TL_y", "BR_x", "BR_y"]],
            )
            if iou > 0.5:
                if df.iloc[i]["confidence"] > filtered_df.iloc[j]["confidence"]:
                    # then replace the jth row of filtered_df with the ith row of df
                    filtered_df.iloc[j] = df.iloc[i]

                found_duplicate = True
                break

        if not found_duplicate:
            # add the ith row of df to filtered_df

            if filtered_df.empty:
                filtered_df = pd.DataFrame(
                    [df.iloc[i]],
                    columns=["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"],
                )
            else:
                filtered_df = pd.concat(
                    [
                        filtered_df,
                        pd.DataFrame(
                            [df.iloc[i]],
                            columns=[
                                "TL_x",
                                "TL_y",
                                "BR_x",
                                "BR_y",
                                "confidence",
                                "class",
                            ],
                        ),
                    ]
                )

    return filtered_df


# def _remove_wbc_df_duplicates(df):  # TODO -- bad algorithm here -- need to fix
#     """Remove duplicate WBCs from the df, in place, return None."""
#     i = 0

#     while i < len(df):
#         j = i + 1
#         while j < len(df):
#             iou = bb_intersection_over_union(
#                 df.iloc[i][["TL_x", "TL_y", "BR_x", "BR_y"]],
#                 df.iloc[j][["TL_x", "TL_y", "BR_x", "BR_y"]],
#             )
#             if iou > 0.5:
#                 if df.iloc[i]["confidence"] > df.iloc[j]["confidence"]:
#                     df = df.drop(df.index[j])
#                     df.reset_index(drop=True, inplace=True)
#                 else:
#                     df = df.drop(df.index[i])
#                     df.reset_index(drop=True, inplace=True)
#                     break
#             else:
#                 j += 1
#         if j == len(df):
#             i += 1


def YOLO_detect(model, image, conf_thres, verbose=False):
    """Apply the YOLO model to an image."""

    # move the model to the gpu if available
    # if torch.cuda.is_available():
    #     model.model.cuda()

    # # convert the image from PIL to torch tensor
    # image_tens = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    # # move the image to the gpu if available
    # if torch.cuda.is_available():
    #     image_tens = image_tens.cuda()

    # conf_thres must be a float between 0 and 1 of not raise a ValueError
    assert 0 <= conf_thres <= 1

    # The model applies to a list of image but we only have one image it is still a list
    result = model([image], conf=conf_thres)[0]

    boxes = result.boxes.data  # This grabs the output annotations

    # This for-loop is for converting the output annotations into a nicely organized
    #     pandas dataframe with the columns TL_x, TL_y, BR_x, BR_y, confidence, class
    #     TL_x means the x coordinate of the top-left corner of the bounding box
    #     in absolute pixel coordinates, and BR_y, for instance, stands for the y
    #     coordinate of the bottom-right corner.

    # df = pd.DataFrame(columns=["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"])

    l1 = len(boxes)

    rows = []

    for i in range(l1):
        box = boxes[i]

        TL_x, TL_y = int(box[0]), int(box[1])
        BR_x, BR_y = int(box[2]), int(box[3])
        conf = float(box[4])
        cls = int(box[5])

        rows.append([TL_x, TL_y, BR_x, BR_y, conf, cls])

        # # use pd.concat instead of append to avoid deprecation
        # df = pd.concat(
        #     [
        #         df,
        #         pd.DataFrame(
        #             [[TL_x, TL_y, BR_x, BR_y, conf, cls]],
        #             columns=["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"],
        #         ),
        #     ]
        # )

    df = pd.DataFrame(
        rows, columns=["TL_x", "TL_y", "BR_x", "BR_y", "confidence", "class"]
    )

    if verbose:  # draw the bounding boxes on the image and display it
        # draw the bounding boxes on the image
        for i in range(len(df)):
            box = df.iloc[i]
            cv2.rectangle(
                image,
                (box["TL_x"], box["TL_y"]),
                (box["BR_x"], box["BR_y"]),
                (0, 255, 0),
                2,
            )

        # display the image
        plt.imshow(image)
        plt.show()

    # go through the df to remove duplicates, if two bounding boxes have iou > 0.5, then remove the one with lower confidence
    cleaned_df = _remove_wbc_df_duplicates(df)

    return cleaned_df


# @ray.remote(num_gpus=num_gpus_per_manager, num_cpus=num_cpus_per_manager)
@ray.remote(num_gpus=1)
class YOLOManager:
    """A Class representing a YOLO Manager that manages the object detection of a WSI.

    === Class Attributes ===
    - model : the YOLO model
    - ckpt_path : the path to the checkpoint of the YOLO model
    - conf_thres : the confidence threshold of the YOLO model
    - save_dir : the directory to save the results
    - hoarding : True of False, whether to hoard the results
    - num_detected : the total number of WBCs already detected by this YOLOManager
    - max_num_wbc : the maximum number of WBCs to detect by this YOLOManager
    """

    def __init__(
        self,
        ckpt_path,
        conf_thres,
        save_dir,
        hoarding=False,
        max_num_wbc=max_num_wbc_per_manager,
    ):
        """Initialize the YOLOManager object."""

        self.model = YOLO(ckpt_path)
        self.ckpt_path = ckpt_path
        self.conf_thres = conf_thres
        self.save_dir = save_dir
        self.hoarding = hoarding

        self.model.to("cuda")
        self.num_detected = 0
        self.max_num_wbc = max_num_wbc

    def async_find_wbc_candidates(self, focus_region):
        """Find WBC candidates in the image."""

        wbc_candidates = []

        if self.num_detected >= self.max_num_wbc:

            wbc_candidate_bboxes = []
            df = pd.DataFrame(
                columns=[
                    "TL_x",
                    "TL_y",
                    "BR_x",
                    "BR_y",
                    "confidence",
                    "focus_region_TL_x",
                    "focus_region_TL_y",
                    "local_idx",
                    "focus_region_idx",
                ]
            )

            focus_region.wbc_candidate_bboxes = wbc_candidate_bboxes
            focus_region.wbc_candidates = wbc_candidates
            focus_region.YOLO_df = df

            # focus_region._save_YOLO_df(self.save_dir)
            return focus_region

        df = YOLO_detect(self.model, focus_region.image, conf_thres=self.conf_thres)

        # add the coordinate of the focus region to the df
        df["focus_region_TL_x"] = focus_region.resampled_coordinate[0]
        df["focus_region_TL_y"] = focus_region.resampled_coordinate[1]

        # add two columns, one is the local_idx, and the other is the focus_region_idx
        df["local_idx"] = np.arange(len(df))
        # make sure the local_idx is of type int
        df["local_idx"] = df["local_idx"].astype(int)
        df["focus_region_idx"] = focus_region.idx

        # wbc_candidate_bboxes : a list of bbox of the WBC candidates in the level 0 view in the format of (TL_x, TL_y, BR_x, BR_y) in relative to the focus region
        wbc_candidate_bboxes = []

        # traverse through the df and create a list of WBCCandidate objects
        for i in range(len(df)):
            # get the ith row of the df as a dictionary
            row = df.iloc[i].to_dict()

            # compute the centroid coordinate of the bounding box in the focus regions
            centroid_x_level_0 = (
                (row["BR_x"] - row["TL_x"]) // 2
                + row["TL_x"]
                + row["focus_region_TL_x"]
            )
            centroid_y_level_0 = (
                (row["BR_y"] - row["TL_y"]) // 2
                + row["TL_y"]
                + row["focus_region_TL_y"]
            )

            centroid_x_intra_image = centroid_x_level_0 - row["focus_region_TL_x"]
            centroid_y_intra_image = centroid_y_level_0 - row["focus_region_TL_y"]

            confidence = row["confidence"]

            if centroid_x_intra_image - snap_shot_size // 2 < 0:
                # then the cell is too close to the left edge of the image and should not be included
                continue
            elif centroid_x_intra_image + snap_shot_size // 2 >= focus_regions_size:
                # then the cell is too close to the right edge of the image and should not be included
                continue
            elif centroid_y_intra_image - snap_shot_size // 2 < 0:
                # then the cell is too close to the top edge of the image and should not be included
                continue
            elif centroid_y_intra_image + snap_shot_size // 2 >= focus_regions_size:
                # then the cell is too close to the bottom edge of the image and should not be included
                continue

            # # check whether if a square of size snap_shot_size centered at the centroid is out of bound of focus_region.image
            # if (
            #     centroid_x_intra_image - snap_shot_size // 2 < 0
            #     or centroid_x_intra_image + snap_shot_size // 2 >= focus_regions_size
            #     or centroid_y_intra_image - snap_shot_size // 2 < 0
            #     or centroid_y_intra_image + snap_shot_size // 2 >= focus_regions_size
            # ):
            #     continue  # if so, then skip this candidate # TODO remove this is deprecated after padding

            # get the YOLO_bbox
            YOLO_bbox_intra_image = (row["TL_x"], row["TL_y"], row["BR_x"], row["BR_y"])

            # use YOLO_bbox_intra_image to crop the focus_region.image
            YOLO_bbox_image = focus_region.image.crop(YOLO_bbox_intra_image)

            # get the snap_shot_bbox
            snap_shot_bbox_intra_image = (
                int(centroid_x_intra_image - snap_shot_size // 2),
                int(centroid_y_intra_image - snap_shot_size // 2),
                int(centroid_x_intra_image + snap_shot_size // 2),
                int(centroid_y_intra_image + snap_shot_size // 2),
            )

            # use snap_shot_bbox to crop the focus_region.image
            snap_shot = focus_region.image.crop(snap_shot_bbox_intra_image)

            # zero pad the YOLO_bbox_image to have square dimension of snap_shot_size
            padded_YOLO_bbox_image = zero_pad(YOLO_bbox_image, snap_shot_size)

            # use the focus_region.location to compute the snap_shot_bbox in the level_0 view and the YOLO_bbox in the level_0 view
            snap_shot_bbox = (
                int(snap_shot_bbox_intra_image[0] + focus_region.resampled_coordinate[0]),
                int(snap_shot_bbox_intra_image[1] + focus_region.resampled_coordinate[1]),
                int(snap_shot_bbox_intra_image[2] + focus_region.resampled_coordinate[0]),
                int(snap_shot_bbox_intra_image[3] + focus_region.resampled_coordinate[1]),
            )

            YOLO_bbox = (
                int(YOLO_bbox_intra_image[0] + focus_region.resampled_coordinate[0]),
                int(YOLO_bbox_intra_image[1] + focus_region.resampled_coordinate[1]),
                int(YOLO_bbox_intra_image[2] + focus_region.resampled_coordinate[0]),
                int(YOLO_bbox_intra_image[3] + focus_region.resampled_coordinate[1]),
            )

            YOLO_bbox_relative = (
                YOLO_bbox_intra_image[0],
                YOLO_bbox_intra_image[1],
                YOLO_bbox_intra_image[2],
                YOLO_bbox_intra_image[3],
            )

            wbc_candidate_bboxes.append(YOLO_bbox_relative)

            # create a WBCCandidate object
            wbc_candidate = WBCCandidate(
                snap_shot,
                YOLO_bbox_image,
                padded_YOLO_bbox_image,
                snap_shot_bbox,
                YOLO_bbox,
                confidence,
                local_idx=row["local_idx"],
                focus_region_idx=focus_region.idx,
            )

            wbc_candidates.append(wbc_candidate)

        focus_region.wbc_candidate_bboxes = wbc_candidate_bboxes
        focus_region.wbc_candidates = wbc_candidates
        focus_region.YOLO_df = df

        focus_region._save_YOLO_df(self.save_dir)

        self.num_detected += len(wbc_candidates)

        return focus_region

    def async_find_wbc_candidates_batch(self, batch):
        """Find WBC candidates in the image."""

        processed_focus_regions = []

        for focus_region in batch:
            if self.num_detected >= self.max_num_wbc:
                processed_focus_regions.append(None)
            else:
                processed_focus_regions.append(
                    self.async_find_wbc_candidates(focus_region)
                )

        return processed_focus_regions

    def get_num_detected(self):
        return self.num_detected
