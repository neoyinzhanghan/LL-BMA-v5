from LLBMA.resources.BMAassumptions import snap_shot_size


def bb_intersection_over_union(boxA, boxB):
    """
    Credit: This code is obtained from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    Compute the intersection over union of two bounding boxes.
    Assuming the boxes are stored in an iterable of length 4. In the format of (TL_x, TL_y, BR_x, BR_y).
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA.iloc[0], boxB.iloc[0])
    yA = max(boxA.iloc[1], boxB.iloc[1])
    xB = min(boxA.iloc[2], boxB.iloc[2])
    yB = min(boxA.iloc[3], boxB.iloc[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA.iloc[2] - boxA.iloc[0]) * (boxA.iloc[3] - boxA.iloc[1]))
    boxBArea = abs((boxB.iloc[2] - boxB.iloc[0]) * (boxB.iloc[3] - boxB.iloc[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_intersection_over_union_fixed_dims(boxA, boxB, box_size=snap_shot_size):
    """
    Compute the intersection over union of two bounding boxes.
    Assuming the boxes are stored in an iterable of length 4. In the format of (TL_x, TL_y, BR_x, BR_y).
    """

    centroid_x_boxA, centroid_y_boxA = (boxA.iloc[0] + boxA.iloc[2]) / 2, (
        boxA.iloc[1] + boxA.iloc[3]
    ) / 2
    centroid_x_boxB, centroid_y_boxB = (boxB.iloc[0] + boxB.iloc[2]) / 2, (
        boxB.iloc[1] + boxB.iloc[3]
    ) / 2

    boxA = (
        centroid_x_boxA - box_size // 2,
        centroid_y_boxA - box_size // 2,
        centroid_x_boxA + box_size // 2,
        centroid_y_boxA + box_size // 2,
    )

    boxB = (
        centroid_x_boxB - box_size // 2,
        centroid_y_boxB - box_size // 2,
        centroid_x_boxB + box_size // 2,
        centroid_y_boxB + box_size // 2,
    )

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + box_size, boxB[0] + box_size)
    yB = min(boxA[1] + box_size, boxB[1] + box_size)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs(box_size * box_size)
    boxBArea = abs(box_size * box_size)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def _one_contains_the_other(
    focus,
    another_focus,
    snap_shot_size,
    focus_region_size,
    search_view_downsampling_rate,
    area_prop_thres=0.5,
):
    """The bounding box is of size snap_shot_size around the focus as centered
    If the snap shot bounding box of another_focus overlaps with the focus region bounding box of focus for more than area_prop_thres proportion of the snap shot area,
    then return True, otherwise return False.
    """

    # must convert the coordinates of the foci to the level 0 view
    focus = (
        focus[0] * search_view_downsampling_rate,
        focus[1] * search_view_downsampling_rate,
    )
    another_focus = (
        another_focus[0] * search_view_downsampling_rate,
        another_focus[1] * search_view_downsampling_rate,
    )

    FR_TL_x = focus[0] - focus_region_size // 2
    FR_TL_y = focus[1] - focus_region_size // 2

    SS_TL_x = another_focus[0] - snap_shot_size // 2
    SS_TL_y = another_focus[1] - snap_shot_size // 2

    # computer the area of intersection between the snap shot bounding box of another_focus and the focus region bounding box of focus
    xA = max(FR_TL_x, SS_TL_x)
    yA = max(FR_TL_y, SS_TL_y)
    xB = min(FR_TL_x + focus_region_size, SS_TL_x + snap_shot_size)
    yB = min(FR_TL_y + focus_region_size, SS_TL_y + snap_shot_size)

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))

    return interArea / snap_shot_size**2 > area_prop_thres
