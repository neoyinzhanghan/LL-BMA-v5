import os
from tqdm import tqdm


def hoard_focus_regions_after_high_mag_scores(focus_regions, save_dir, num_sig_figs=6):
    """
    Focus regions have an attributes called high_mag_score which is a float and an attribute called image.
    """

    os.makedirs(save_dir, exist_ok=True)

    # for each focus region object we will save the image in the save_dir naming it as the int(high_mag_score * 10^num_sig_figs) with .jpeg extension
    for focus_region in tqdm(
        focus_regions, "Hoarding Focus Region Images Based on High Mag Scores"
    ):
        image = focus_region.image
        high_mag_score = focus_region.adequate_confidence_score_high_mag

        image.save(
            os.path.join(
                save_dir,
                f"{int(high_mag_score * 10**num_sig_figs)}_{focus_region.idx}.jpeg",
            ),
            "JPEG",
        )


def hoard_focus_regions_after_high_mag_scores_from_tracker(
    tracker, save_dir, num_sig_figs=6
):
    focus_regions = tracker.focus_regions

    hoard_focus_regions_after_high_mag_scores(
        focus_regions=focus_regions, save_dir=save_dir, num_sig_figs=num_sig_figs
    )
