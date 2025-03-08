import openslide
from LLBMA.resources.BMAassumptions import topview_level
from LLBMA.vision.processing import read_with_timeout
from LLBMA.brain.SpecimenClf import get_specimen_type, calculate_specimen_conf
from LLBMA.BMACounter import BMACounter
from LLBMA.BMATopView import SpecimenError


def get_specimen_conf_dict(slide_path):
    """Get the confidence score for each specimen type of the slide."""

    try:
        wsi = openslide.OpenSlide(slide_path)
        top_view = read_with_timeout(
            wsi, (0, 0), topview_level, wsi.level_dimensions[topview_level]
        )

        return calculate_specimen_conf(top_view)

    except Exception as e:
        print(e)
        print(
            f"Could not get the confidence scores of {slide_path}, which means it will be classified as Others"
        )
        return {
            "Bone Marrow Aspirate": 0,
            "Peripheral Blood": 0,
            "Manual Peripheral Blood or Inadequate Bone Marrow Aspirate": 0,
            "Others": 2,
        }  ### We are going to record the confidence of Others as 2


def classify_specimen_type(slide_path):
    """Get the top view of the wsi and classify it"""

    try:
        wsi = openslide.OpenSlide(slide_path)
        top_view = read_with_timeout(
            wsi, (0, 0), topview_level, wsi.level_dimensions[topview_level]
        )

        specimen_type = get_specimen_type(top_view)

    except Exception as e:
        print(e)
        print(
            f"Could not classify the specimen type of {slide_path}, which means it will be classified as Others"
        )
        specimen_type = "Others"

    if specimen_type == "Bone Marrow Aspirate":
        return "BMA"

    elif specimen_type == "Peripheral Blood":
        return "PB"

    elif specimen_type == "Manual Peripheral Blood or Inadequate Bone Marrow Aspirate":
        return "MPBorIBMA"

    else:
        return "Others"


def analyse_bma(
    slide_path,
    dump_dir,
    hoarding=True,
    continue_on_error=False,
    do_extract_features=False,
    check_specimen_clf=False,
    ignore_specimen_type=True,
    tiling_dump_dir=None,
    tiling_format="dzi",
    **kwargs,
):
    """First classify the slide specimen type.
    --If BMA, then use BMACounter to tally differential.
    --If PB, then use PBCounter to tally differential.
    --If MPB or IBMA, then use BMACounter to tally differential.
    --If Others, then do nothing.

    In all situations, return the specimen type.
    """
    if check_specimen_clf:
        raise NotImplementedError(
            "The check_specimen_clf option is not yet implemented"
        )
        # # classify the slide specimen type
        # specimen_type = classify_specimen_type(slide_path)

        # print("Classified Specimen Type:", specimen_type)

        # if specimen_type == "BMA":
        #     # use BMACounter to tally differential
        #     bma_counter = BMACounter(
        #         slide_path,
        #         dump_dir=dump_dir,
        #         hoarding=hoarding,
        #         continue_on_error=continue_on_error,
        #         do_extract_features=do_extract_features,
        #     )
        #     bma_counter.tally_differential()

        # elif specimen_type == "PB":
        #     raise SpecimenError(
        #         f"Cannot analyze the slide {slide_path} as it is classified as PB"
        #     )

        # elif specimen_type == "MPBorIBMA":

        #     # warn the user that the slide is classified as MPBorIBMA without raising an error
        #     print(
        #         f"UserWarning: The slide {slide_path} as it is classified as MPBorIBMA by the Specimen Classifier. Running BMACounter to tally differential on inadequate BMA or manual PB specimens would likely lead to bad results."
        #     )
        #     # use BMACounter to tally differential
        #     bma_counter = BMACounter(
        #         slide_path,
        #         dump_dir=dump_dir,
        #         hoarding=hoarding,
        #         continue_on_error=continue_on_error,
        #         do_extract_features=do_extract_features,
        #     )
        #     bma_counter.tally_differential()

    else:
        # use BMACounter to tally differential
        bma_counter = BMACounter(
            slide_path,
            dump_dir=dump_dir,
            hoarding=hoarding,
            continue_on_error=continue_on_error,
            do_extract_features=do_extract_features,
            ignore_specimen_type=ignore_specimen_type,
            tiling_dump_dir=tiling_dump_dir,
            tiling_format=tiling_format,
        )
        bma_counter.tally_differential()

        return bma_counter.save_dir, bma_counter.error
