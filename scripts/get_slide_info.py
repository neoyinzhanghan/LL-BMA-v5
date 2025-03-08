import openslide

slide_path = "/media/hdd3/neo/error_slides_ndpi/test_slide.ndpi"

# get the level 0 dimensions
slide = openslide.OpenSlide(slide_path)
level_0_dimensions = slide.level_dimensions[0]

# print the width and height of the slide
print("Width: ", level_0_dimensions[0])
print("Height: ", level_0_dimensions[1])
