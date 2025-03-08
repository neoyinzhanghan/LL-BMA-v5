# LL-BMA-v5

**LeukoLocator for Bone Marrow Aspirate Analysis v5**

A comprehensive pipeline for automated cellular analysis of bone marrow aspirate slides using computer vision and deep learning.

## Overview

LL-BMA-v5 is a specialized tool designed for hematopathologists and researchers that automatically:

- Processes whole slide images (WSI) from bone marrow aspirates
- Identifies and classifies cellular components using deep learning
- Performs differential counting of white blood cells
- Generates detailed reports and visualizations of cell populations
- Exports results in standardized formats for further analysis

The pipeline combines state-of-the-art computer vision techniques with domain-specific models trained on thousands of annotated cells to deliver high-accuracy results.

## Installation

```bash
git clone https://github.com/yourusername/LL-BMA-v5.git
cd LL-BMA-v5
pip install -e .
```

## Usage

The main entry point for analyzing slides is through the `analyse_bma` function:

```python:README.md
from LLBMA.front_end.api import analyse_bma

# Basic usage
result_dir, error = analyse_bma(
    slide_path="/path/to/slide.svs",
    dump_dir="/path/to/output",
    hoarding=True,
    continue_on_error=False,
    do_extract_features=False,
    check_specimen_clf=False
)

# With DZI tiling (for web visualization)
result_dir, error = analyse_bma(
    slide_path="/path/to/slide.svs",
    dump_dir="/path/to/output",
    hoarding=True,
    continue_on_error=False,
    tiling_dump_dir="/path/to/dzi_tiles",
    tiling_format="dzi"
)
```

## Example Script

```python
import os
import time
from LLBMA.front_end.api import analyse_bma

slide_path = "/path/to/your/slide.svs"
dump_dir = "/path/to/output"
tiling_dump_dir = "/path/to/dzi_tiles"

start_time = time.time()

# Run the analysis
analyse_bma(
    slide_path=slide_path,
    dump_dir=dump_dir,
    hoarding=True,
    extra_hoarding=False,
    continue_on_error=False,
    do_extract_features=False,
    check_specimen_clf=False,
    tiling_dump_dir=tiling_dump_dir,
    tiling_format="dzi",
)

print(f"Time taken: {time.time() - start_time:.2f} seconds to process {slide_path}")
```

## Key Parameters

- `slide_path`: Path to the whole slide image file (.svs, .ndpi, etc.)
- `dump_dir`: Directory where results will be saved
- `hoarding`: Whether to save intermediate results for debugging/visualization (recommended)
- `continue_on_error`: Whether to continue processing if non-critical errors occur
- `do_extract_features`: Extract features for machine learning (optional)
- `check_specimen_clf`: Verify slide is a bone marrow aspirate with confidence scores
- `tiling_dump_dir`: Directory to save pyramid tiles for web visualization
- `tiling_format`: Format for tiling output (e.g., "dzi")

## Output

Results are saved in the specified `dump_dir` and include:
- Cell classification reports
- Differential count statistics
- Annotated regions of interest
- Detected cell images categorized by cell type
- Quality control metrics

## Requirements

- Python 3.8+
- PyTorch
- OpenSlide
- Ray (for parallel processing)
- Various deep learning and image processing libraries

For detailed documentation and advanced usage, please refer to the docs directory.