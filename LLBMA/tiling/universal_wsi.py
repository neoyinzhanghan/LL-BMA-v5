import openslide
from PIL import Image
from pathlib import Path


def get_slide_mpp(slide_path):
    """
    Get the microns per pixel (resolution) from a whole slide image.
    
    Args:
        slide_path (str): Path to the slide file
        
    Returns:
        tuple: (mpp_x, mpp_y, mpp_candidates)
            - mpp_x (float): Microns per pixel in x direction, or None if not found
            - mpp_y (float): Microns per pixel in y direction, or None if not found
            - mpp_candidates (dict): Dictionary of potential MPP-related properties if standard methods fail
    """
    slide = openslide.OpenSlide(slide_path)
    
    # Try to get MPP directly from standard OpenSlide properties
    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        slide.close()
        return mpp_x, mpp_y, None
    except (KeyError, ValueError):
        pass
    
    # Try to calculate from resolution information in TIFF properties
    try:
        spacing_x = float(slide.properties['tiff.XResolution'])
        spacing_y = float(slide.properties['tiff.YResolution'])
        unit = slide.properties.get('tiff.ResolutionUnit')
        
        # Convert based on unit (2 = inch, 3 = centimeter)
        if unit == '2':  # inches
            mpp_x = 25400 / spacing_x  # 25400 microns per inch
            mpp_y = 25400 / spacing_y
        elif unit == '3':  # centimeters
            mpp_x = 10000 / spacing_x  # 10000 microns per centimeter
            mpp_y = 10000 / spacing_y
            
        slide.close()
        return mpp_x, mpp_y, None
    except (KeyError, ValueError):
        pass
    
    # Get available properties if standard methods fail
    properties = dict(slide.properties)
    slide.close()
    
    # Look for any property that might contain resolution info
    mpp_candidates = {k: v for k, v in properties.items() 
                     if 'resolution' in k.lower() or 'mpp' in k.lower() or 'pixel' in k.lower()}
    
    return None, None, mpp_candidates  # Return None and candidates for manual inspection

class UniversalWSI():
    """
    Universal Whole Slide Image handler that supports multiple file formats.
    
    This class provides a unified interface for working with different whole slide image
    formats including .ndpi, .svs, and .dcm files.
    """
    
    def __init__(self, wsi_path):
        """
        Initialize the UniversalWSI object.
        
        Args:
            wsi_path (str): Path to the whole slide image file
            
        Raises:
            ValueError: If the file extension is not supported
        """
        self.wsi_path = wsi_path
        self.ext = Path(wsi_path).suffix
        if self.ext in ['.ndpi', '.svs', '.dcm']:
            self.wsi = openslide.OpenSlide(wsi_path)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")
    
    def get_level_0_dimensions(self):
        """
        Get the dimensions of the whole slide image at level 0 (highest resolution).
        
        Returns:
            tuple: (width, height) in pixels
        """
        if self.ext in ['.ndpi', '.svs']:
            width, height = self.wsi.level_dimensions[0]
        elif self.ext == '.dcm':
            width, height = self.wsi.dimensions
        
        return width, height
    
    def get_level_0_mpp(self):
        """
        Get the microns per pixel (resolution) at level 0.
        
        Returns:
            float: Microns per pixel value
            
        Raises:
            ValueError: If MPP information cannot be found
            AssertionError: If MPP values in x and y directions are not equal
        """
        mpp_x, mpp_y, mpp_candidates = get_slide_mpp(self.wsi_path)
        
        if mpp_x is not None and mpp_y is not None:
            assert round(mpp_x, 3) == round(mpp_y, 3), f"MPP x {mpp_x} and y {mpp_y} are not equal for {self.wsi_path}"
            return mpp_x
        else:
            raise ValueError(f"No MPP found for {self.wsi_path}, check out alternative output {mpp_candidates}")
    
    def read_ground_region(self, TL, tile_shape):
        """
        Read a region from the whole slide image at level 0.
        
        Args:
            TL (tuple): Top-left coordinate (x, y) of the region
            tile_shape (tuple): Shape (width, height) of the region to read
            
        Returns:
            PIL.Image: The extracted region as an RGB image
        """
        if self.ext in ['.ndpi', '.svs', '.dcm']:
            image = self.wsi.read_region(TL, 0, tile_shape)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            return image
        
        
if __name__ == "__main__":
    """Example usage of the UniversalWSI class with different file formats."""
    svs_path = "/home/neo/Documents/neo/neo_is_slide_tiling_god/23.CFNA.81 A1 H&E _124937.svs"
    ndpi_path = "/home/neo/Documents/neo/neo_is_slide_tiling_god/test.ndpi"
    dcm_path = "/home/neo/Documents/neo/neo_is_slide_tiling_god/ANONJ4JJKJ17J_1_2.dcm"
    
    for path in [svs_path, ndpi_path, dcm_path]:
        print(f"Processing {path}")
        wsi = UniversalWSI(path)
        print(f"Level 0 dimensions: {wsi.get_level_0_dimensions()}")
        print(f"Level 0 MPP: {wsi.get_level_0_mpp()}")
        print(f"Level 0 ground region: {wsi.read_ground_region((0, 0), (100, 100))}")
        print("+"*100)
