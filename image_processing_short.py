import numpy as np
from skimage import exposure        # Standard Library: Package for image processing.

'''
------------------------------------------------------------------------------------------------------------------------
Functions for Image Processing and Enhancement
------------------------------------------------------------------------------------------------------------------------
'''

def array_normalisation(array,new_min=0.0,new_max=255.0):
    """To normalise an input array."""
    
    array = array.astype(float)
    
    old_min = np.amin(array)
    old_max = np.amax(array)
    
    array = new_min + (array - old_min) * (new_max - new_min) / (old_max - old_min)
    
    array = array.astype('uint8')
    
    return array

#----------------------------------------------------------------------------------------------------------------------

	
def imageStretch(img_array, low_border, high_border):
    """Linear Stretching between Min and Max Values."""
    
    img_array = img_array.astype(float)
    image_array_clip = np.clip(img_array, low_border, high_border)
    image_array_new = array_normalisation(image_array_clip)
    image_array_new_int = image_array_new.astype(int)
    
    return image_array_new_int

# ----------------------------------------------------------------------------------------------------------------------


def image_meanStd_clip(img_array, std_factor=2):
    """Percentage Linear Contrast Stretch (Stretching around the mean value)."""
    
    img_array = img_array.astype(float)
    
    image_array_noData = np.extract(img_array>0,img_array)
    image_array_noData_meanValue = np.mean(image_array_noData)
    image_array_noData_stdValue = np.std(image_array_noData)
    low_border  = image_array_noData_meanValue - ( image_array_noData_stdValue * std_factor)
    high_border = image_array_noData_meanValue + ( image_array_noData_stdValue * std_factor)
    
    

    image_array_str = imageStretch(img_array, low_border, high_border)
    return image_array_str
        
# ----------------------------------------------------------------------------------------------------------------------


def image_percentile_clip(img_array, low_clip, high_clip):
    """Image Percentil Clipping and Stretching."""
    
    img_array = img_array.astype(float)
    
    image_array_noData = np.extract(img_array>0,img_array)
    low_border  = np.percentile(image_array_noData,low_clip)
    high_border = np.percentile(image_array_noData,high_clip)
    
    image_array_str = imageStretch(img_array, low_border, high_border)
    return image_array_str    
    
# ----------------------------------------------------------------------------------------------------------------------


def adjust_log(img_array, gain=1, inv=False, borders=[0,255]):
    """Performs Logarithmic correction on the input image.
       This function transforms the input image pixelwise according to the equation O = gain*log(1 + I) after scaling
       each pixel to the range 0 to 1. For inverse logarithmic correction, the equation is O = gain*(2**I - 1)."""
    
    img_array = exposure.adjust_log(img_array, gain, inv)
    img_array = array_normalisation(img_array, borders[0], borders[1])    
    return img_array

# ----------------------------------------------------------------------------------------------------------------------


def adjust_sigmoid(img_array, cutoff=0, gain=2, inv=False, borders=[0,255]):
    """Performs Sigmoid Correction on the input image.
       Also known as Contrast Adjustment. This function transforms the input image pixelwise according to the equation
       0 = 1/(1 + exp*(gain*(cutoff - I))) after scaling each pixel to the range 0 to 1.
       Rescaling to 0,255 by default."""
    
    img_array = exposure.adjust_sigmoid(img_array, cutoff, gain, inv)
    img_array = array_normalisation(img_array, borders[0], borders[1])
    return img_array
