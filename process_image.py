import image_processing_short as ips
import S2png
import numpy as np
import holoviews as hv

hv.notebook_extension()

'''
------------------------------------------------------------------------------------------------------------------------
Functions for Dynamic Map creating and Image Processing. The module image_processing_short was written by Anreas B. G. Baumann (ESRIN, Frascati).
------------------------------------------------------------------------------------------------------------------------
'''

def create_dmap(raw_bands, method='clip'):
    '''This function creates a Dynamic Map after HoloViews with the raw band data being the only input. '''
    
    global dummy
    dummy = raw_bands
    
    if method=='clip':
    
        dmap = hv.DynamicMap(clip, kdims=['lower', 'upper'])
    
        n = 11
        upper_limit = np.linspace(2000, 5000, num=n)
        lower_limit = np.linspace(0, 1000, num=n)

        return dmap.redim.values(lower=lower_limit, upper=upper_limit)
    
    if method=='std':
    
        dmap = hv.DynamicMap(stdclip, kdims=['stdfactor'])
    
        n = 10
        factor = np.linspace(1, 10, num=n)

        return dmap.redim.values(stdfactor=factor)
    
    if method=='percentile':
        
        dmap = hv.DynamicMap(percentile, kdims=['lower', 'upper'])
        
        n = 11
        upper_limit = np.linspace(90, 100, num=n)
        lower_limit = np.linspace(0, 10, num=n)
        
        return dmap.redim.values(lower=lower_limit, upper=upper_limit)
    
    if method=='adjlog':
            
        dmap = hv.DynamicMap(adjlog, kdims=['gain', 'inv'])
        
        n = 10
        gain_val = np.linspace(10, 100, num=n)
        inv_bool = ['False', 'True']
        
        return dmap.redim.values(gain=gain_val, inv=inv_bool)
     
    if method=='adjsig':
        
        dmap = hv.DynamicMap(adjsig, kdims=['cutoff', 'gain', 'inv'])
        
        n=10
        cutoff_val = np.linspace(0, 1, num=11)
        gain_val = np.linspace(10, 100, num=n)
        inv_bool = ['False', 'True']
        
        return dmap.redim.values(cutoff=cutoff_val, gain=gain_val, inv=inv_bool)
    
    if method=='clahe':
        
        dmap = hv.DynamicMap(clahe, kdims=['factor', 'clip_limit'])
        
        n = 10
        factor_val = [3] #np.linspace(1, 10, num=n)
        clip_lim_val= [0.05] #np.linspace(0.01, 0.1, num=n)
        
        return dmap.redim.values(factor=factor_val, clip_limit=clip_lim_val)

#---------------------------------------------------------------------------------------------------------------------------
# supporting functions for the Dynamic Map creation functions
#---------------------------------------------------------------------------------------------------------------------------

def clip(lower_lim, upper_lim):
    '''this functions cuts off the values below/above the lower/upper limit, respectively, and then stretches the remaining values on [0, 255].'''
    
    bands_array = np.asarray(dummy)
    
    bands_proc = (ips.imageStretch(bands_array, lower_lim, upper_lim))
    bands_proc = bands_proc/255.1
    arr = np.dstack(bands_proc)
    
    return hv.RGB(arr)

def stdclip(factor):
    '''only values in the interval [mean - a x std, mean + a x std] are considered. The factor a is modifiable.'''
    
    bands_array = np.asarray(dummy)
    
    bands_proc = (ips.image_meanStd_clip(bands_array, std_factor=factor))
    bands_proc = bands_proc/255.1
    arr = np.dstack(bands_proc)
    
    return hv.RGB(arr)

def percentile(lower_lim, upper_lim):
    '''a certain percentage interval is used to cut off the outlying values.'''
    
    bands_array = np.asarray(dummy)
    
    bands_proc = (ips.image_percentile_clip(bands_array, lower_lim, upper_lim))
    bands_proc = bands_proc/255.1
    arr = np.dstack(bands_proc)
    
    return hv.RGB(arr)

def adjlog(gain, inv):
    '''performs logarithmic correction on the image according to the equation 0 = gain x log(I + 1) (for inverse log correction 0 = gain x (2^I - 1)  after scaling the values to [0, 1].'''
    
    bands_array = np.asarray(dummy)
    
    bands_proc = (ips.adjust_log(bands_array, gain, inv))
    bands_proc = bands_proc/255.1
    arr = np.dstack(bands_proc)
    
    return hv.RGB(arr)

def adjsig(cutoff, gain, inv):
    '''Sigmoid Correction, also known as Contrast Adjustment, according to 0 = 1/(1 + exp(gain x (cutoff - I))) after scaling to [0, 1].'''
    
    bands_array = np.asarray(dummy)

    bands_proc = (ips.adjust_sigmoid(bands_array, cutoff, gain, inv))
    bands_proc = bands_proc/255.1
    arr = np.dstack(bands_proc)
    
    return hv.RGB(arr)




