##### DEFINE PRODUCT TO BE PLOTTED AND SET UP THE VARIABLES & NAMES

import numpy as np
import os
import fnmatch
import matplotlib as plt
import matplotlib
import matplotlib.pylab as plt
from matplotlib.transforms import Bbox
from skimage import exposure, transform
import glymur
'''
------------------------------------------------------------------------------------------------------------------------
Functions for Image Processing and Enhancement
------------------------------------------------------------------------------------------------------------------------
'''
def get_bands(prefix):
    '''import image data (bands) from chosen product'''

    prod_name = prefix[-66:-6]
    prod_date = prefix[-55:-47]

    colors = [None]*3

    print('get all the required names/strings of the product... ')

    for item in os.listdir(prefix):
        if os.path.isdir(os.path.join(prefix, item)):
            if item == 'GRANULE':
                granule = os.path.join(prefix,item)
                for item1 in os.listdir(granule):
                    subgranule = os.path.join(granule,item1)
                    tileID = subgranule[-30:-24]
                    for item2 in os.listdir(subgranule):
                        if item2 == 'IMG_DATA':
                            img_data = os.path.join(subgranule, item2+'/')
                            for item3 in os.listdir(img_data):
                            
                                if fnmatch.fnmatch(item3, '*B04*'):
                                    colors[2] = item3
                                if fnmatch.fnmatch(item3, '*B03*'):
                                    colors[1] = item3
                                if fnmatch.fnmatch(item3, '*B02*'):
                                    colors[0] = item3

    name_out = tileID+'_'+prod_date
    print('Tile & Date: ',name_out)

    # the png will be saved under this name_out

    print('----------------------------------------------------------------------------------------------')

    ##### READ RGB VALUES OUT OF JP2000 FILES

    print('import RGB values and scale the array...')


    # bands are read in the order B04 B03 B02 for RGB
    ims = list( reversed( list( map( glymur.Jp2k, [ img_data + c for c in colors ]))))
    scale = 8 # 8 is a good compromise between speed and detail


    raw_bands = [i[::scale, ::scale] for i in ims]

    #rgb, nx, ny = np.shape(raw_bands)
    global nx
    global ny
    global rgb


    rgb, nx, ny = np.shape(raw_bands)
    
    return raw_bands, name_out

    print('----------------------------------------------------------------------------------------------')

def get_coordinates(prefix):
    '''extract tile coordinates from meta data xml file '''

    import xml.etree.ElementTree as et

    print('get lat/lon values of image corners..........')

    tree = et.parse(prefix+'MTD_MSIL1C.xml')
    root = tree.getroot()
    a = root.find('.//EXT_POS_LIST')
    x = [float(i) for i in a.text.split()]     # convert values from string to float
    del(x[-4:])                                # delete obsolete values at end of array
    lat = x[0::2]
    lon = x[1::2]
    
    return lat, lon

    print('----------------------------------------------------------------------------------------------')
    
def image_processing(raw_bands):
    '''using (all the) methods in Andreas' image_processing_short.py script'''

    import image_processing_short as ips
    

    new_bands = np.ones((rgb+1,nx+2,ny+2))


    #### IMAGE PROCESSING

    # image stretching

    bands_array = np.asarray(raw_bands)
    bands = (ips.imageStretch(bands_array, 0, 4000))
    bands = bands/255.1

    for i in range(3):
        placeholder = np.vstack([ np.ones(nx), bands[i] ])
        placeholder = np.vstack([ placeholder, np.ones(ny) ])
        placeholder = np.column_stack([ placeholder, np.ones(ny+2) ])
        placeholder = np.column_stack([ np.ones(ny+2), placeholder ])
        new_bands[i] = placeholder

    arr1 = np.dstack([new_bands[0], new_bands[1], new_bands[2], new_bands[3]])

    # 95% percentile clipping

    bands = (ips.image_percentile_clip(bands_array, 5, 95))
    bands = bands/255.1

    for i in range(3):
        placeholder = np.vstack([ np.ones(nx), bands[i] ])
        placeholder = np.vstack([ placeholder, np.ones(ny) ])
        placeholder = np.column_stack([ placeholder, np.ones(ny+2) ])
        placeholder = np.column_stack([ np.ones(ny+2), placeholder ])
        new_bands[i] = placeholder

    arr2 = np.dstack([new_bands[0], new_bands[1], new_bands[2], new_bands[3]])


    # the bands matrix is of type 'list'

    return arr1, arr2
    
    print('----------------------------------------------------------------------------------------------')

def create_grid(lat, lon):
    '''initialize different grids and zip them in order to create the cKDTree'''

    # whereas lam & phi (small letters) denote the coordinates for each pixel in the original image


    for i in range(4):
        print(lat[i],lon[i])


    print('set up coordinate system of tilted image...')

    # the full-size image consists of 10'980 x 10'980 datapoints, i.e. pixel. due to memory
    # restrictions, the number of pixels is reduced to 10'980 / x where x = 2^n.

    #n = np.shape(new_bands)[1]
    #m = np.shape(new_bands)[2]

    phi_SWtoNW = np.linspace(lat[3], lat[0], num=nx)
    phi_SEtoNE = np.linspace(lat[2], lat[1], num=nx)
    lam_SWtoNW = np.linspace(lon[3], lon[0], num=ny)
    lam_SEtoNE = np.linspace(lon[2], lon[1], num=ny)

    lam = np.ones((nx,ny))
    phi = np.ones((nx,ny))

    a = 0

    for i in range(nx):
        a = a + 1
        phi[:][-a] = np.linspace(phi_SWtoNW[i], phi_SEtoNE[i], num=ny)
        lam[:][-a] = np.linspace(lam_SWtoNW[i], lam_SEtoNE[i], num=nx)


    print('----------------------------------------------------------------------------------------------')

    print('set up untilted coordinate system ...')


    # set up the untilted coordinate system

    Lam = np.ones((nx,ny))
    Phi = np.ones((nx,ny))

    phiphi = np.linspace(np.min(phi), np.max(phi), nx)          # lam = lon, phi = lat

    a = 0

    for i in range(nx):
        a = a + 1
        Phi[:][-a] = phiphi[i]
        Lam[:][-a] = np.linspace(np.min(lam), np.max(lam), nx)


    print('----------------------------------------------------------------------------------------------')

    print('zip Lam/Phi and lam/phi matrizes together for the KD Tree...')

    goal_img = zip(Phi.ravel(), Lam.ravel())
    orig_img = zip(phi.ravel(), lam.ravel())

    print('----------------------------------------------------------------------------------------------')

    print('create index matrix for the extraction of the RGB values of the unzipped matrix')
    # create matrix with indizes of the original Phi/Lam & phi/lam matrizes:

    indizes = np.zeros((nx*ny,2), dtype='i')

    i = 0
    j = 0
    l = 0
    running = True

    while running:
    
        indizes[l] = [i, j]
    
        if j == nx-1:
            i = i + 1
            j = -1
            
        j = j + 1
        l = l + 1
    
        if i == nx:
            running = False
        
    print('----------------------------------------------------------------------------------------------')

    ##### SET UP KD-TREE OF THE TILTED COORDINATE SYSTEM AND FIND NEAREST NEIGHBORS

    from scipy import spatial

    print('create KD Tree of lam/phi data pairs...')

    import time
      
    start = time.time()
    tree = spatial.cKDTree(orig_img)
    end = time.time()
    print(end - start)

    return tree, goal_img, indizes
    
    print('----------------------------------------------------------------------------------------------')

    print('iterating over the whole goal image to find nearest neighbors and assign RGB values to it...')
    print

def find_neighbors(tree, goal_img):
    '''project the image on the new grid (i.e. "rotate" it) by finding the nearest neighbor for each coordinate pair. Then assign the RGBA values to the corresponding pixel'''
    
    import time
    # two methods of image processing requires two resulting arrays

    num = nx*ny                                 #number of iterations for statistics of periods
    global periods
    periods = np.empty((num,2))
    queries = np.empty(num, dtype=int)

    for l in range(num):

        start_query = time.time()
        query = tree.query(goal_img[l])
        queries[l] = query[1]
        end_query = time.time()
        periods[l,0] = end_query - start_query
                

    print('times [s] to find nearest neighbor: mean of all queries = ', np.mean(periods[:,0]), 'total time = ', np.sum(periods[:,0]))
    return queries

def assign_pxl_values(arr1, arr2, queries, indizes):

    import time
    num = nx*ny    
    new_arr1 = np.ones((ny,nx,4))
    new_arr2 = np.ones((ny,nx,4))
    a = 0
    b = 0
    periods = np.empty((num,2))
    
    for l in range(num):
        
        start_rest = time.time()
        if all(arr1[indizes[queries[l], 0], indizes[queries[l], 1]][x] ==1. for x in range(4)):
            new_arr1[a][b] = 0
            new_arr2[a][b] = 0
        else:
            new_arr1[a][b] = arr1[indizes[queries[l], 0], indizes[queries[l], 1]]
            new_arr2[a][b] = arr2[indizes[queries[l], 0], indizes[queries[l], 1]]

        b = b + 1
    
        if l == ((a+1)*ny)-1:
            a = a + 1
            b = 0

        end_rest = time.time()    
        periods[l,1] = end_rest - start_rest


    # statistics


    print('times [s] to assign RGB values: mean of all assignments =  ', np.mean(periods[:,1]), 'total time = ', np.sum(periods[:,1]))

    return new_arr1, new_arr2
    print('----------------------------------------------------------------------------------------------')

def plot_L1C(new_arr1, new_arr2, name_out):
    '''acutal plot section where the final products are saved as .png file.'''
    print('plotting section including saving the figure...')

    # first time is for plotting the figure in the notebook if show_plot = True:

    import matplotlib.pylab as plt
    import time
    
    show_plot = False

    if show_plot:

        plt.rcParams['figure.figsize'] = (nx/50, ny/50) 

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        plt.axis('off')
        plt.imshow(new_arr2)
        plt.show()

    # second time is for saving the picture as a png file:

    plt.rcParams['figure.figsize'] = (nx/50, ny/50)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.imshow(new_arr1)

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())    
    path_out = './'
    plt.savefig(path_out+''+name_out+'_1.png', transparent=True, bbox_inches=extent)

    plt.rcParams['figure.figsize'] = (nx/50, ny/50)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.imshow(new_arr2)
    
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    plt.savefig(path_out+''+name_out+'_2.png', transparent=True, bbox_inches=extent)



    
    print('----------------------------------------------------------------------------------------------')
