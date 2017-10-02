##### DEFINE PRODUCT TO BE PLOTTED AND SET UP THE VARIABLES & NAMES

import numpy as np
import os
import fnmatch
import matplotlib as plt
import matplotlib
import matplotlib.pylab as plt
from matplotlib.transforms import Bbox
from skimage import exposure, transform
from mpl_toolkits.basemap import Basemap
import glymur
'''
------------------------------------------------------------------------------------------------------------------------
Functions for Image Processing and Enhancement
------------------------------------------------------------------------------------------------------------------------
'''

class S2processing:
    
    def __init__(self, productName, outputName=True):

        '''import image data (bands) from chosen product'''
        
        self.prefix = productName

        prod_name = self.prefix[-66:-6]
        prod_date = self.prefix[-55:-47]

        colors = [None]*3

        for item in os.listdir(self.prefix):
            if os.path.isdir(os.path.join(self.prefix, item)):
                if item == 'GRANULE':
                    granule = os.path.join(self.prefix,item)
                    for item1 in os.listdir(granule):
                        subgranule = os.path.join(granule,item1)
                        tileID = subgranule[-30:-24]
                        for item2 in os.listdir(subgranule):
                            if item2 == 'IMG_DATA':
                                img_data = os.path.join(subgranule, item2+'\\')
                                for item3 in os.listdir(img_data):

                                    if fnmatch.fnmatch(item3, '*B04*'):
                                        colors[2] = item3
                                    if fnmatch.fnmatch(item3, '*B03*'):
                                        colors[1] = item3
                                    if fnmatch.fnmatch(item3, '*B02*'):
                                        colors[0] = item3

        if outputName:
            self.nameOut = tileID+'_'+prod_date
        else:
            self.nameOut = outputName
        
        # the png will be saved under this nameOut

        
        ##### READ RGB VALUES OUT OF JP2000 FILES

        print('import RGB values and scale down the array by a factor of 8.')


        # bands are read in the order B04 B03 B02 for RGB
        ims = list( reversed( list( map( glymur.Jp2k, [ img_data + c for c in colors ]))))
        scale = 8 # 8 is a good compromise between speed and detail

        raw_bands = [i[::scale, ::scale] for i in ims]

        #rgb, nx, ny = np.shape(raw_bands)

        self.rgb, self.nx, self.ny = np.shape(raw_bands)
        
        self.rgb = np.asarray(raw_bands)
   
    ##########################################################################################
    ##########################################################################################

    def getCoordinates(self):
        '''extract tile coordinates from meta data xml file '''

        import xml.etree.ElementTree as et

        print('get lat/lon values of image corners.')

        tree = et.parse(self.prefix+'MTD_MSIL1C.xml')
        root = tree.getroot()
        a = root.find('.//EXT_POS_LIST')
        x = [float(i) for i in a.text.split()]     # convert values from string to float
        del(x[-4:])                                # delete obsolete values at end of array
        self.lat = x[0::2]
        self.lon = x[1::2]
   
        lat, lon = self.lat, self.lon


        print('set up coordinate system of tilted image.')

        # the full-size image consists of 10'980 x 10'980 datapoints, i.e. pixel. due to memory
        # restrictions, the number of pixels is reduced to 10'980 / x where x = 2^n.


        phi_SWtoNW = np.linspace(lat[3], lat[0], num=self.nx)
        phi_SEtoNE = np.linspace(lat[2], lat[1], num=self.nx)
        lam_SWtoNW = np.linspace(lon[3], lon[0], num=self.ny)
        lam_SEtoNE = np.linspace(lon[2], lon[1], num=self.ny)

        self.lam = np.ones((self.nx,self.ny))
        self.phi = np.ones((self.nx,self.ny))

        a = 0

        for i in range(self.nx):
            a = a + 1
            self.phi[:][-a] = np.linspace(phi_SWtoNW[i], phi_SEtoNE[i], num=self.ny)
            self.lam[:][-a] = np.linspace(lam_SWtoNW[i], lam_SEtoNE[i], num=self.nx)

    ##########################################################################################
    ##########################################################################################
    
    def createBasemap(self):
    
        lonCorners = self.getCorners(self.lam)
        latCorners = self.getCorners(self.phi)
        
        self.basemap = Basemap(projection = 'merc',
                               llcrnrlat = latCorners.min(),
                               urcrnrlat = latCorners.max(),
                               llcrnrlon = lonCorners.min(),
                               urcrnrlon = lonCorners.max(),
                               resolution = 'i')
        
        self.xCorners, self.yCorners = self.basemap(lonCorners, latCorners)
        
    def getCorners(self, centers):
        
        one = centers[:-1,:]
        two = centers[1:, :]
        dl = (two - one) / 2.
        one = one - dl
        two = two + dl
        stepOne = np.zeros((centers.shape[0] + 1, centers.shape[1]))
        stepOne[:-2, :] = one
        stepOne[-2:, :] = two[-2:, :]
        one = stepOne[:, :-1]
        two = stepOne[:, 1:]
        d2 = (two - one) / 2.
        one = one - d2
        two = two + d2
        stepTwo = np.zeros((centers.shape[0] + 1, centers.shape[1] + 1))
        stepTwo[:, :-2] = one
        stepTwo[:, -2:] = two[:, -2:]
        
        return stepTwo
    
    def savePNG(self, array=None, nameOutNew=None):
        '''acutal plot section where the final products are saved as .png file.'''
        plt.clf()
        plt.close('all')
        
        rgb0 = np.empty(self.rgb.shape)
        for i in range(3):
            rgb0[i] = array[:,:,i]
        rgb = rgb0.T
        
        colorTuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/rgb.max()

        plt.rcParams['figure.figsize'] = (50, 50)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        
        plt.axis('off')
        self.cm = self.basemap.pcolormesh(self.xCorners,
                                          self.yCorners,
                                          rgb0[1,:,:],
                                          color=colorTuple,
                                          linewidth = 0)
        self.cm.set_array(None)
        
        try:
            self.nameOutNew = nameOutNew
            plt.savefig(nameOutNew+'.png', transparent=True, bbox_inches=extent, origin='lower')
        except:
            plt.savefig(nameOut+'.png', transparent=True, bbox_inches=extent, origin='lower')
        
        plt.clf()
        plt.close('all')
        
    def plotImg(self, array=None):
        
        plt.clf()
        plt.close('all')
        
        rgb0 = np.empty(self.rgb.shape)
        for i in range(3):
            rgb0[i] = array[:,:,i]
        rgb = rgb0.T
        
        self.basemap.drawmapboundary()
        self.basemap.drawcoastlines()
        self.basemap.drawparallels(np.arange(-80., 80., 5.))
        self.basemap.drawmeridians(np.arange(-180., 181., 5.))
        
        colorTuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/rgb.max()

        plt.rcParams['figure.figsize'] = (50, 50)
        
        self.cm = self.basemap.pcolormesh(self.xCorners,
                                          self.yCorners,
                                          rgb0[1,:,:],
                                          color=colorTuple,
                                          linewidth = 0)
        self.cm.set_array(None)
        plt.show()
        plt.close('all')
        plt.clf()
        
    def plotWebMap(self):
        
        from ipyleaflet import Map, ImageOverlay
        
        nx = self.nx
        ny = self.ny
        
        center = [(self.phi.max()+self.phi.min())/2.,
                  (self.lam.max()+self.lam.max())/2.]
        self.map = Map(center=center,
                       zoom=8,
                       width='100%',
                       heigth=6000)
        
        imgurl = self.nameOutNew + '.png'
        self.map.add_layer(ImageOverlay(url=imgurl,
                                        bounds=[[self.phi.min(), self.lam.min()],
                                                [self.phi.max(), self.lam.max()]]))
        
        return self.map
        
        
        
        
        
        