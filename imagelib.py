# This is a script that takes as input 
# - the name of an image
# - the name of a catalog
"""
need simple view script for asdf files: 
e.g.,  view_asdf_image.py <image_name>
eventually will need slurm-like system to generate few E2/E3 images
another issue I forgot to mention on call:
 to modify exposure time , need to tweak MA_table parameter, 
 but according to Tyler the Nexus env is out of date for updating MA 
 parameter and I would need to install updated romanIsim package myself 
 as indicated in screen shot below. Wouldn’t it make more sense to 
 update Nexus env rather than asking each user to update their env

 jupytext --to notebook your_script_name.py

"""
import argparse
import numpy as np
import os 
import copy

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import vstack
from astropy.table import QTable
from astropy import visualization
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS


from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colormaps as cm


import roman_datamodels as rdm
import asdf
#import s3fs
#import time
#import nbformat as nbf


def ra2pix(catalog="", image="", save=False, local=True, overwrite=True):
    """Translate catalog ra,dec to pixel locations contained in a given image

    Inputs
    ------

    local : bool
        True -> regular posix file storage <- just this right now
        False -> s3 object file storage

    catalog : str
        Name of the catalog file, this can be the full FOV coverage catalog
        IF you want to save a new catalog with just the sources
        that overlap the image that is provided, use save=True
        Catalog should be in a format astropy can read, assummed ecsv right now
        and have ra and dec columns in decimal degrees

    image : str
        The name of the image to use

    save : bool
        Set to true if you would like a smaller catalog saved
        that only contains the sources that overlap the image given
        The filename will be a hash of hte catalog  and image
        Right now, the catalog is saved as an ecsv

    overwrite : bool
        overwrite the output catalog if save=True

    Returns
    -------
    An astropy table object with the selected catalog objects

    """
    if save:
        smallcat_name = image.split(".")[0] + "_catalog.ecsv"
        if os.path.isfile(smallcat_name):
            if not overwrite:
                raise IOError(f"{smallcat_name} already exists")
            else:
                os.remove(smallcat_name)

    if not os.access(image,os.R_OK):
        raise IOError(f"Can't' find the image: {image} ")

    if not os.access(catalog, os.R_OK):
        raise IOError(f"Can't find the catalog file {catalog}")

    

    #open up the image into a datamodel and grab the wcs transform
    image_model = rdm.open(image)
    
    #you can get the forward and backwards transform, or use the object and .inverse
    #world2pix = image_model.meta.wcs.get_transform('world','detector')
    #pix2world = image_model.meta.wcs.get_transform('detector','world')

    # min and max bounding square in pixels
    #xmin,xmax= image_model.meta.wcs.bounding_box['x0']
    #ymin,ymax = image_model.meta.wcs.bounding_box['x1']

    # get the center of the image
    #sky_center=image_model.meta.wcs.pixel_to_world(xcenter, ycenter)
    #xcen,ycen = image_model.meta.wcs.invert(sky_center.ra, sky_center.dec)
    #footprint=image_model.meta.wcs.footprint() #footprint on the sky

    fullcat = Table.read(catalog, format='ascii.ecsv')
    #fullcat = QTable.read(cat_table, format='ascii.ecsv')
    mask = image_model.meta.wcs.in_image(fullcat['ra'],fullcat['dec'])
    selected_catalog = fullcat[mask]
    x,y = image_model.meta.wcs.invert(selected_catalog['ra'], selected_catalog['dec'])
    selected_catalog['x']=x
    selected_catalog['y']=y

    if save:
        selected_catalog.write(smallcat_name)
    return selected_catalog


def mkfigure(image="", plotname="plot.png",ext='SCI',
             save=True, cmap="gray",
             xsize=8, ysize=8, dpi=100,
             centerx=0,centery=0,
             delta=0):

    """Make a figure using matplotlib, no overlays here

        Inputs
        ------
        image: str
            The name of an image to plot

        ext : str
            if fits, the extension for plotting

        xsize : int
            The size of the plot in x inches

        ysize : int
            The size of the plot in y inches

        dpi : int
            The dots per inch for the plot

        centerx : int
            center the plot on this pixel in x

        centery : int
            center the plot on this pixel in y

        delta : int
            make the plot this many pixels*2 big

        cmap : str
            color map in matplotlib to use, gray is default
            cividis and viridis are also perceptually uniform

        Returns
        -------
        None

    """
    #assume asdf, then fits
    isfits=False
    try:
        file = rdm.open(image)
        if ext=="SCI":
            if (len(file.data.shape) > 2):
                if (file.data).shape[0] > 1:
                    data = file.data[0] #get the final image in ramp
            else:
                data = file.data # get the sci
        if ext=="ERR":
            data = file.err 
        if ext=="DQ":
            data=file.dq
        wcs = file.meta.wcs
    except TypeError:
        file = fits.open(image)
        try:
            data = file[ext].data #fits data
            wcs = WCS(file[ext].header)
            isfits=True
        except KeyError:
            print(f"{ext} not found in fits image")
        except UnboundLocalError:
            data = file[1].data
            wcs = None 
            isfits = True
    except ValueError:
        file=asdf.open(image)
        data = file.tree['output']  # test file made by romanisim
        wcs = None 
        isfits = False


    
    colormap = cm[cmap]
    norm = visualization.ImageNormalize(data,
                                    interval=visualization.ZScaleInterval(contrast=0.5),
                                    stretch=visualization.AsinhStretch(a=1))

    if wcs:   
        fig, ax = plt.subplots(figsize=(ysize, xsize), subplot_kw={'projection':wcs})
    else:
        fig, ax = plt.subplots(figsize=(ysize, xsize))

    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(dpi)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.imshow(data, norm=norm, cmap=colormap, aspect="equal")
    if wcs:
        ax.coords.grid(linestyle='dotted', color='white', alpha=0.5)
        lon = ax.coords[0]
        lat = ax.coords[1]
        lon.set_axislabel('Right Ascension')
        lat.set_axislabel('Declination')
        lon.set_major_formatter('d.ddd')
        lat.set_major_formatter('d.ddd')
        if (delta==0 or delta>2000):
            lon.set_ticks(spacing=2 * u.arcminute)
            lat.set_ticks(spacing=2 * u.arcminute)
        else:
            lon.set_ticks(spacing=40 * u.arcsecond)
            lat.set_ticks(spacing=40 * u.arcsecond)
    ax.set_title(f'{image} {ext}')
    if delta !=0:
        ax.set_xlim(centerx-delta, centerx+delta)
        ax.set_ylim(centery-delta, centery+delta)
    canvas.draw()
    canvas.print_png(plotname)
    file.close()



def create_regions(catalog, radius=20):
    """create a ds9 style regions given an astropy table
    
    Inputs
    ------

    catalog : object
        astropy table object

    radius : int
        radius in pixels for circle overlay


    Notes
    -----
    catalog input must have x,y columns

    Returns
    -------
    a list of regions that can be saved to file
    and loaded into ds9

    """
    regions=[]
    for x,y in zip(catalog['x'],catalog['y']):
        regions.append(f"circle {x} {y} {radius}")
    return regions


def plot_image(image="r0003201001001001004_0001_wfi01_f106_cal.asdf",
               catname="", save=True, cmap="gray",ext='SCI',
               xsize=8, ysize=8, dpi=100,plotname="",
               centerx=0,centery=0, save_regions=False,
               delta=0, radius=50):
    """Show an image in matplotlib view with catalog overlays

        Inputs
        ------
        image: str
            The name of an image to plot

        catname: str
            The name of the catalog to use

        save: bool
            Set to True to save a png of the plot to disk instead

        xsize : int
            The size of the plot in x inches

        ysize : int
            The size of the plot in y inches

        dpi : int
            The dots per inch for the plot

        centerx : int
            center the plot on this pixel in x

        centery : int
            center the plot on this pixel in y

        delta : int
            make the plot this many pixels*2 big

        cmap : str
            color map in matplotlib to use

        Returns
        -------
        None

        Notes
        -----
        #Exmaple for s3 data that has been provided
        #asdf_dir_uri = 's3://stpubdata/roman/nexus/soc_simulations/tutorial_data/'
        #if the data is in an S3 bucket us this
        #fs = s3fs.S3FileSystem(anon=True)
        #asdf_file_uri = asdf_dir_uri + 'r0003201001001001004_0001_wfi01_f106_cal.asdf'
        #file = rdm.open(fs.open(asdf_file_uri, 'rb'))

        #if the data is in the nexus as part of the EFS storage
        filename="/home/sosey/teams/sn411/SNPIT_VISIT607050033_WFI01_F129_cal.asdf"
        # s3 file open with roman datamodels
        #file = rdm.open(fs.open(filename, 'rb'))
        #open with roman datamodels in the nexus file system
        file=rdm.open(filename)
    """
    #assume asdf, then fits
    isfits=False
    try:
        file = rdm.open(image)
        if ext=="SCI":
            data = file.data # get the sci
        if ext=="ERR":
            data = file.err 
        if ext=="DQ":
            data=file.dq
        wcs = file.meta.wcs

    except TypeError:
        file = fits.open(image)
        try:
            data = file[ext].data #fits data 
            wcs = WCS(file[ext].header)
            isfits=True
        except KeyError:
            print(f"{ext} not found in fits image")
    

    
    colormap = cm[cmap]
    norm = visualization.ImageNormalize(data,
                                    interval=visualization.ZScaleInterval(contrast=0.5),
                                    stretch=visualization.AsinhStretch(a=1))

    try:
        fig, ax = plt.subplots(figsize=(ysize, xsize), subplot_kw={'projection':wcs})        
    except NotImplementedError:
        print(f"Make your asdf file ignoring SIP for now, \nthen try again, one of the models isn't seperable yet")

    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(dpi)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.imshow(data, norm=norm, cmap=colormap, aspect="equal")
    ax.coords.grid(linestyle='dotted', color='white', alpha=0.5)
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_axislabel('Right Ascension')
    lat.set_axislabel('Declination')
    lon.set_major_formatter('d.ddd')
    lat.set_major_formatter('d.ddd')
    if (delta==0 or delta>2000):
        lon.set_ticks(spacing=2 * u.arcminute)
        lat.set_ticks(spacing=2 * u.arcminute)
    else:
        lon.set_ticks(spacing=40 * u.arcsecond)
        lat.set_ticks(spacing=40 * u.arcsecond)
    ax.set_title(f'{image} {ext}')
    if delta !=0:
        ax.set_xlim(centerx-delta, centerx+delta)
        ax.set_ylim(centery-delta, centery+delta)

    catalog = ra2pix(catname, image, save=True)
    if save_regions:
        regions=create_regions(catalog, radius=radius)
        with open(image.split(".")[0]+".reg", "w") as file:
            file.write("\n".join(regions))

    for obj in catalog:
        patch=patches.Circle((obj['x'], obj['y']),radius,
                             fill=False,edgecolor='white', linewidth=1)
        ax.add_patch(patch)
        
    canvas.draw()
    if not plotname:
        plotname = image.split(".")[0]+"_cat.png"
    canvas.print_png(plotname)
    file.close()


def create_gaia_catalog(ra=80., dec=30., radius=1, obs_time = '2026-10-31T00:00:00',
                        filename='gaia_catalog.ecsv', overwrite=True):
    """Create a catalog
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    radius : float
        Search radius in degrees
    obs_time : str
        observation time

    """
    try:
        from romanisim import (gaia, bandpass, catalog, log, wcs)
    except ImportError:
        print(f"You need romanisim installed to create gaia catalogs")

    query = f'SELECT * FROM gaiadr3.gaia_source WHERE distance({ra}, {dec}, ra, dec) < {radius}'
    job = Gaia.launch_job_async(query)
    result = job.get_results()
    gaia_catalog = gaia.gaia2romanisimcat(result,
                    Time(obs_time),
                    fluxfields=set(bandpass.galsim2roman_bandpass.values()))


    # Clean out bad sources from Gaia catalog:
    bandpass = [f for f in gaia_catalog.dtype.names if f[0] == 'F']
    bad = np.zeros(len(gaia_catalog), dtype='bool')
    for b in bandpass:
         bad = ~np.isfinite(gaia_catalog[b])
         if hasattr(gaia_catalog[b], 'mask'):
              bad |= gaia_catalog[b].mask
         gaia_catalog = gaia_catalog[~bad]

    gaia_catalog = gaia_catalog[np.isfinite(gaia_catalog['ra'])]

    gaia_catalog.write(filename, overwrite=overwrite)
    return filename


# if __name__ == "__main__":
#     """ in progress
#     sim_image: simulate image with somanisim
#     plot_image: plot image with object overlay using matplotlib
#     save_notebook: save plots to notebook
#     create_gaia_catalog: create a gaia catalog
#     ra2pix: convert ra/dec in catalog to pixels in image

#     #example_image="r0003201001001001004_0001_wfi01_f106_cal.asdf"

#     """
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("image", help="Name of the asdf image", type=str)
#     parser.add_argument("-h","--help",help="print arguments")
#     parser.add_argument("--mpl", help="Save matplotlib catalog overlay image",
#                         action="store_true", default=False)
#     parser.add_argument("--cat", help="name of the catalog to use", type=str)
#     parser.add_arguemt("--reg", help="store ds9 regions file", 
#                        default=False, action="store_true")
#     parser.add_argment("--dpi", help="plot dpi", type=int, default=100)
#     parser.add_argument("--radius", help="radious of object marker", type=int, default=50)
#     parser.add_argument("--ra2pix", help="save catalog of overlapping sources",
#                         action="store_true")
#     parser.add_argument("--delta", help="pixel area to plot around center", type=int,default=0)
#     args = parser.parse_args()

#     if args.mpl:
#         if not args.cat:
#             raise ValueError("Need to provide a catalog name for overplotting")
#         else:
#             plot_image(args.image,catname=args.cat, 
#                save=True, cmap="gray",
#                dpi=args.dpi, save_regions=args.reg,
#                delta=args.delta, radius=args.radius)

#     if args.ra2pix:
#         ra2pix(catalog=args.cat, image=args.image, save=True,
#                local=True, overwrite=True)
