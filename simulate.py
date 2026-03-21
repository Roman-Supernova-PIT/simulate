#
# MA table:
#  https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-multiaccum-ma-tables
#
# Questions;
#   - how to specify exposure time ? Through the MA Table that is selected
#   - is lensing applied ? no
#   - is Galactic extinction applied ?
#   - how to retrieve subset of objects overlaid on image ? via wcs of image and input catalogs
#   - slurm-like system to distribute jobs?  CPU limit per user?  Not on nexus
#   - is there a dumb-fast option to quickly test infrastructure ? no
#
#
# default image with romanisim script
#romanisim-make-image --radec 270.00 66.00 --level 1 --sca -1 --bandpass F158 --catalog gaia_catalog.ecsv --psftype stpsf --usecrds --ma_table_number 1001 --date 2027-06-01T00:00:00 --rng_seed 1 --drop-extra-dq r0000101001001001001_0001_{}_{bandpass}_uncal.asdf
#
# default spectroscopic image
#romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --psftype stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:10:00 --rng_seed 3 --drop-extra-dq r0000201001001001001_0001_{}_{bandpass}_uncal.asdf --pretend-spectral GRISM
# 
# prism image
#romanisim-make-image --radec 270.00 66.00 --level 1 --sca 1 --bandpass F158 --catalog gaia-270-66-2027-06-01.ecsv --psftype stpsf --usecrds --ma_table_number 109 --date 2027-06-01T00:30:00 --rng_seed 8 --drop-extra-dq r0000301001001001001_0001_{}_{bandpass}_uncal.asdf --pretend-spectral PRISM
# ===========================================================================

import os, sys
import logging, time, datetime, yaml, glob
import argparse
import numpy as np

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table, vstack, join, MaskedColumn
from astropy import units as u
from astropy.visualization import simple_norm
from astropy.modeling.functional_models import Sersic2D

import copy

import roman_datamodels as rdm
from roman_datamodels.datamodels import ImageModel, ScienceRawModel

try:
    from romanisim import l1, log, parameters, nonlinearity
    from romanisim import catalog as riscatalog, persistence, parameters
    from romanisim.image import inject_sources_into_l2
    import romanisim.image 
    from romanisim import gaia, log, wcs as rwcs, util, ris_make_utils as ris
    import romanisim.bandpass   
except ImportError:
    print(f"You need romanisim installed to create gaia catalogs and images")

import pysiaf
from dataclasses import dataclass
from typing import Union


import galsim
import galsim.roman

import asdf
from crds import client
from crds.client import api as crdsapi

import imagelib

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)6s %(message)s")


# ============================================================================
# define hard-coded inputs ... perhaps later will read from input config file

# these are in the config too
BAND_LIST_SNANASIM = [ 'R062-R','Z087-Z','Y106-Y','J129-J','H158-H','F184-F','K213-K' ]
BAND_LIST_HOSTLIB  = [ 'R_obs', 'Z_obs', 'Y_obs', 'J_obs', 'H_obs', 'F_obs', 'K_obs' ]
BAND_LIST_SOC      = [ 'F062',  'F087',  'F106',  'F129',  'F158',  'F184',  'F213' ]
BAND_LIST_INPUT    = [ b[0] for b in BAND_LIST_HOSTLIB ]  # single-char represenation for user input

# note that a_Sersic and b_Sersic are replaced with a0,b0 or a1,b1
HOSTLIB_COLUMNS = [ 'GALID', 'RA_GAL', 'DEC_GAL', 'a_rot',
                    'n_Sersic', 'a_Sersic', 'b_Sersic', 'w1_Sersic', 'MAGNIFICATION' ] + \
                    BAND_LIST_HOSTLIB

SOC_COLUMNS     = [ 'id',    'ra',     'dec',     'pa',
                    'n',     'a',      'b',     'w1', 'MAGNIFICATION' ] + \
                    BAND_LIST_SOC

TYPE_SERSIC = "SER"   # for galaxies
TYPE_PSF    = "PSF"   # for point sources (transients and stars

# keys in config input file
STAR_CATNAME_GAIA = "GAIA"
STAR_CATNAME_SYN  = "SYN"


# ===========

def read_config(input_config_file):
    path_expand = os.path.expandvars(input_config_file)
    logging.info(f"Reading YAML input from {path_expand}")
    with open(path_expand) as f:
        return yaml.safe_load(f.read())

# ---------------------------
def print_banner(banner, level=1):
    logging.info('')

    if level == 1:
        logging.info('# ================================================================')
    elif level == 2:
        logging.info('# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
    elif level == 9:
        logging.info('# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ')
        logging.info('# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ')

    logging.info(f"{banner}")



def get_sky_region(config):
    ra     = config['SKY_REGION']['RA_CEN']
    dec    = config['SKY_REGION']['DEC_CEN']
    radius = config['SKY_REGION']['RADIUS'] 
    roll   = config['SKY_REGION']['ROLL']
    return ra, dec,  radius, roll


def get_mjd_ranges(config):
    mjdmin_list = []
    mjdmax_list = []
    for mjd_range in config['MJD_RANGES']:
        mjd_list = mjd_range.split()
        mjdmin_list.append( float(mjd_list[0]) )
        mjdmax_list.append( float(mjd_list[1]) )

    return mjdmin_list, mjdmax_list


def get_galaxy_cat_HOSTLIB(config, hostlib_file, iser=0):
    # input iser = 0 or 1 = Sersic profile (e.g., disk & bulge)
    print_banner(f"Read galaxy catalog with Sersic component {iser}")

    hostlib_col  = HOSTLIB_COLUMNS
    ra, dec, radius = get_sky_region(config)

    # replace a_Sersic and b_Sersic ..
    sersic_replace_list = [ 'n', 'a', 'b' ]
    for s in sersic_replace_list:
        #logging.info(f"\t replace {s} with {s}{iser} ")
        s_old = f"{s}_Sersic"
        s_new = f"{s}{iser}_Sersic"
        hostlib_col  = [ s_new if item == s_old else item for item in hostlib_col]

    logging.info(f"HOSTLIB file to read:    {hostlib_file}")
    logging.info(f"HOSTLIB columns to read: {hostlib_col} ")

    t_all = Table.read(hostlib_file, format='ascii', include_names = hostlib_col )
    ntot_gal = len(t_all)

    t_select = apply_radius_cut(config, t_all, 'RA_GAL', 'DEC_GAL')

    # - - - - - -
    ncut_gal = len(t_select)
    logging.info(f"Number of galaxies (all -> radiusCut): {ntot_gal} -> {ncut_gal}")
    logging.info(f"Rename columns for query:")

    for colname_h, colname_soc in zip(hostlib_col, SOC_COLUMNS):
        if colname_h in t_select.colnames:  # note that w0_Sersic or w1_sersic could be missing
            t_select.rename_column(colname_h, colname_soc)

    # for iser = 0, create w0_Sersic column = 1 - w1
    if iser == 0:
        t_select['w0'] = 1.0 - t_select['w1']
        t_select['w0'] = [float(f"{x:.4f}") for x in t_select['w0']] # avoid spurious extra digits

    var_w = f"w{iser}"  # w0 or w1, used to select fraction of flux for this Sersic component
    logging.info(f"Convert ABmag columns to maggie flux = 10**(-0.4*ABmag) * SersicFraction")
    for mag in BAND_LIST_SOC:  # here is a mag
        if not reject_band(args,mag):
            logging.info(f"\t convert {mag} ABmag to maggie flux")
            t_select[mag] = t_select[var_w] * 10.0**(-0.4*t_select[mag])  # convert to maggie


    logging.info(f"Add column for half_light_radius = a and  ba = b/a:")
    t_select['half_light_radius'] = t_select['a']
    t_select['ba'] = t_select['b'] / t_select['a']
    t_select['ba'] = [float(f"{x:.4f}") for x in t_select['ba']] # avoid spurious extra digits

    logging.info(f"Add column for type = {TYPE_SERSIC} ")
    t_select['type'] = TYPE_SERSIC

    logging.info(f"Table columns for query: {t_select.colnames}")

    logging.info(f"{t_select[0:5]}")
    return t_select  # end get_galaxy_cat_HOSTLIB



def apply_mjd_cut(config, t, COLNAME_MJD ):
    mjdmin_list, mjdmax_list = get_mjd_ranges(config)
    mask_sum = None
    for mjdmin, mjdmax in zip(mjdmin_list,mjdmax_list):
        mask0 = t[COLNAME_MJD] >= (mjdmin-0.0001)
        mask1 = t[COLNAME_MJD] <= (mjdmax+0.0001)
        mask  = mask0 & mask1
        #sys.exit(f"\n xxx mask = {mask[0:20]}")
        if mask_sum is None:
            mask_sum = mask
        else:
            mask_sum = mask_sum | mask


    t_cut = t[mask_sum]
    return t_cut



def apply_radius_cut(config, t, COLNAME_RA, COLNAME_DEC):
    # return table with radius cut around central RA,DEC

    # apply radius cut 
    ra, dec, radius, roll = get_sky_region(config)

    t_coords = t[ COLNAME_RA, COLNAME_DEC ]
    center = SkyCoord(ra= ra*u.deg, dec = dec*u.deg, frame='icrs')

    #  Create a SkyCoord object for the events
    event_coords = SkyCoord(ra=t_coords[COLNAME_RA]*u.deg,
                            dec=t_coords[COLNAME_DEC]*u.deg, frame='icrs')

    #  Calculate the separation (on-sky angular distance)
    sep = center.separation(event_coords)

    # 5. Apply the cut
    mask = sep < radius * u.deg
    t_select = t[mask]

    return t_select


def run_sim(catalog, config, obs_time='2026-01-31T00:00:00', mjd_shift=None,
            mjd_transient=None, sca=1, band="F087",psftype="stpsf",
            ma_table_number=1001, level=2, usecrds=True,
            drop_extra_dq=True,pretend_spectral=None,
            seed=7, make_plot=False,rootname='r0003201001001001004'):
    """
    catalog is an astropy table
    config is from this scripts config
    """
    if mjd_shift:
        mjd_image = mjd_transient + mjd_shift
        t_mjd     = Time(mjd_image, format='mjd')
        obs_date  = t_mjd.isot
        # compute visit id = int(mjd * 1E4)
        visit_id = int(mjd_image * 1.0E4 + 0.5)
    else: 
        obs_date = obs_time
        visit_id = 50000


    # Output file name on disk. Only change first part up to _WFI to change the rootname of the file.
    cal_level = 'cal' if level == 2 else 'uncal' 
    jobname  = f"VISIT{visit_id}_WFI{sca:02d}_{band}_{cal_level}"
    filename = f'{rootname}_0003_wfi{sca:02d}_{band.lower()}_{cal_level}.asdf'

    print_banner(f"Run the sim for {filename}: Level {level} in {jobname}")
    if mjd_shift:
        logging.info(f"\t mjd={mjd_image} -> {obs_date}")


    ra, dec, radius, roll = get_sky_region(config)

    logging.info('\t Set Galsim RNG object')
    rng = galsim.UniformDeviate(seed)

    logging.info(f'\t Set default persistance information')
    persist = persistence.Persistence()

    if usecrds:
        logging.info('\t Set reference files in romanisim to None for CRDS')
        for k in parameters.reference_data:
            parameters.reference_data[k] = None

    logging.info(f'\t Set metadata')
    metadata = ris.set_metadata(date=obs_date, bandpass=band, sca=sca,
                                ma_table_number=ma_table_number, usecrds=usecrds)


    logging.info(f'\t Update the WCS info ')
    rwcs.fill_in_parameters(metadata, SkyCoord(ra, dec, unit='deg', frame='icrs'),
                           boresight=False, pa_aper=roll)

    sim_parser = argparse.ArgumentParser()
    sim_parser.set_defaults(usecrds=usecrds, psftype=psftype, level=level,
                      filename=filename, drop_extra_dq=drop_extra_dq, sca=sca,
                      bandpass=band, pretend_spectral=None)    
    sim_args = sim_parser.parse_args([])


    logging.info(f'\t Running the sim')
    t_start = time.time()
    ris.simulate_image_file(sim_args, metadata, catalog, rng, persist)
    t_end = time.time()
    t_sim = t_end - t_start # seconds
 
    logging.info(f"CPU({jobname}):  {t_sim:.0f} seconds ")
    logging.info(f"File created: {filename}")
    if make_plot:
        imagelib.mkfigure(filename, plotname=filename.split(".")[0]+".png")

    return


def create_galaxy_catalog(ra, dec, radius, n_gal=10_000,index=0.4,
                          faint_mag=22, hlight_radius=0.3,
                          band=['F087'], rng=None, seed=5346):


    galaxy_cat = riscatalog.make_galaxies(SkyCoord(ra, dec, unit='deg'),
                                          n_gal, radius=radius, index=index,
                                          faintmag=faint_mag, 
                                          hlr_at_faintmag=hlight_radius,
                                          bandpasses=band,
                                          rng=rng, seed=seed)
    
    # create a list of catalog objects
    #catalog = riscatalog.table_to_catalog(galaxy_cat, bandpasses=band)

    return galaxy_cat 



def create_star_catalog(ra, dec, radius, n_star=30_000,
                        index=5./3., faint_mag=22,rng=None,
                        band=['F087'], seed=7):
    
   
    star_cat = riscatalog.make_stars(SkyCoord(ra, dec, unit='deg'), n_star,
                                  radius=radius, index=3/5., faintmag=faint_mag, 
                                  truncation_radius=None, bandpasses=band,
                                  rng=rng,seed=seed)
    #catalog = riscatalog.table_to_catalog(star_cat, bandpasses=band)

    return star_cat  



def create_gaia_catalog(ra=80., dec=30., radius=1, obs_time='2026-01-31T00:00:00',
                        filename='gaia_catalog.ecsv', overwrite=True):
    """Create a catalog from gaia sources

    Inputs
    ------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    radius : float
        Search radius in degrees
    obs_time : str
        observation time
    filename : str
        Filename to save output catalog
    overwrite : bool
        Whether to overwrite catalog file on disk


    Returns
    -------
    filename: str
        The name of the gaia catalog saved to disk


    """
    logging.info("Creating gaia catalog....")

    query = f'SELECT * FROM gaiadr3.gaia_source WHERE distance({ra}, {dec}, ra, dec) < {radius}'
    job = Gaia.launch_job_async(query)
    result = job.get_results()

    logging.info('Filtering the Gaia results for stars and exclude bright stars')
    result = result[result['classprob_dsc_combmod_star'] >= 0.7]
    result = result[result['phot_g_mean_mag'] > 16.5]

    logging.info('Make the Roman I-Sim formatted catalog')
    gaia_catalog = gaia.gaia2romanisimcat(result,   
                                      Time(obs_time),
                                      fluxfields=set(romanisim.bandpass.galsim2roman_bandpass.values()))

    # Clean out bad sources from Gaia catalog:
    filters = [f for f in gaia_catalog.dtype.names if f[0] == 'F']
    bad = np.zeros(len(gaia_catalog), dtype='bool')
    for b in filters:
         bad = ~np.isfinite(gaia_catalog[b])
         if hasattr(gaia_catalog[b], 'mask'):
              bad |= gaia_catalog[b].mask
         gaia_catalog = gaia_catalog[~bad]

    gaia_catalog = gaia_catalog[np.isfinite(gaia_catalog['ra'])]
    gaia_catalog.write(filename, overwrite=overwrite)
    logging.info("GAIA catalog saved to {filename}")
    n_star = len(gaia_catalog)
    logging.info(f"Number of Gaia stars in catalog: {n_star} ")
    logging.info(f"Gaia catalog columns: {gaia_catalog.colnames} ")    
    

def print_matable(crds_server_url, start_time="2026-01-01T00:00:00.000",
                crds_context="", crds_path="./", resultants=9,
                crds_url="https://roman-crds.stsci.edu/"):
    """print the current MA Table reference file from CRDS

    """
    logging.info("Getting crds server and checking MA Table File")
    os.environ["CRDS_SERVER_URL"]=crds_server_url
    os.environ["CRDS_PATH"] = crds_path
    if crds_context=="":
        crds_context = crdsapi.get_default_context()
        logging.info(f"Using latest CRDS_CONTEXT = {crds_context}")
        os.environ["CRDS_CONTEXT"] = crds_context

    # get the matable reference file, it will only retrieve if missing from local
    try:
        reference_file=crds.getreferences({"ROMAN.META.INSTRUMENT.NAME":"WFI",
                            "ROMAN.META.EXPOSURE.START_TIME":start_time},
                            reftypes=['matable'], observatory='roman')
    except Exception as e:
        raise IOError(f"Problem getting MA Table frm CRDS: {e}") from e
    
    #print the tables with less that x resultants
    print(f"\n{'#'*100}\nName, ID, exposure times, resultants\n\n")
    matable = rdm.open(reference_file["matable"])
    for key,val in matable.science_tables.items():
        if matable.science_tables[key]['num_science_resultants'] < resultants:
            print(key,matable.science_tables[key]['ma_table_number'],
                  matable.science_tables[key]['accumulated_exposure_time'],
                  matable.science_tables[key]['num_science_resultants'])



def check_for_pypi_compat(cal_name="romancal", cal_version="latest"):
    """Check for the compatibility of the specified CAL version and PYTHON version
    and return the string that should be used for pip install. 

    Inputs
    ------
    cal_name : str
        The name of the CAL software package

    cal_version : str
        The version of the cal software package to install

    Returns
    -------
    desciption : str
        Formatted string describing the CAL release

    cal_string : str
        The fully qualified package==version string for installation by pip
    """
    python_version = str(sys.version_info[0] + sys.version_info[1]/100)

    # cal_name is the CAL package name
    if cal_version == "latest":
        # assume cal_name is package name and pypi name
        # use latest on pypi
        cal_string = cal_name
    else:
        cal_string = f"{cal_name}=={cal_version}"   #for the pip call


    # check for version string in cal and compatibility with pypi and python for the session
    try:
        check_version = subprocess.run(["pip","install",f"{cal_string}", "--dry-run", 
                            "--python-version", python_version,"--no-deps","--target","foo",
                            "--quiet","--report","-"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise ConnectionError from e(f"Error checking pypi for {cal_name}")

    if (check_version.stderr):
        raise ValueError(f"No matching versions for python={python_version} and {cal_string} found in pypi")
    else:
        # get the version string from pypi
        json_version = json.loads(check_version.stdout)
        pypi_matched = json_version['install'][0]['metadata']['version']
        description = json_version['install'][0]['metadata']['description']
    return (description, f"{cal_name}=={pypi_matched}")


def test_image_input(config):
    # make some simple example images
    imsz = 99
    cenpix = imsz // 2
    yy, xx = np.meshgrid(np.arange(imsz) - cenpix, np.arange(imsz) - cenpix)
    im1 = ((xx ** 2 + yy ** 2) < 30 ** 2) * 1.0  # circle
    im2 = im1 * 0
    im2[35:65, 10:-10] = 1  # rectangle
    rng = galsim.UniformDeviate(config["SEED"])

    # make a PSF for these
    psf = im1 * 0
    psf[cenpix, cenpix] = 1
    from scipy.ndimage import gaussian_filter
    sigma = 1
    psf = gaussian_filter(psf, sigma)

    # set up the image catalog
    from astropy.io import fits
    filenames = ['im1.fits','im2.fits']
    fits.writeto(filenames[0], gaussian_filter(im1, sigma))
    fits.writeto(filenames[1], gaussian_filter(im2, sigma))
    base_rgc_filename = 'test_image_catalog'
    riscatalog.make_image_catalog(filenames, psf, base_rgc_filename)

    # make some metadata to describe an image for us to render
    galsim.roman.n_pix = 2000
    ra,dec,rad, roll = get_sky_region(config)
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    meta = util.default_image_meta(coord=coord, filter_name='F087')
    wcs.fill_in_parameters(meta, coord)
    imwcs = wcs.get_wcs(meta, usecrds=True)

    # make a table of sources for us to render
    tab = Table()
    cen = imwcs.toWorld(galsim.PositionD(galsim.roman.n_pix / 2, galsim.roman.n_pix / 2))
    offsets = np.array([[-300, 300, -100, -200, 0, 0],
                        [0, 100, -200, -100, 0, -300]])
    offsets = offsets * 0.1 / 60 / 60
    tab['ra'] = util.skycoord(cen).ra.to(u.degree).value + offsets[0, :]
    tab['dec'] = util.skycoord(cen).dec.to(u.degree).value + offsets[1, :]
    tab['ident'] = np.arange(6) % 2  # alternate circles and rectangles
    tab['rotate'] = np.arange(6) * 60
    tab['shear_pa'] = (5 - np.arange(6)) * 60
    tab['shear_ba'] = [0.5, 0.3, 0.9, 0.8, 1, 1]
    tab['dilate'] = [0.5, 0.1, 0.9, 1.1, 2, 0.8]
    tab['F087'] = [1e-7, 2e-7, 3e-7, 3e-7, 2e-7, 1e-7]
    tab.meta['real_galaxy_catalog_filename'] = str(base_rgc_filename) + '.fits'

    # render the image
    res = romanisim.image.simulate(meta, tab, usecrds=False, psftype='galsim',rng=rng)

    artifactdir = '.'
    if artifactdir is not None:
        af = asdf.AsdfFile()
        af.tree = {'im1': im1,
                   'im2': im2,
                   'psf': psf,
                   'catalog': tab,
                   'output': res[0].data,
                   }
        af.write_to(os.path.join(artifactdir, 'test_image.asdf'))


@dataclass(init=True, repr=True)
class PointWFI:
    """
    Inputs
    ------
    ra (float): Right ascension of the target placed at the geometric 
                center of the Wide Field Instrument (WFI) focal plane
                array. This has units of degrees.
    dec (float): Declination of the target placed at the geometric
                 center of the WFI focal plane array. This has units
                 of degrees.
    position_angle (float): Position angle of the WFI relative to the V3 axis
                            measured from North to East. A value of 0.0 degrees
                            would place the WFI in the "smiley face" orientation
                            (U-shaped) on the celestial sphere. To place WFI
                            such that the position angle of the V3 axis is 
                            zero degrees, use a WFI position angle of -60 degrees.

    Description
    -----------
    This was copied from the roman_notebook example
    To use this class, insantiate it with your initial pointing like so:

        >>> point = PointWFI(ra=30, dec=-45, position_angle=10)
    
    and then dither using the dither method:

        >>> point.dither(x_offset=10, y_offset=140)

    This would shift the WFI 10 arcseconds along the X-axis of the WFI
    and 140 arcseconds along the Y-axis of the WFI. These axes are in the ideal
    coordinate system of the WFI, i.e, with the WFI oriented in a U-shape with 
    +x to the right and +y up. You can pull the new pointing info out of the object 
    either as attributes or by just printing the object:

        >>> print(point.ra, point.dec)
        >>> 29.95536280064078 -44.977122003232786

    or

        >>> point
        >>> PointWFI(ra=29.95536280064078, dec=-44.977122003232786, position_angle=10)
    """

    # Set default pointing parameters
    ra: float = 0.0
    dec: float = 0.0
    position_angle: float = -0.0

    # Post init method sets some other defaults and initializes
    # the attitude matrix using PySIAF.
    def __post_init__(self) -> None:
        self.siaf_aperture = pysiaf.Siaf('Roman')['WFI_CEN']
        self.v2_ref = self.siaf_aperture.V2Ref
        self.v3_ref = self.siaf_aperture.V3Ref
        self.attitude_matrix = pysiaf.utils.rotations.attitude(self.v2_ref, self.v3_ref, self.ra,
                                        self.dec, self.position_angle)
        self.siaf_aperture.set_attitude_matrix(self.attitude_matrix)

        # Compute the V3 position angle
        self.tel_roll = pysiaf.utils.rotations.posangle(self.attitude_matrix, 0, 0)

        # Save initial pointing
        self.att0 = self.attitude_matrix.copy()

        # Save a copy of the input RA and Dec in case someone needs it
        self.ra0 = copy.copy(self.ra)
        self.dec0 = copy.copy(self.dec)

    def dither(self, x_offset: Union[int, float],
               y_offset: Union[int, float]) -> None:
        """
        Purpose
        -------
        Take in an ideal X and Y offset in arcseconds and shift the telescope
        pointing to that position.

        Inputs
        ------
        x_offset (float): The offset in arcseconds in the ideal X direction.

        y_offset (float): The offset in arcseconds in the ideal Y direction.
        """

        self.ra, self.dec = self.siaf_aperture.idl_to_sky(x_offset, y_offset)


def set_crds(config):
    """Set CRDS context information

    if calling from the cmdline you'll have to set this yourself
    """
    logging.info(config['CRDS_SERVER_URL'])
    logging.info(config['CRDS_PATH'])
    os.environ["CRDS_SERVER_URL"]=config["CRDS_SERVER_URL"]
    os.environ["CRDS_PATH"] = config["CRDS_PATH"]
    if config["CRDS_CONTEXT"] == "latest":
        crds_context = crdsapi.get_default_context()
        
    else:
        crds_context = config["CRDS_CONTEXT"]

    os.environ["CRDS_CONTEXT"] = crds_context
    logging.info(f"Using CRDS_CONTEXT = {crds_context}")


def make_sky_image(config):
    """ Make a sky image and save a fits

    Make a sky image with sources in counts
    that you can provide to romanisim as the source fluxes you want to add
    using the extra_counts parameter


    Notes
    -----
    extra_counts needs to be a fits file
    if we want to add transients into the simulation, we should
    create a starting fits file with the transient sources in it with their full counts
    romanisim will take this file and disperse the countrs along the ramp
    for a level 1 image

    The --extra-counts argument(s) allows the user to pass in a FITS file with
    an array of counts (not counts/time). If a second argument is passed, it 
    is assumed to be the HDU number to read. This argument is useful to wrap 
    idealized images into the Roman L1/L2 datamodel, including detector effects.
    You will probably want to set --nobj 0 here to avoid simulating additional
    sources. We do not get any additional information/metadata from this file,
    so you will need to set other parameters (e.g. --bandpass, --date, --radec, etc)
    appropriately to match the image you are wrapping.
    
    make_l1(counts, read_pattern[, read_noise, ...]) Make an L1 image from a total electrons image.
    
    """
    import galsim
    import numpy as np

    # 1. Image setup (pixel scale)
    pixel_scale = 0.11  # arcsec/pixel
    image = galsim.Image(4088, 4088, scale=pixel_scale)

    # 2. Define object (flux in counts)
    # A galaxy with 50,000 counts, half-light radius 1.5"
    obj = galsim.Sersic(n=1, half_light_radius=1.5, flux=50000)

    # 3. Define PSF
    psf = galsim.Gaussian(fwhm=0.7)

    # 4. Convolve and Draw
    final_obj = galsim.Convolve([obj, psf])
    final_obj.drawImage(image, method='phot')

    # 5. Add Poisson Noise (sky + source)
    sky_level = 100 # counts/pixel
    image += np.random.poisson(sky_level, image.array.shape)

    # Save the image
    image.write('sky_simulation.fits')


def get_filter_params(config):
    """Read in the filter parameters from the github file

    https://github.com/RomanSpaceTelescope/roman-technical-information/tree/main
    """
    params = Table.read(config['WFI_FILTER_PARAMS']['NAME'])
    return params


if __name__ == "__main__":

    # the recommended calls to romanisim
    # https://romanisim.readthedocs.io/en/latest/romanisim/running.html#running-the-simulation
    # https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-multiaccum-ma-tables/imaging-multiaccum-tables
    #
    # The example notebooks in the nexus are also here, clone it locally
    # git clone git@github.com:spacetelescope/roman_notebooks.git
    #
    # You can provide the following flag inputs to run with other than the defaults
    parser_inputs = argparse.ArgumentParser(description="%(prog)s simulation inputs")

    msg = "MultiAccum table number (1000<)"
    parser_inputs.add_argument("--ma_table_number",
                               choices=[1001, 1002, 1003, 1004, 1005,1006,1007,1008,1009,1010,1011,1034,1035,9001,9002],
                               required=False, type=int,
                               help=msg, default=1018)

    msg = "Roman Bandpass"
    parser_inputs.add_argument("--bandpass",
                               choices=['F062','F087','F106','F129',
                                        'F146','F158','F184','F213'],
                               required=False, type=str,
                               help=msg, default='F129')

    msg = "ra  (deg)"
    parser_inputs.add_argument("--ra",
                               required=False, type=float,
                               help=msg, default=9.671532)
 
    msg = "dec  (deg)"
    parser_inputs.add_argument("--dec",
                               required=False, type=float,
                               help=msg, default=-45.202805)

    msg = "radius (deg)"
    parser_inputs.add_argument("--radius",
                               required=False, type=float,
                               help=msg, default=1)

    
    msg = "roll  (deg)"
    parser_inputs.add_argument("--roll",
                               required=False, type=float,
                               help=msg, default=-60)
 
    
    msg = "PSF Type (stpsf)"
    parser_inputs.add_argument("--psftype",
                               required=False, type=str,
                               choices=["galsim","stpsf"],
                               help=msg, default='stpsf')



    msg = "Data level to simulate for image sim [1,2]"
    parser_inputs.add_argument("--level", "-l", choices=[1,2],
                               help=msg, default=1, required=False,
                               type=int )

    msg = "Required: name of sim config input file"
    parser_inputs.add_argument("--date",
                               required=False, type=str,
                               help=msg, default='2026-01-01T00:00:00.000')

    parser_inputs.add_argument('--usecrds',
                        action="store_true",
                        default=True, required=False,
                        help='Use crds reference files')    

    msg = "Required: name of sim config input file"
    parser_inputs.add_argument("--input_config_file",
                               required=True, type=str,
                               help=msg, default=None)

    msg = "Required: SCA number (1-18)"
    parser_inputs.add_argument("--scanum",
                               help=msg, default=1, 
                               type=int, required=False )

    msg = f"Required: BAND {' '.join(BAND_LIST_INPUT)}"
    parser_inputs.add_argument("--band_snana",
                               help=msg, default='R062-R', 
                               type=str, required=False)

    msg = "optional: MJD shift (w.r.t transient sim) for image sim "
    parser_inputs.add_argument("--mjd_shift",
                               help=msg, default=0, required=False,
                               type=float )

    msg = "Run quick_sim_test with prescaled sources and no transients"
    parser_inputs.add_argument("--quick_sim_test",required=False,
                               default=False,
                               help=msg, action="store_true")

    msg = "Create a spectral image"
    parser_inputs.add_argument("--pretend_spectral", default=None,
                               help=msg, required=False)

    msg = "Create png plot for each image"
    parser_inputs.add_argument("--plot", default=False, help=msg, 
                              required=False, action="store_true")

    msg = "keep extra dq from sim"
    parser_inputs.add_argument("--dropdq", default=True, help=msg, 
                              required=False, action="store_false")


    args = parser_inputs.parse_args()

    # read config file from the command line
    config = read_config(args.input_config_file)
    logging.info("STARTING SIMULATON RUN WITH\n")
    print_banner(config)

    set_crds(config)

    # create catalogs
    ra, dec, radius, roll = get_sky_region(config)
    gaia_name = f'gaia_catalog_{ra:.2f}_{dec:.2f}_{roll:.2f}.ecsv'  #just easier for now
    if not os.access(gaia_name, os.F_OK):
        create_gaia_catalog(ra=ra, dec=dec, radius=radius, obs_time=config['OBS_TIME'],
                            filename=gaia_name, overwrite=True)
    gaia_catalog = Table.read(gaia_name)

    # make a catalog with all the filters
    star_catalog = create_star_catalog(ra, dec, radius,seed=config['SEED'],rng=None,
                                       band=list(config['BAND_LIST_SOC'].values()))

    # make a catalog with all the filters
    gal_catalog = create_galaxy_catalog(ra, dec, radius,
                                        hlight_radius=0.3,seed=config['SEED'],rng=None,
                                        band=list(config['BAND_LIST_SOC'].values()))

    # save a combined catalog
    full_name = f'parametric_catalog_{ra:.2f}_{dec:.2f}_{roll:.2f}.ecsv'
    if not os.access(full_name, os.F_OK):
        full_catalog = vstack([gal_catalog, star_catalog])
        full_catalog.write(full_name, format='ascii.ecsv', overwrite=True)
    else:
        full_catalog = Table.read(full_name)


    # extra_counts needs to be a fits file
    # if we want to add transients into the simulation, we should
    # create a starting fits file with the transient sources in it with their full counts
    # romanisim will take this file and disperse the countrs along the ramp
    # for a level 1 image
    # The --extra-counts argument(s) allows the user to pass in a FITS file with
    # an array of counts (not counts/time). If a second argument is passed, it 
    # is assumed to be the HDU number to read. This argument is useful to wrap 
    # idealized images into the Roman L1/L2 datamodel, including detector effects.
    # You will probably want to set --nobj 0 here to avoid simulating additional
    # sources. We do not get any additional information/metadata from this file,
    # so you will need to set other parameters (e.g. --bandpass, --date, --radec, etc)
    # appropriately to match the image you are wrapping.
    
    #
    # make_l1(counts, read_pattern[, read_noise, ...]) Make an L1 image from a total electrons image.
    #

    run_sim(full_catalog, config, obs_time=config["OBS_TIME"], mjd_shift=args.mjd_shift,
            sca=args.scanum, band=args.bandpass,drop_extra_dq=args.dropdq,
            psftype=args.psftype, ma_table_number=args.ma_table_number, 
            level=args.level, usecrds=args.usecrds, seed=config["SEED"],
            rootname='r0003201001001001007',make_plot=args.plot)

    # somethings off with the stars here, flux maybe
    run_sim(gal_catalog, config, obs_time=config["OBS_TIME"], mjd_shift=args.mjd_shift,
            sca=args.scanum, band=args.bandpass,drop_extra_dq=args.dropdq,
            psftype=args.psftype, ma_table_number=args.ma_table_number, 
            level=args.level, usecrds=args.usecrds, seed=config["SEED"],
            rootname='r0003201001001001006',make_plot=args.plot)

    # somthings off with the stars here, flux maybe
    run_sim(star_catalog, config, obs_time=config["OBS_TIME"], mjd_shift=args.mjd_shift,
            sca=args.scanum, band=args.bandpass,drop_extra_dq=args.dropdq,
            psftype=args.psftype, ma_table_number=args.ma_table_number, 
            level=args.level, usecrds=args.usecrds, seed=config["SEED"],
            rootname='r0003201001001001005',make_plot=args.plot)

    # read in the gaia catalog
    logging.info(f"Using catalog: {gaia_name}")
    run_sim(gaia_catalog, config, obs_time=config["OBS_TIME"], mjd_shift=args.mjd_shift,
            sca=args.scanum, band=args.bandpass,drop_extra_dq=args.dropdq,
            psftype=args.psftype, ma_table_number=args.ma_table_number, 
            level=args.level, usecrds=args.usecrds, seed=config["SEED"],
            rootname='r0003201001001001004', make_plot=args.plot)

    #
    # dither the last image in X and Y and make a new image
    #
    pointing = PointWFI(ra=ra, dec=dec, position_angle=roll)
    pointing.dither(x_offset=-25.5, y_offset=10.5)
    logging.info(pointing)

    config["SKY_REGION"]["RA_CEN"] = pointing.ra 
    config["SKY_REGION"]["DEC_CEN"] = pointing.dec 
    config["SKY_REGION"]["ROLL"] = pointing.position_angle
    config["OBS_TIME"] = "2026-01-31T00:01:00" # add a different start time

    logging.info(f"Moving telescope to: {config['SKY_REGION']}")
    ra, dec, radius, roll = get_sky_region(config)
    gaia_name = f'gaia_catalog_{ra:.2f}_{dec:.2f}_{roll:.2f}.ecsv'  #just easier for now
    if not os.access(gaia_name, os.F_OK):
        create_gaia_catalog(ra=ra, dec=dec, radius=radius, obs_time=config['OBS_TIME'],
                            filename=gaia_name, overwrite=True)
    gaia_catalog = Table.read(gaia_name)
    run_sim(gaia_catalog, config, obs_time=config["OBS_TIME"], mjd_shift=args.mjd_shift,
            sca=args.scanum, band=args.bandpass,drop_extra_dq=args.dropdq,
            psftype=args.psftype, ma_table_number=args.ma_table_number, 
            level=args.level, usecrds=args.usecrds, seed=config["SEED"],
            rootname='r0003201001001001008', make_plot=args.plot)


