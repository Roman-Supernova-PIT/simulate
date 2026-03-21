# Convert FITS to asdf
# this is not an exaustive or complete example
# it will convert with SIP if it find them, but not the inverse

import numpy as np

from astropy.modeling import models
from astropy.modeling.mappings import UnitsMapping
from astropy.io import fits
from astropy import coordinates as coords 
from astropy import units as u 
from astropy.modeling.models import SIP, Polynomial2D

from gwcs import wcs 
from gwcs import coordinate_frames as cf 

import roman_datamodels as rdm 
from roman_datamodels import datamodels as dm


def convert(fits_name="data.fits", outfile="data.asdf",ignore_sip=False):
    """ Convert the fits image to an asdf file that conforms to roman_datamodels
    
        Inputs
        ------
        fits_name : str
            The name of the fits file

        ignore_sip : bool
            Ignore any SIP coefficients in the header

        outfile : str
            Name of the output asdf file

        Notes
        -----
        IF your image is MEF it should have EXTNAMES for idefication of data

        Creates an ASDF file that's the same as a Roman WFI Level 2 image

        Will convert float64 to float32 or 16 for the roman datamodel

    """
    simple=False # not a simple fits image with data in primary
    try:
        image=fits.open(fits_name)
    except:
        print(f"Unexpected non-fits data in {fits_name}\n")
    extens = len(image) # how many extensions

    if not outfile:
            outfile=fits_name.split(".")[0]+".asdf"

    # get all the extension names
    # see if the primary extension has data
    extlist=[]
    if extens == 1:
        try:
            res = len(image[0].data)
        except TypeError:
            simple=True
            make_simple(image,fits_name, outfile=outfile)

    if not simple:
        for i in range(1,extens):
            extlist.append(image[i].header["EXTNAME"])
        if "SCI" in extlist:
            xsize, ysize = (image["SCI"].header["NAXIS1"], image["SCI"].header["NAXIS2"])
        adm=dm.ImageModel.create_fake_data(shape=(xsize, ysize))
        
        #convert simple wcs information
        wcsobj = make_gwcs(image["SCI"].header, ignore_sip=ignore_sip)
        adm.meta.wcs=wcsobj

        #add the data to the model
        #convert datatypes if necessary
        if "SCI" in extlist:
            adm.data = image["SCI"].data.astype(np.float32, copy=False)
        if "ERR" in extlist:
            adm.err = image["ERR"].data.astype(np.float16, copy=False)
        if "DQ" in extlist:
            adm.dq = image["DQ"].data.astype(np.uint32, copy=False)

        #save the asdf file
        adm.save(outfile)
    image.close()
    

def make_simple(image, fits_name, outfile="", ignore_sip=False):
    """write a simple fits to rdm file

    Inputs
    ------
    image: astropy.io.fits object
        fits image object

    fits_name: str
        name of the fits file

    Notes
    -----
    Assumes that the data is attached to the primary extension

    """
    # empty imagemodel
    xsize,ysize=(image[0].header['NAXIS1'], image[0].header['NAXIS2'])
    adm=dm.ImageModel.create_fake_data(shape=(xsize,ysize))
    wcsobj = make_gwcs(image[0].header, ignore_sip=ignore_sip)
    adm.meta.wcs=wcsobj
    adm.data=image[0].data.astype(np.float32, copy=False)
    adm.save(fits_name.split(".")[0]+".asdf")

    

def sip_helper(header):
    """ Helper include the SIP coefficients from the header

    Notes
    -----
    The SIP coefficients describe the distortion in X and Y
    only A_ORDER and B_ORDER coeffs are used, not any AP,AB inverse coeefs if provided
    (that can be added later though)

    """
    # sip in header
    # polynomial2d doesn't accept units
    crpix1,crpix2=(header['CRPIX1'], header['CRPIX2'])
    shift_by_crpix = models.Shift(-(crpix1 - 1)*u.pix) & models.Shift(-(crpix2 - 1)*u.pix)
    if header["CTYPE1"] == 'RA---TAN-SIP':                                                     
        if header["CTYPE2"] == 'DEC--TAN-SIP':
            A_ORDER = int(header["A_ORDER"])
            B_ORDER = int(header["B_ORDER"])
            a_coeffs = header["A_?_?"].cards
            b_coeffs = header["B_?_?"].cards
            a_coeff={}
            for a in a_coeffs:
                a_coeff[a[0]] = a[1]
            b_coeff={}
            for b in b_coeffs:
                b_coeff[b[0]] = b[1]
            sip_x = Polynomial2D(A_ORDER)
            parameters=[0,1] + [v for k,v in a_coeff.items()]
            parameters.insert(5, 0)
            sip_x.parameters = parameters
            sip_y = Polynomial2D(B_ORDER) 
            parameters=[0,0] + [v for k,v in b_coeff.items()]
            parameters.insert(5,1)
            sip_y.parameters=parameters
            # this changes x,y inputs to none for the poly2d eval
            unit_map_in = UnitsMapping(((u.pix,None),(u.pix,None)))
            # this changes the output of poly2d to pixels
            unit_map_out = UnitsMapping(((None,u.pix),(None,u.pix)))

            distortion = (shift_by_crpix | unit_map_in | models.Mapping((0,1,0,1)) | 
                           (sip_x & sip_y) | unit_map_out |
                           shift_by_crpix)
            
            distortion.name = "distortion"

    else:
        raise ValueError("Expected to find SIP coefficients in header")
    
    return distortion


def make_gwcs(header, ignore_sip=False):
    """Make a GWCS object from FITS header information

    Inputs
    ------
    header : astropy.io.fits header object
        header object to use

    ignore_sip : bool
        ignore any SIP information in the header even if it's there

    Returns
    -------
    wcsojb : gwcs.wcs object

    Notes
    -----
    This will use SIP coefficients found in header unless told otherwise
    """

    crpix1 = header['CRPIX1'] 
    crpix2 = header['CRPIX2']
    # offset by crpix in header
    shift_by_crpix = models.Shift(-(crpix1 - 1)*u.pix) & models.Shift(-(crpix2 - 1)*u.pix)

    # rotate through the CD matrix
    try:
        CD1_1=header['CD1_1']
        CD1_2=header['CD1_2']
        CD2_1=header['CD2_1']
        CD2_2=header['CD2_2']
    except:
        print(f"Need CD matrix in header to continue\n{header['CD1_1']}")

    matrix = np.array([[CD1_1, CD1_2],
                      [CD2_1 , CD2_2]])
    
    rotation = models.AffineTransformation2D(matrix * u.deg,
                                             translation=[0, 0] * u.deg)
    rotation.input_units_equivalencies = {"x": u.pixel_scale(1*u.deg/u.pix),
                                          "y": u.pixel_scale(1*u.deg/u.pix)}
    rotation.inverse = models.AffineTransformation2D(np.linalg.inv(matrix) * u.pix,
                                                     translation=[0, 0] * u.pix)
    rotation.inverse.input_units_equivalencies = {"x": u.pixel_scale(1*u.pix/u.deg),
                                                  "y": u.pixel_scale(1*u.pix/u.deg)}

    # create the tangent projection
    tan = models.Pix2Sky_TAN()
    crval1,crval2=(header['CRVAL1']*u.deg, header['CRVAL2']*u.deg)
    celestial_rotation =  models.RotateNative2Celestial(crval1, crval2, 180*u.deg)
    
    dosip=False
    if not ignore_sip:
        if (header["CTYPE1"] == 'RA---TAN-SIP'):
            dosip=True
            #get sip part of the transform, and add it to the pipeline
            # detector pixels -> distorted pixels -> sky coordinates
            distortion=sip_helper(header)
            
            # remove units from this frame and then add them back
            undistorted_frame = cf.Frame2D(name="undistorted_frame",
                                           axes_names=(("undist_x", "undist_y")),
                                           unit=(u.pix, u.pix))

            det2sky = rotation | tan | celestial_rotation
            det2sky.name = "linear_transform"

    else: # detector pixesl -> sky coordinates
        det2sky = shift_by_crpix | rotation | tan | celestial_rotation
        det2sky.name = "linear_transform"


    # create detector and sky coordinate frame, and if needed SIP
    detector_frame = cf.Frame2D(name="detector",
                                axes_names=(("x", "y")),
                                unit=(u.pix, u.pix))

    sky_frame = cf.CelestialFrame(reference_frame=coords.ICRS(),
                                  name='icrs',
                                  unit=(u.deg, u.deg))

    # make the pipeline wihtout distortion
    if dosip:
        pipeline = [(detector_frame, distortion),
                    (undistorted_frame, det2sky),
                    (sky_frame, None)]
    else:
        pipeline = [(detector_frame, det2sky), (sky_frame, None)]
    
    wcsobj = wcs.WCS(pipeline)
    return wcsobj


# Example call for the given data.fits

# import fits2asdf
# import imagelib
# fits2asdf.convert("data.fits", ignore_sip=False, outfile="data_with_sip.asdf")
# fits2asdf.convert("data.fits", ignore_sip=True, outfile="data_no_sip.asdf")
#
# make a matplotlib figure
# imagelib.mkfigure('data_with_sip.asdf',cmap='cividis', plotname='asdf_sci_sip_data.png')
# imagelib.mkfigure('data_no_sip.asdf',cmap='cividis', plotname='asdf_sci_nosip_data.png')

# imagelib.mkfigure('data.fits',cmap='cividis', plotname='fits_sci_data.png')



