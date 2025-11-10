import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage import map_coordinates
import time
from drizzlepac import pixtopix
import tqdm

def new_blot(drc_input, flc_output, samples_per_flc_pixel = 4, scale_by_exptime = False):
    print("Running new_blot", drc_input, flc_output, samples_per_flc_pixel)
    
    f_flc = fits.open(flc_output)
    flc_wcs = wcs.WCS(fits.getheader(flc_output, 0), f_flc)  # CHANGED: use ext=0
    print("flc_wcs", flc_wcs)

    EXPTIME = f_flc[0].header["EXPTIME"]
    flc_shape = f_flc[0].data.shape  # CHANGED: use ext=0
    f_flc.close()

    f_drc = fits.open(drc_input)
    print("f_drc", f_drc.info())
    drc_wcs = wcs.WCS(fits.getheader(drc_input, 1), f_drc)
    print("drc_wcs", drc_wcs)
    drc_data = f_drc[1].data*1.
    f_drc.close()

    flc_x = np.arange(flc_shape[1]*samples_per_flc_pixel, dtype=np.float64)/float(samples_per_flc_pixel)
    flc_y = np.arange(flc_shape[0]*samples_per_flc_pixel, dtype=np.float64)/float(samples_per_flc_pixel)

    flc_x -= np.mean(flc_x[:samples_per_flc_pixel])
    flc_y -= np.mean(flc_y[:samples_per_flc_pixel])

    flc_x += 1
    flc_y += 1

    assert np.isclose(np.mean(flc_x[:samples_per_flc_pixel]), 1)
    assert np.isclose(np.mean(flc_y[:samples_per_flc_pixel]), 1)

    print("flc_x,y", flc_x, flc_y)

    flc_X, flc_Y = np.meshgrid(flc_x, flc_y)
    flc_XY = np.stack((flc_X.flatten(), flc_Y.flatten()), axis=1)
    print("flc_XY", flc_XY)

    print("Transforming", time.asctime())

    x,y = pixtopix.tran(flc_output + "[0]", drc_input + "[sci,1]", "forward",  # CHANGED: ext [0]
                        flc_XY[:,0], flc_XY[:,1], verbose = False)

    inds_flc_in_drc = np.array([y, x]) - 1
    
    mapped_image = map_coordinates(input = drc_data, coordinates = inds_flc_in_drc, prefilter = False, order = 3)
    mapped_image = np.reshape(mapped_image, (flc_shape[0]*samples_per_flc_pixel, flc_shape[1]*samples_per_flc_pixel))
    if scale_by_exptime:
        mapped_image *= EXPTIME
    mapped_image /= samples_per_flc_pixel**2.

    mapped_image_orig_number_pixels = 0.
    for i in range(samples_per_flc_pixel):
        for j in range(samples_per_flc_pixel):
            mapped_image_orig_number_pixels += mapped_image[i::samples_per_flc_pixel, j::samples_per_flc_pixel]
    return mapped_image_orig_number_pixels


def new_drz(flc_inputs, drc_output, samples_per_drc_pixel = 4):
    print("Running new_drz", flc_inputs, drc_output, samples_per_drc_pixel)

    f_drc = fits.open(drc_output)
    drc_wcs = wcs.WCS(fits.getheader(drc_output, 1), f_drc)
    print("drc_wcs", drc_wcs)

    drc_shape = f_drc[1].data.shape
    f_drc.close()

    drc_x = np.arange(drc_shape[1]*samples_per_drc_pixel, dtype=np.float64)/float(samples_per_drc_pixel)
    drc_y = np.arange(drc_shape[0]*samples_per_drc_pixel, dtype=np.float64)/float(samples_per_drc_pixel)

    drc_x -= np.mean(drc_x[:samples_per_drc_pixel])
    drc_y -= np.mean(drc_y[:samples_per_drc_pixel])

    drc_x += 1
    drc_y += 1

    assert np.isclose(np.mean(drc_x[:samples_per_drc_pixel]), 1)
    assert np.isclose(np.mean(drc_y[:samples_per_drc_pixel]), 1)

    print("drc_x,y", drc_x, drc_y)

    drc_X, drc_Y = np.meshgrid(drc_x, drc_y)
    drc_XY = np.stack((drc_X.flatten(), drc_Y.flatten()), axis=1)
    print("drc_XY", drc_XY)

    # Prepare an output array of zeros
    mapped_image = np.zeros(drc_shape[0]*drc_shape[1], dtype=np.float64)
    mapped_dq = np.zeros(drc_shape[0]*drc_shape[1], dtype=np.float32)
    
    for flc_input in tqdm.tqdm(flc_inputs):
        f_flc = fits.open(flc_input)
        print("f_flc", f_flc.info())
        flc_wcs = wcs.WCS(fits.getheader(flc_input, 0), f_flc)  # CHANGED: use ext=0
        print("flc_wcs", flc_wcs)
        flc_data = f_flc[0].data*1.  # CHANGED: use ext=0

        # CHANGED: DQ mask optional (Roman files may not have ext=3)
        mask = ~(512 | 2048 | 8192)
        if len(f_flc) > 3 and getattr(f_flc[3], "data", None) is not None and f_flc[3].data.shape == flc_data.shape:
            dq = f_flc[3].data
        else:
            dq = np.zeros_like(flc_data, dtype=np.int16)
        flc_good_mask = np.array((dq & mask) == 0, dtype=np.int16)

        f_flc.close()

        print("Transforming", time.asctime())

        x,y = pixtopix.tran(flc_input + "[0]", drc_output + "[sci,1]", "reverse",  # CHANGED: ext [0]
                            drc_XY[:,0], drc_XY[:,1], verbose = False)
        print("Done Transforming", time.asctime())

        inds_drc_in_flc = np.array([y, x]) - 1  # These are indices in the input flc image
        inds_drc_in_flc = np.array(np.around(inds_drc_in_flc), dtype=np.int32)

        # Start Indexing
        rows = inds_drc_in_flc[0]
        cols = inds_drc_in_flc[1]

        # Find where rows, cols are within flc_data shape
        valid_mask = ((rows >= 0) & (rows < flc_data.shape[0]) &
                      (cols >= 0) & (cols < flc_data.shape[1]))

        # Only assign to valid locations
        mapped_image[valid_mask] += (flc_data*flc_good_mask)[rows[valid_mask], cols[valid_mask]]
        mapped_dq[valid_mask] += flc_good_mask[rows[valid_mask], cols[valid_mask]]
        # End Indexing

    # CHANGED: avoid divide-by-zero (pixels never hit by any input)
    mapped_dq[mapped_dq == 0] = 1.0

    mapped_image /= mapped_dq

    print("mapped_image", mapped_image.shape)    
    
    mapped_image = np.reshape(mapped_image, (drc_shape[0]*samples_per_drc_pixel, drc_shape[1]*samples_per_drc_pixel))

    #if scale_by_exptime:
    #    mapped_image *= EXPTIME
    mapped_image /= samples_per_drc_pixel**2.

    mapped_image_orig_number_pixels = 0.
    for i in range(samples_per_drc_pixel):
        for j in range(samples_per_drc_pixel):
            mapped_image_orig_number_pixels += mapped_image[i::samples_per_drc_pixel, j::samples_per_drc_pixel]
    return mapped_image_orig_number_pixels



if __name__ == "__main__":
    from astropy.io import fits 

    drc_input = "/sum0.fits"
    flc_output = "/Archive/icqt94a9q_flt.fits"

    samples_per_flc_pixel = 4
    mapped_image = new_blot(drc_input, flc_output, samples_per_flc_pixel)
    fits.writeto("blotted_image.fits", mapped_image, overwrite=True)

  
