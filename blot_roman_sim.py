import os
import glob
import numpy as np
from astropy.io import fits
from astropy import wcs
from numpy.fft import fft2, ifft2, fftshift, ifftshift
# we keep the import, but we'll skip calling AstroDrizzle for sims
from drizzlepac import astrodrizzle
from blot_test import new_blot, new_drz

########################################
# 1) CONFIG / INPUTS
########################################

# Folder that contains your *smaller* simulated images
archive_dir = "/sim_wcs_dithers"

# These should now be your small simulated frames
input_list = sorted(glob.glob(os.path.join(archive_dir, "*.fits")))
if not input_list:
    raise FileNotFoundError(f"No FITS files found under {archive_dir}")

print(f"Found {len(input_list)} simulated inputs (showing up to 5):")
for p in input_list[:5]:
    print("  ", p)

# Make sure every input has EXPTIME so new_blot doesn't crash
def ensure_exptime_keyword(fpath, value=1.0):
    with fits.open(fpath, mode="update") as hdul:
        phdr = hdul[0].header
        if "EXPTIME" not in phdr:
            phdr["EXPTIME"] = (value, "added for iDrizzle simulated data")

for f in input_list:
    ensure_exptime_keyword(f)

# We want to test multiple band-limits
# band limit for 0.05"/pix and 1024x1024 J129 ~ 250
radii_to_try = [
    (200, 260),
    # add more here
    # (200, 240),
    # (230, 250),
]

# number of iDrizzle iterations you wanted to push toward 10–20
n_iterations = 12

# Drizzle target: 1024 × 1024 at 0.05"/pix
target_scale = 0.05      # arcsec/pixel
final_outnx = 1024
final_outny = 1024
pixfrac = 0.6

# Base output dir
base_output_dir = os.path.join(archive_dir, "idrizzle_runs")
os.makedirs(base_output_dir, exist_ok=True)


########################################
# 2) UTILITIES
########################################

def first_image_hdu(hdul):
    """Return index of first 2D image HDU."""
    for idx, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 2:
            return idx
    raise ValueError("No 2D image HDU found in FITS file.")

def make_blank_reference_fits_from_example(path, nx, ny, scale_arcsec, example_fits):
    """
    Create a blank 2D image (SCI ext=1) with TAN WCS centered on the image.
    Try to match CRVAL of the example FITS, so blotting uses similar sky coords.
    """
    # defaults
    ra0, dec0 = 0.0, 0.0

    # try to read CRVAL1/CRVAL2 from example
    with fits.open(example_fits) as ex:
        for hdu in ex:
            hdr = hdu.header
            if "CRVAL1" in hdr and "CRVAL2" in hdr:
                ra0 = float(hdr["CRVAL1"])
                dec0 = float(hdr["CRVAL2"])
                break

    data = np.zeros((ny, nx), dtype=np.float32)

    # build WCS: e.g. 0.05"/pix -> deg/pix
    w = wcs.WCS(naxis=2)
    cdelt = scale_arcsec / 3600.0
    w.wcs.cdelt = np.array([-cdelt, cdelt])  # RA decreases with +x
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    w.wcs.crval = [ra0, dec0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()

    # PRIMARY + SCI (ext=1) because new_drz reads ext=1
    phdu = fits.PrimaryHDU()
    sci = fits.ImageHDU(data=data, header=hdr, name="SCI")
    fits.HDUList([phdu, sci]).writeto(path, overwrite=True)
    return path

def circular_taper_mask(ny, nx, r_in, r_out):
    y, x = np.indices((ny, nx), dtype=float)
    cy, cx = ny // 2, nx // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    mask = np.zeros_like(r)
    mask[r <= r_in] = 1.0
    mask[r >= r_out] = 0.0
    between = (r > r_in) & (r < r_out)
    mask[between] = 0.5 * (1.0 + np.cos(np.pi * (r[between] - r_in) / (r_out - r_in)))
    return mask

def fourier_filter(input_fits_path, output_fits_path, r_in, r_out):
    with fits.open(input_fits_path) as hdul:
        im_idx = first_image_hdu(hdul)
        im = np.nan_to_num(hdul[im_idx].data)
        ny, nx = im.shape

        fft_data = fftshift(fft2(im))
        mask = circular_taper_mask(ny, nx, r_in, r_out)
        filtered_fft_data = fft_data * mask
        filtered_data = np.real(ifft2(ifftshift(filtered_fft_data)))

        hdul[im_idx].data = filtered_data.astype(np.float32, copy=False)
        hdul.writeto(output_fits_path, overwrite=True)


########################################
# 3) MAIN LOOP OVER (r_in, r_out)
########################################

for (r_in, r_out) in radii_to_try:
    print("\n========================================")
    print(f"Running iDrizzle for r_in={r_in}, r_out={r_out}")
    print("========================================")

    # Each radius pair gets its own output dir
    output_dir = os.path.join(base_output_dir, f"run_r{r_in}_{r_out}")
    os.makedirs(output_dir, exist_ok=True)

    # Make a blank template 1024x1024 @ 0.05"/pix, matched to first sim file
    sum0_path = os.path.join(output_dir, "sum0.fits")
    make_blank_reference_fits_from_example(
        sum0_path,
        final_outnx,
        final_outny,
        target_scale,
        input_list[0],
    )

    ##############################################
    # 3a) Initial drizzle onto this blank frame
    ##############################################
    # We know AstroDrizzle is going to complain about INSTRUME, so skip for sims.
    # If you want to try again later, uncomment this.
    #
    # try:
    #     astrodrizzle.AstroDrizzle(
    #         ",".join(input_list),
    #         output=sum0_path,
    #         final_scale=target_scale,
    #         final_pixfrac=pixfrac,
    #         final_outnx=final_outnx,
    #         final_outny=final_outny,
    #         final_wcs=True,
    #         in_memory=True,
    #         build=True
    #     )
    #     print("AstroDrizzle completed.")
    # except Exception as e:
    #     print("AstroDrizzle failed or is not applicable. Falling back to new_drz. Error:", e)

    print("Running new_drz to produce sum0...")
    new_drizzle_data = new_drz(input_list, sum0_path, samples_per_drc_pixel=1)

    # overwrite the SCI/data in sum0.fits with the drizzle output
    with fits.open(sum0_path, mode="update") as f:
        im_idx = first_image_hdu(f)
        f[im_idx].data = new_drizzle_data.astype(np.float32, copy=False)

    # Save a copy as sum0_new
    sum0_new_path = os.path.join(output_dir, "sum0_new.fits")
    fits.open(sum0_path).writeto(sum0_new_path, overwrite=True)

    # Apply Fourier filter to get the first "global" model
    sum0_filt_path = os.path.join(output_dir, "sum0_filtered.fits")
    fourier_filter(sum0_new_path, sum0_filt_path, r_in, r_out)
    print("Initial filtered image saved as sum0_filtered.fits")

    # Initialize global accumulator image
    with fits.open(sum0_filt_path) as hdul:
        im_idx = first_image_hdu(hdul)
        global_image = hdul[im_idx].data.copy()
        global_output = os.path.join(output_dir, "sum0_global.fits")
        hdul.writeto(global_output, overwrite=True)

    ##############################################
    # 4) iDrizzle Iteration Loop
    ##############################################
    for i in range(n_iterations):
        print(f"\n>>> Starting iDrizzle iteration {i} for r_in={r_in}, r_out={r_out} <<<")

        # model to blot from
        model_path = os.path.join(output_dir, f"sum{i}_global.fits")

        # dir to hold residual-subtracted frames
        subtracted_dir = os.path.join(output_dir, f"iter{i}_subtracted")
        os.makedirs(subtracted_dir, exist_ok=True)

        # blot & subtract for each input
        for file_path in input_list:
            file_name = os.path.basename(file_path)
            out_name = file_name.replace(".fits.gz", "_sub.fits").replace(".fits", "_sub.fits")
            sub_file = os.path.join(subtracted_dir, out_name)

            # new_blot will now find EXPTIME in the input header
            blotted_data = new_blot(
                model_path,
                file_path,
                samples_per_flc_pixel=1,
                scale_by_exptime=False
            )

            with fits.open(file_path) as hdul:
                im_idx = first_image_hdu(hdul)
                original_data = hdul[im_idx].data
                residual = original_data - blotted_data
                hdul[im_idx].data = residual.astype(np.float32, copy=False)
                hdul.writeto(sub_file, overwrite=True)

        print(f"Subtraction complete for iteration {i}")

        # gather residuals
        residual_list = sorted(
            os.path.join(subtracted_dir, f)
            for f in os.listdir(subtracted_dir)
            if f.endswith("_sub.fits")
        )
        print("Residuals to drizzle:", len(residual_list))

        # drizzle residuals back onto the SAME 1024x1024 template
        sum_output = os.path.join(output_dir, f"sum{i+1}.fits")
        new_drizzle_data = new_drz(residual_list, sum0_path, samples_per_drc_pixel=1)

        # write to new sumX
        with fits.open(sum0_path) as f:
            im_idx = first_image_hdu(f)
            f[im_idx].data = new_drizzle_data.astype(np.float32, copy=False)
            f.writeto(sum_output, overwrite=True)

        # filter
        filtered_output = os.path.join(output_dir, f"sum{i+1}_filtered.fits")
        fourier_filter(sum_output, filtered_output, r_in, r_out)

        # accumulate to global
        with fits.open(filtered_output) as f:
            im_idx = first_image_hdu(f)
            global_image += f[im_idx].data

        # save global
        global_output = os.path.join(output_dir, f"sum{i+1}_global.fits")
        with fits.open(sum_output) as f:
            im_idx = first_image_hdu(f)
            f[im_idx].data = global_image.astype(np.float32, copy=False)
            f.writeto(global_output, overwrite=True)

        print(f"Iteration {i} complete -> {filtered_output}")

    print(f"\nAll iterations done for r_in={r_in}, r_out={r_out}. Outputs in {output_dir}")
