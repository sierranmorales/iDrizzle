import os
import glob
import numpy as np
from astropy.io import fits
from astropy import wcs
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from drizzlepac import astrodrizzle
from blot_test import new_blot, new_drz
from scipy.ndimage import map_coordinates

############################
# 1) Inputs and Setup
############################

# Point to the folder where you downloaded the Roman FITS images
archive_dir = "/Users/sierra/Documents/GitHub/iDrizzle/downloads"

# Settings
r_in = 1200
r_out = 2000
n_iterations = 6  # Number of iterations to perform (adjustable)

# Define output directory for saving results
output_dir = os.path.join(archive_dir, f"output_{r_in}_{r_out}")
os.makedirs(output_dir, exist_ok=True)

# Pick one: "truth" or "simple_model"
dataset_kind = "truth"

# Gather all Roman FITS (handles .fits or .fits.gz), recursively
input_list = sorted(
    glob.glob(os.path.join(archive_dir, "**", f"Roman_TDS_{dataset_kind}_J129_*_*.fits*"), recursive=True)
)

if not input_list:
    raise FileNotFoundError(
        "No Roman FITS images found. "
        f"Make sure files like J129/<id>/Roman_TDS_{dataset_kind}_J129_<id>_<sca>.fits.gz exist under {archive_dir}"
    )

# Limit to the first N inputs
max_inputs = 15
selected_list = input_list[:max_inputs]

print(f"Found {len(input_list)} total inputs; using {len(selected_list)} (showing up to 5):")
for p in selected_list[:5]:
    print("  ", p)

# Set drizzle parameters: output scale, pixfrac, and image dimensions
target_scale = 0.032    # arcsec/pixel
pixfrac = 0.6
final_outnx = 4096
final_outny = 4096

#############################################
# 2) Taper Mask and Fourier Filter Functions
#############################################

def first_image_hdu(hdul):
    """Return index of the first 2D image HDU (handles SCI vs PRIMARY)."""
    for idx, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and getattr(hdu.data, "ndim", 0) == 2:
            return idx
    raise ValueError("No 2D image HDU found in FITS file.")

def circular_taper_mask(ny, nx, r_in, r_out):
    """
    Generate a 2D circular taper mask for filtering in Fourier space.
    Inside r_in: pass (value=1.0)
    Outside r_out: block (value=0.0)
    Between: smooth cosine taper.
    """
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
    """
    Apply Fourier-space filtering with a circular taper mask.
    - Load input FITS.
    - Perform FFT.
    - Apply taper mask.
    - Inverse FFT to real space.
    - Save filtered output.
    """
    with fits.open(input_fits_path) as hdul:
        im_idx = first_image_hdu(hdul)
        im = np.nan_to_num(hdul[im_idx].data)
        ny, nx = im.shape
        fft_data = fftshift(fft2(im))
        mask = circular_taper_mask(ny, nx, r_in, r_out)
        filtered_fft_data = fft_data * mask
        filtered_data = np.real(ifft2(ifftshift(filtered_fft_data)))
        hdul[im_idx].data = filtered_data
        hdul.writeto(output_fits_path, overwrite=True)

##############################################
# 3) Initial Drizzle and Filter (Iteration 0)
##############################################

sum0_path = os.path.join(output_dir, "sum0.fits")

# Try AstroDrizzle (likely to fail for non-HST). Fall back to a stub with SCI ext=1.
try:
    astrodrizzle.AstroDrizzle(
        ",".join(selected_list),
        output=sum0_path,
        final_scale=target_scale,
        final_pixfrac=pixfrac,
        final_outnx=final_outnx,
        final_outny=final_outny,
        final_wcs=True,
        in_memory=True,
        build=True
    )
    print("AstroDrizzle completed.")
except Exception as e:
    print("AstroDrizzle failed or is not applicable. Falling back to new_drz. Error:", e)
    # Create a stub sum0.fits with a SCI extension (ext=1) so new_drz can read it
    with fits.open(selected_list[0]) as ref:
        im_idx = first_image_hdu(ref)  # for Roman, this will be 0 (PRIMARY)
        ref_hdr = ref[im_idx].header.copy()
        shape = ref[im_idx].data.shape
        phdu = fits.PrimaryHDU()  # keep primary minimal
        sci  = fits.ImageHDU(data=np.zeros(shape, dtype=np.float32), header=ref_hdr, name='SCI')
        fits.HDUList([phdu, sci]).writeto(sum0_path, overwrite=True)

# Apply custom drizzle to generate the data for sum0.fits
print("Running new_drz to produce sum0...")
new_drizzle_data = new_drz(selected_list, sum0_path, samples_per_drc_pixel=1)

with fits.open(sum0_path) as f:
    im_idx = first_image_hdu(f)   # should be the SCI ext we created / AstroDrizzle wrote
    f[im_idx].data = new_drizzle_data.astype(np.float32, copy=False)
    f.writeto(os.path.join(output_dir, "sum0_new.fits"), overwrite=True)

# Apply Fourier filter to the initial corrected image
fourier_filter(os.path.join(output_dir, "sum0_new.fits"),
               os.path.join(output_dir, "sum0_filtered.fits"),
               r_in, r_out)
print("Initial filtered image saved as sum0_filtered.fits")

#############################
# 4) iDrizzle Iteration Loop
#############################

with fits.open(os.path.join(output_dir, "sum0_filtered.fits")) as hdul:
    im_idx = first_image_hdu(hdul)
    global_image = hdul[im_idx].data.copy()
    global_output = os.path.join(output_dir, "sum0_global.fits")
    hdul.writeto(global_output, overwrite=True)

for i in range(n_iterations):
    print(f"\n>>> Starting iDrizzle iteration {i} <<<")

    # Path to the filtered model image for this iteration
    model_path = os.path.join(output_dir, f"sum{i}_global.fits")

    # Create a directory to store the subtracted residuals
    subtracted_dir = os.path.join(output_dir, f"iter{i}_subtracted")
    os.makedirs(subtracted_dir, exist_ok=True)

    # For each input file, blot the model back and subtract
    for file_path in selected_list:
        file_name = os.path.basename(file_path)
        out_name = file_name.replace(".fits.gz", "_sub.fits").replace(".fits", "_sub.fits")
        sub_file = os.path.join(subtracted_dir, out_name)

        # Blot model onto the input frame (Roman image pixels live at PRIMARY/0; handled inside new_blot)
        blotted_data = new_blot(
            model_path,
            file_path,
            samples_per_flc_pixel=1,
            scale_by_exptime=False
        )

        # Subtract blotted model from original image (we detect the 2D HDU index)
        with fits.open(file_path) as hdul:
            im_idx = first_image_hdu(hdul)  # Roman -> 0
            original_data = hdul[im_idx].data
            residual = original_data - blotted_data
            hdul[im_idx].data = residual.astype(np.float32, copy=False)
            hdul.writeto(sub_file, overwrite=True)

    print(f"Subtraction complete for iteration {i}")

    # Create a list of residual files for re-drizzling
    residual_list = sorted(
        os.path.join(subtracted_dir, f)
        for f in os.listdir(subtracted_dir)
        if f.endswith("_sub.fits")
    )
    print("Residuals:", len(residual_list))

    # Drizzle residuals to produce a new combined image
    sum_output = os.path.join(output_dir, f"sum{i+1}.fits")
    print("Running new_drz for iteration", i+1)
    new_drizzle_data = new_drz(residual_list, os.path.join(output_dir, "sum0.fits"), samples_per_drc_pixel=1)

    # Overwrite the SCI extension with the corrected drizzle output
    with fits.open(os.path.join(output_dir, "sum0.fits")) as f:
        im_idx = first_image_hdu(f)    # SCI ext
        f[im_idx].data = new_drizzle_data.astype(np.float32, copy=False)
        f.writeto(sum_output, overwrite=True)

    # Apply Fourier filter to the new drizzle output
    filtered_output = os.path.join(output_dir, f"sum{i+1}_filtered.fits")
    fourier_filter(sum_output, filtered_output, r_in, r_out)

    # Accumulate the global image
    with fits.open(filtered_output) as f:
        im_idx = first_image_hdu(f)
        global_image += f[im_idx].data

    # Save the global image at the end of each iteration
    global_output = os.path.join(output_dir, f"sum{i+1}_global.fits")
    with fits.open(sum_output) as f:
        im_idx = first_image_hdu(f)
        f[im_idx].data = global_image.astype(np.float32, copy=False)
        f.writeto(global_output, overwrite=True)

    print(f"Iteration {i} re-drizzle + filter complete -> {filtered_output}")

# End of iterations
print("\nAll iDrizzle iterations finished. Check your outputs in:")
print(output_dir)
