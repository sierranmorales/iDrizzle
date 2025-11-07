import os
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

# Define archive directory containing input FLT images
archive_dir = "/Users/sierra/Documents/GitHub/iDrizzle/Archive"

# Settings
# Set Fourier filtering mask radii
r_in = 500
r_out = 900

n_iterations = 3  # Number of iterations to perform (adjustable)

# Define output directory for saving results
output_dir = os.path.join(archive_dir, f"output_{r_in}_{r_out}")
os.makedirs(output_dir, exist_ok=True)

# Gather all _flt.fits files in the archive directory
input_list = [
    os.path.join(archive_dir, f)
    for f in os.listdir(archive_dir)
    if f.endswith("_flt.fits")
]

# Set drizzle parameters: output scale, pixfrac, and image dimensions
target_scale = 0.032    # arcsec/pixel
pixfrac = 0.6
final_outnx = 4096
final_outny = 4096

#############################################
# 2) Taper Mask and Fourier Filter Functions
#############################################

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
        drizzled_data = np.nan_to_num(hdul["SCI"].data)
        ny, nx = drizzled_data.shape
        fft_data = fftshift(fft2(drizzled_data))
        mask = circular_taper_mask(ny, nx, r_in, r_out)
        filtered_fft_data = fft_data * mask
        filtered_data = np.real(ifft2(ifftshift(filtered_fft_data)))
        hdul["SCI"].data = filtered_data
        hdul.writeto(output_fits_path, overwrite=True)

##############################################
# 3) Initial Drizzle and Filter (Iteration 0)
##############################################

# Run AstroDrizzle once to combine input FLT images into 'sum0.fits'
astrodrizzle.AstroDrizzle(
    ",".join(input_list),
    output=os.path.join(output_dir, "sum0.fits"),
    final_scale=target_scale,
    final_pixfrac=pixfrac,
    final_outnx=final_outnx,
    final_outny=final_outny,
    final_wcs=True,
    in_memory=True,
    build=True
)

# Apply custom drizzle correction using new_drz function
new_drizzle = new_drz(input_list, output_dir + "/sum0.fits", samples_per_drc_pixel=1)

# Overwrite the SCI extension with the corrected drizzle output
with fits.open(output_dir + "/sum0.fits") as f:
    f["SCI"].data = new_drizzle
    f.writeto(output_dir + "/sum0_new.fits", overwrite=True)

# Apply Fourier filter to the initial corrected image
fourier_filter(output_dir + "/sum0_new.fits", output_dir + "/sum0_filtered.fits", r_in, r_out)
print("Initial filtered image saved as sum0_filtered.fits")


#############################
# 4) iDrizzle Iteration Loop
#############################

with fits.open(output_dir + "/sum0_filtered.fits") as hdul:
    global_image = hdul[1].data
    global_output = os.path.join(output_dir, "sum0_global.fits")
    hdul.writeto(global_output, overwrite=True)

# Save the global image at the end of each iteration
for i in range(n_iterations):
    print(f"\n>>> Starting iDrizzle iteration {i} <<<")

    # Path to the filtered model image for this iteration
    model_path = os.path.join(output_dir, f"sum{i}_global.fits")

    # Create a directory to store the subtracted FLT residuals
    subtracted_dir = os.path.join(output_dir, f"iter{i}_subtracted")
    os.makedirs(subtracted_dir, exist_ok=True)

    # For each input FLT file, blot the model back and subtract
    for file_path in input_list:
        file_name = os.path.basename(file_path)
        sub_file = os.path.join(subtracted_dir, file_name.replace(".fits", "_sub.fits"))

        # Blot model onto the FLT frame
        blotted_data = new_blot(
            model_path,
            file_path,
            samples_per_flc_pixel=1,
            scale_by_exptime=False
        )

        # Subtract blotted model from original FLT data
        with fits.open(file_path) as hdul:
            original_data = hdul[1].data
            residual = original_data - blotted_data
            hdul[1].data = residual
            hdul.writeto(sub_file, overwrite=True)

    print(f"Subtraction complete for iteration {i}")

    # Create a list of residual files for re-drizzling
    residual_list = [
        os.path.join(subtracted_dir, f)
        for f in os.listdir(subtracted_dir)
        if f.endswith("_sub.fits")
    ]

    print("residual_list:", residual_list)

    # Drizzle residuals to produce a new combined image
    sum_output = os.path.join(output_dir, f"sum{i+1}.fits")

    # new drz
    print("Running new_drz, sum_output=", sum_output)
    new_drizzle = new_drz(residual_list, output_dir + "/sum0.fits", samples_per_drc_pixel=1)
    
    # Overwrite the SCI extension with the corrected drizzle output
    with fits.open(output_dir + "/sum0.fits") as f:
        f["SCI"].data = new_drizzle
        f.writeto(sum_output, overwrite=True)

    """
    astrodrizzle.AstroDrizzle(
        ",".join(residual_list),
        output=sum_output,
        final_scale=target_scale,
        final_pixfrac=pixfrac,
        final_outnx=final_outnx,
        final_outny=final_outny,
        final_wcs=True,
        in_memory=True,
        build=True
    )
    """

    # Apply Fourier filter to the new drizzle output
    filtered_output = os.path.join(output_dir, f"sum{i+1}_filtered.fits")
    fourier_filter(sum_output, filtered_output, r_in, r_out)

    # global image that accumulated iterations
    with fits.open(filtered_output) as f:
        global_image += f["SCI"].data

    # Save the global image at the end of each iteration
    global_output = os.path.join(output_dir, f"sum{i+1}_global.fits")
    with fits.open(sum_output) as f:
        f["SCI"].data = global_image
        f.writeto(global_output, overwrite=True)


    print(f"Iteration {i} re-drizzle + filter complete -> {filtered_output}")

# End of iterations
print("\nAll iDrizzle iterations finished. Check your outputs in:")
print(output_dir)
