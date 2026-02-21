from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import numpy as np

data = fits.getdata("/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda/h_e_20170608_stacked.fits") / fits.getheader("/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda/h_e_20170608_stacked.fits")["EXPTIME"]

# Empty sky near CV1
x, y = 900, 950  # adjust

ap = CircularAperture((x, y), r=10)
ann = CircularAnnulus((x, y), r_in=15, r_out=20)

ap_phot = aperture_photometry(data, ap)
ann_phot = aperture_photometry(data, ann)

sky_per_pix = ann_phot['aperture_sum'][0] / ann.area
flux = ap_phot['aperture_sum'][0] - sky_per_pix * ap.area

print(f"Sky flux in aperture: {flux:.3f}")
print(f"Sky instrumental mag: {-2.5 * np.log10(abs(flux)):.3f}")