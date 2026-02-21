import json
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u

# ----- USER INPUT -----
fits_file = "your_image.fits"
# pixel coordinates of 5 chosen reference stars
ref_stars = {
    "ref1": {"x": 1822.2847, "y": 1935.9641},
    "ref2": {"x": 1595.6322, "y": 818.53086},
    "ref3": {"x": 2641.3142, "y": 3047.2638},
    "ref4": {"x": 3015.019, "y": 1277.7626},
    "ref5": {"x": 849.90332, "y": 2542.1704}
}

# ----------------------

# Open FITS and WCS
hdu = fits.open(fits_file)[0]
wcs = WCS(hdu.header)

# Initialize Vizier for Pan-STARRS DR2
Vizier.ROW_LIMIT = 1  # get closest star only
catalog_name = "II/349/ps1"  # Pan-STARRS DR2

output_dict = {}

for key, star in ref_stars.items():
    # Pixel to sky
    ra, dec = wcs.all_pix2world(star["x"], star["y"], 0)
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    
    # Query Pan-STARRS for g-band
    try:
        result = Vizier.query_region(coord, radius=2*u.arcsec, catalog=catalog_name)
        if len(result) == 0 or len(result[0]) == 0:
            g_mag = "NaN"
        else:
            g_mag = float(result[0]['gmag'][0])
    except Exception as e:
        print(f"Error querying {key}: {e}")
        g_mag = "NaN"
    
    output_dict[key] = {
        "x-coord": star["x"],
        "y-coord": star["y"],
        "G_true": g_mag
    }

# Print JSON-style output
print(json.dumps(output_dict, indent=4))