import json
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u

# ----- USER INPUT -----
fits_file = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Andromeda/h_e_20170608_stacked.fits"
# pixel coordinates of 5 chosen reference stars
ref_stars = {
    "ref1": {"x-coord": "1263.3168", "y-coord": "706.224"}, 
    "ref2": {"x-coord": "421.70867", "y-coord": "1355.8273"},
    "ref3": {"x-coord": "1593.3986", "y-coord": "1492.3688"},
    "ref4": {"x-coord": "192.33158", "y-coord": "236.39108"},
    "ref5": {"x-coord": "1027.8659", "y-coord": "1575.0207"}
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