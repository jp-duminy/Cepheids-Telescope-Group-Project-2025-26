import os
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import astropy.units as u
import csv


#set input and output folder path to any desired folder containing FITS files 
folder = "path/to/fits/files"
csv_path = "path/to/output/airmass.csv" #airmass.csv name of file, but can be changed

def safe_float(v):
    '''
    Safely convert to float, return None on failure.

    Inputs:
        v - value to convert
    Returns:
        float value or None
    '''
    try:
        return float(v)
    except Exception:
        return None

def parse_ra_dec(header):
    """Parse RA and DEC from FITS header.
    
    Inputs:
        header - FITS header object
    Returns:
        (ra_deg, dec_deg) in decimal degrees or (None, None) if not found
    """
    if "RA-TEL" in header and "DEC-TEL" in header:
        return float(header["RA-TEL"]), float(header["DEC-TEL"])

    if "RA" in header and "DEC" in header:
        ra, dec = header["RA"], header["DEC"]
        try:
            ra, dec = float(ra), float(dec)
            if ra < 24:  # RA in hours
                ra *= 15
            return ra, dec
        except Exception:
            ra_angle = Angle(str(ra), unit=u.hourangle)
            dec_angle = Angle(str(dec), unit=u.deg)
            return ra_angle.degree, dec_angle.degree
    return None, None

def deg_to_hms_dms(ra_deg, dec_deg):
    """
    Convert decimal degrees to HMS/DMS strings.
    
    Inputs:
        ra_deg - Right Ascension in decimal degrees
        dec_deg - Declination in decimal degrees
    Returns:
        (ra_str, dec_str) formatted strings
    """
    ra_angle = Angle(ra_deg * u.deg)
    dec_angle = Angle(dec_deg * u.deg)
    ra_str = ra_angle.to_string(unit=u.hour, sep=":", precision=2, pad=True)
    dec_str = dec_angle.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True, pad=True)
    return ra_str, dec_str

def process_fits(path):
    """
    Process a single FITS file to compute airmass and return formatted string.
    
    Inputs:
        path - path to FITS file
    Returns:
        multi-line string with results or None on failure
    """
    with fits.open(path) as hdul:
        h = hdul[0].header
        obj = h.get("OBJECT", "Unknown")
        date_obs = h.get("DATE-OBS") or h.get("GPSSTART")
        if not date_obs:
            return None

        ra_deg, dec_deg = parse_ra_dec(h)
        if ra_deg is None or dec_deg is None:
            return None

        lat = safe_float(h.get("OBSLAT"))
        lon = safe_float(h.get("OBSLON"))
        elev = safe_float(h.get("OBSALT"))
        if None in (lat, lon, elev):
            return None

        # Compute Alt/Az + Airmass
        sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
        t = Time(date_obs)
        altaz = sky.transform_to(AltAz(obstime=t, location=loc))
        alt = altaz.alt.degree
        airmass = altaz.secz

        ra_str, dec_str = deg_to_hms_dms(ra_deg, dec_deg)

        # Create multi-line string for CSV
        block = (
            f"Object;        {obj}\n"
            f"RA (h:m:s);    {ra_str}\n"
            f"DEC (d:m:s);   {dec_str}\n"
            f"Date Obs;      {t.iso.split('.')[0]}\n"
            f"Altitude (°);  {alt:.2f}\n"
            f"Airmass;       {airmass:.2f}\n"
        )

        return block

if __name__ == "__main__":
    fits_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".fits"):
                fits_files.append(os.path.join(root, f))
    if not fits_files:
        print("No FITS files found.")
        exit()

    results = [r for f in fits_files if (r := process_fits(f))]
    if not results:
        print("No valid FITS files processed.")
        exit()

    # Write CSV in a readable format
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for block in results:
            for line in block.strip().split("\n"):
                writer.writerow([line])
            writer.writerow([])  # Blank line between entries

    # Print in terminal
    print("\n========== AIRMASS RESULTS ==========\n")
    for block in results:
        print(block + "------------------------------------\n")

    print(f"Results saved to: {csv_path}")
