import os
from pyclbr import Class
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
import astropy.units as u
import csv

class AirmassInfo:
    '''Class to extract airmass information'''
    def __init__(self, filename):
        '''Initialise filename and airmass'''
        self.filename = filename
        self.airmass = None
        self.RA = None
        self.Dec = None

    def safe_float(self, v):
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

    def parse_ra_dec(self, header):
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

    def deg_to_hms_dms(self, ra_deg, dec_deg):
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

    def process_fits(self, path):
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
            date_obs = h.get("DATE-OBS")

            ra_deg, dec_deg = self.parse_ra_dec(h)
            if ra_deg is None or dec_deg is None:
                return None

            lat = self.safe_float(h.get("OBSLAT"))
            lon = self.safe_float(h.get("OBSLON"))
            elev = self.safe_float(h.get("OBSALT"))
            if None in (lat, lon, elev):
                return None

            # Compute Alt/Az + Airmass
            sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elev * u.m)
            t = Time(date_obs)
            altaz = sky.transform_to(AltAz(obstime=t, location=loc))
            alt = altaz.alt.degree
            airmass = altaz.secz

            ra_str, dec_str = self.deg_to_hms_dms(ra_deg, dec_deg)

            # Create multi-line string for CSV
            block = (
                f"Object;        {obj}\n"
                f"RA (h:m:s);    {ra_str}\n"
                f"DEC (d:m:s);   {dec_str}\n"
                f"Date Obs;      {t.iso.split('.')[0]}\n"
                f"Altitude (°);  {alt:.2f}\n"
                f"Airmass;       {airmass:.2f}\n"
            )

            return airmass
