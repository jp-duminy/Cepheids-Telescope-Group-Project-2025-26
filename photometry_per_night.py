import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
import re
from matplotlib import pyplot as plt
from astropy.io import fits

from Cepheid_apertures import AperturePhotometry
from Cepheid_apertures import Airmass
from Cepheid_apertures import DustExtinction
from AirmassInfo import AirmassInfo

import scienceplots
plt.style.use('science')
plt.rcParams['text.usetex'] = False

# someone please double check these

# cepheids and their RA/Dec coords (I did this by hand and double checked)
cepheid_catalogue = {
    "01": {
        "ra":  "20 12 22.83",
        "dec": "+32 52 17.8",
        "e(b-v)": "0.68",
        "name": "MW Cyg"
    },
    "02": {
        "ra":  "20 54 57.53",
        "dec": "+47 32 01.7",
        "e(b-v)": "0.80",
        "name": "V520 Cyg"
    },
    "03": {
        "ra":  "20 57 20.83",
        "dec": "+40 10 39.1",
        "e(b-v)": "0.79",
        "name": "VX Cyg"
    },
    "04": {
        "ra":  "21 04 16.63",
        "dec": "+39 58 20.1",
        "e(b-v)": "0.65",
        "name": "VY Cyg"
    },
    "05": {
        "ra":  "21 57 52.69",
        "dec": "+56 09 50.0",
        "e(b-v)": "0.68",
        "name": "CP Cep"
    },
    "06": {
        "ra":  "22 40 52.15",
        "dec": "+56 49 46.1",
        "e(b-v)": "0.40",
        "name": "Z Lac"
    },
    "07": {
        "ra":  "22 48 38.00",
        "dec": "+56 19 17.5",
        "e(b-v)": "0.36",
        "name": "V Lac"
    },
    "08": {
        "ra":  "23 07 10.08",
        "dec": "+58 33 15.1",
        "e(b-v)": "0.49",
        "name": "SW Cas"
    },
    "09": {
        "ra":  "00 26 19.45",
        "dec": "+51 16 49.3",
        "e(b-v)": "0.12",
        "name": "TU Cas"
    },
    "10": {
        "ra":  "00 29 58.59",
        "dec": "+60 12 43.1",
        "e(b-v)": "0.53",
        "name": "DL Cas"
    },
    "11": {
        "ra":  "01 32 43.22",
        "dec": "+63 35 37.7",
        "e(b-v)": "0.70",
        "name": "V636 Cas"
    },
}

# standard star catalogue, I also did this by hand
standard_catalogue = {
    "114176": {
        "ra":  "+22 43 11.0",
        "dec": "+00 21 16.0",
        "mag": "9.239",
        "e(b-v)": "0.0013"
    },
    "SA111775": {
        "ra":  "+19 37 17.0",
        "dec": "+00 11 14.0",
        "mag": "10.74",
        "e(b-v)": "	0.0009"
    },
    "F_108": {
        "ra":  "+23 16 12.0",
        "dec": "-01 50 35.0",
        "mag": "12.96",
        "e(b-v)": "0.0016"
    },
    "SA112_595": {
        "ra":  "+20 41 19.0",
        "dec": "+00 16 11.0",
        "mag": "11.35",
        "e(b-v)": "0.0016"
    },
    "GD_246": {
        "ra":  "+23 12 21.6",
        "dec": "+10 47 04.0",
        "mag": "13.09",
        "e(b-v)": "0.0015"
    },
    "G93_48": {
        "ra":  "+21 52 25.4",
        "dec": "+02 23 23.0",
        "mag": "12.74",
        "e(b-v)": "0.0012"
    },
    "G156_31": {
        "ra":  "+22 38 28.0",
        "dec": "-15 19 17.0",
        "mag": "12.36",
        "e(b-v)": "0.0049"
    }
}

andromeda_catalogue = {
    "CV1": {
        "ra": "+00 41 27.30",
        "dec": "+41 10 10.4",
        "e(b-v)": "0.06",
        "name": "Andromeda CV1"
    }
}

# directories
input_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-09-22" # change to the name of the night
output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/Photometry" # where we store the results

class PhotometryDataManager:

    def __init__(self, input_dir, output_dir):
        """
        Store parameters for I/O.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.date = Path(input_dir).name

    def find_cepheid_files(self):
        """
        Locates reduced cepheid files and returns a list.
        """
        return sorted(self.input_dir.glob("cepheid_*_stacked.fits"))
    
    def find_standard_files(self):
        """
        Locates reduced standard files and returns a list.
        """
        return sorted(self.input_dir.glob("standard_*_stacked.fits"))
    
    def extract_cepheid_id(self, filename):
        """
        Extracts the number from the cepheid file name so they may be distinguished.
        """
        match = re.search(r'cepheid[_-]?(\d+)', filename.lower())
        return f"{int(match.group(1)):02d}" if match else None
    
    def extract_standard_id(self, filename):
        """
        Extracts the ID of a standard star (more tricky as these are separated by name)
        """
        filename_lower = filename.lower()
        
        # Try to match each catalog key
        for std_id in standard_catalogue.keys():
            # Make comparison case-insensitive and flexible with underscores
            std_id_pattern = std_id.lower().replace("_", "[_-]?")
            if re.search(std_id_pattern, filename_lower):
                return std_id
        
        return None
    
    def save_results(self, df, filename):
        path = self.output_dir / filename
        df.to_csv(path, index=False)
        return path
    
class SinglePhotometry:
    """
    Performs photometry on one .fits file.
    """
    def __init__(self, fits_path, ra, dec, name, ebv):
        self.fits_path = Path(fits_path)
        self.name = name
        self.ebv = ebv
        self.ra = ra
        self.dec = dec

        self.ap = AperturePhotometry(str(fits_path))

        self.ccd_params()

    def ccd_params(self):
        """
        Extract parameters relevant to CCD equation.
        """
        with fits.open(self.fits_path) as hdul:
            hdr = hdul[0].header  # assuming primary HDU stores relevant keywords
    
            gain = hdr.get('MEANGAIN', None)           # e-/ADU
            read_noise = hdr.get('TOTRN', None)  # e-
            exp_time = hdr.get('TOTEXP', None)    # seconds
            stack_size = hdr.get('NSTACK', 1)    # default to 1 if single image
            
        self.gain = gain
        self.read_noise = read_noise
        self.exp_time = exp_time
        self.stack_size = stack_size

    def get_airmass(self):
        """
        Use .fits file headers to compute the airmass.
        """
        airmass_info = AirmassInfo(str(self.fits_path))
        airmass = airmass_info.process_fits(str(self.fits_path))
        
        return airmass
    
    def curve_of_growth(self, data, centroid, fwhm, inner=1.5, outer=2.0, plot=False):
        """
        Determine the optimal aperture size from a curve-of-growth plot.
        """
        ap_radii = np.arange(0.1, 4.1, 0.1) * fwhm
        fluxes = []
        bckgnds = []

        for radius in ap_radii:
            flux, _, bckgnd, _ = self.ap.aperture_photometry(
                data, centroid, ap_rad=radius, 
                inner=inner, outer=outer, plot=False
            )[0]
            fluxes.append(flux if flux is not None else 0)
            bckgnds.append(bckgnd)

        bckgnds = np.array(bckgnds)
        normalised_ssaf = bckgnds / np.max(bckgnds)

        fluxes = np.array(fluxes)
        max_flux = np.max(fluxes)
        target_flux = 0.90 * max_flux

        idx = np.argmax(fluxes >= target_flux)
        optimal_radius = ap_radii[idx]

        if plot:
            plt.plot(ap_radii, normalised_ssaf, color='red', marker='o')
            plt.xlabel('Aperture Radius [pix]')
            plt.ylabel('Sky-Subtracted Flux through Aperture [Arbitrary Units]')
            plt.title("Normalised curve of growth")
            plt.show()

        return optimal_radius


    def raw_photometry(self, width=100):
        """
        Raw photometry (computes instrumental magnitude and associated error.)
        """
        # locate approximate pixel coordinates of star
        x_guess, y_guess = self.ap.get_pixel_coords(self.ra, self.dec)

        # cut out a 100x100 rectangle containing the star
        masked_data = self.ap.mask_data_and_plot(x_guess, y_guess, width, plot=True)

        centroid, fwhm = self.ap.get_centroid_and_fwhm(masked_data, plot=True)

        ap_rad = self.curve_of_growth(masked_data, centroid, fwhm, inner=1.5, outer=2.0, plot=True)

        flux, ap_area, sky_bckgnd, annulus_area = self.ap.aperture_photometry(
            masked_data, centroid, ap_rad, ceph_name=self.name, date=self.date,
            inner=1.5, outer=2.0, plot=True, savefig=False
        )

        instrumental_mag = self.ap.instrumental_magnitude(flux)
        instrumental_mag_error = self.ap.get_inst_mag_error(flux, ap_area, sky_bckgnd, annulus_area,
                                                            self.gain, self.exp_time, self.read_noise, self.stack_size)
        
        return instrumental_mag, instrumental_mag_error

    def dust_correction(self):
        """
        Uses The Don's dust class to compute Av.
        """
        dust = DustExtinction(filter="V", colour_excess=float(self.ebv))
        A_V = dust.compute_extinction_mag()

        return A_V

    def standard_magnitudes(self, calibrations):
        """
        Compute a standard magnitude, appropriately corrected.
        """
        m_inst, m_inst_err = self.raw_photometry(width=100)
        airmass = self.get_airmass()
        A_V = self.dust_correction()

        k = calibrations["k"]
        Z1 = calibrations["Z1"]
        k_err = calibrations["k_err"]
        Z1_err = calibrations["Z1_err"]

        m_true = (m_inst - A_V) + Z1 + (k * airmass)
        m_true_err = np.sqrt(m_inst_err**2 + (airmass * k_err)**2 + Z1_err**2)

        return m_true, m_true_err
    
class Corrections:

    def __init__(self, data_manager, standard_catalogue, cepheid_catalogue):
        self.data_manager = data_manager
        self.standard_catalogue = standard_catalogue
        self.cepheid_catalogue = cepheid_catalogue
        
        self.standards_results = []
        self.calibration = None

    def process_standards(self):
        """
        Compute instrumental magnitude and its error for standard stars on the night.
        """
        standard_files = self.data_manager.find_standard_files()
        print(f"\nProcessing {len(standard_files)} standard stars...")

        for std_file in standard_files:
            # identify the standard and get its data
            std_id = self.data_manager.extract_standard_id(std_file.name)
            std_data = self.standard_catalogue[std_id]
            
            # initialise photometry object
            phot = SinglePhotometry(
            fits_path=std_file,
            ra=std_data["ra"],
            dec=std_data["dec"],
            name=std_id,
            ebv=0.0
            )

            m_inst, m_err = phot.raw_photometry(width=100)
            airmass = phot.get_airmass()

            self.standards_results.append({
            "name": std_id,
            "V_true": float(std_data["mag"]),
            "m_inst": m_inst,
            "m_inst_err": m_err,
            "airmass": airmass
            })

        self.standards_df = pd.DataFrame(self.standards_results)
    
    def fit_extinction(self):
        """
        Use Harsha's airmass class to fit extinction.
        """
        airmass_fitter = Airmass(
            airmass = self.standards_df['airmass'].values,
            Vmag=self.standards_df["V_true"].values,
            m_inst=self.standards_df["m_inst"].values,
            m_err=self.standards_df["m_inst_err"].values
        )

        k, Z1, k_err, Z1_err = airmass_fitter.fit_extinction_weighted()
        airmass_fitter.plot_atmospheric_extinction()
        airmass_fitter.plot_parameter_space()

        calibration = {
        "k": k,
        "Z1": Z1,
        "k_err": k_err,
        "Z1_err": Z1_err
        }

        return calibration
    
def main(input_dir=input_dir, output_dir=output_dir):
    # initialise data manager
    data_manager = PhotometryDataManager(input_dir, output_dir)
    
    # process standard stars to extract calibration parameters
    corrections = Corrections(data_manager, standard_catalogue, cepheid_catalogue)
    corrections.process_standards()
    calibration = corrections.fit_extinction()
    
    # loop over cepheids for a given night
    cepheid_files = data_manager.find_cepheid_files()
    results = []
    
    for cep_file in cepheid_files:
        cep_id = data_manager.extract_cepheid_id(cep_file.name)
        cep_data = cepheid_catalogue[cep_id]
        
        phot = SinglePhotometry(
            fits_path=cep_file,
            ra=cep_data["ra"],
            dec=cep_data["dec"],
            name=cep_data["name"],
            ebv=cep_data["e(b-v)"]
        ) # photometry object
        
        m_true, m_err = phot.standard_magnitudes(calibration)
        results.append({
            "ID": cep_id,
            "Name": cep_data["name"],
            "m_true": m_true,
            "m_err": m_err
        }) # can extend this to include ISOT & absolute magnitude
    
    # save results to a csv
    df = pd.DataFrame(results)
    filename = f"photometry_{data_manager.date}.csv"
    saved_path = data_manager.save_results(df, filename)
    
    print(f"Photometry complete. Results saved to: {saved_path}")

if __name__ == "__main__":
    main()



