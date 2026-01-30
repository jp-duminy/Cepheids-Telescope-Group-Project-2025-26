import numpy as np
import astropy.io.fits as fits
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

from Cepheid_apertures import AperturePhotometry

class Reduction_To_Photometry:
    """
    A class which will take in the files from reduction_pipeline and pass them on to
    Cepheid_apertures in the correct format
    """
    def __init__(self, file_dir):
        """
        Docstring for __init__
        
        :param self: Description
        :param file_dir: Description
        """
        self.file_dir = Path(file_dir)
        self.file_pattern = re.compile(r'cepheids?_(\d+).*\.fits$', re.IGNORECASE)
        self.dir_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})') # 4 digit year, 2 digit month, 2 digit day

    def date_dirs(self):
        # where are the date files stored
        base = self.file_dir
        date_dirs = [d for d in base.iterdir() if d.is_dir() and self.dir_pattern.match(d.name)]
        return date_dirs
    
    def list_observation_nights(self):
        nights = sorted([night for night in self.file_dir.iterdir() if night.is_dir()])
        return nights

    def sort_files(self):
        """
        sort the files in the directory created by the reduction pipeline
        """
        sorted_files = defaultdict(list)

        for file in Path(file_dir).glob("*.fits"):
            pattern_match = self.file_pattern.search(file.name)
            if pattern_match:
                ceph_num = int(pattern_match.group(1))
                sorted_files[ceph_num].append(file)
        return dict(sorted_files)
    
    def organise_all_data(self):
        """
        organise all data into a dictionary for each cepheid
        """
        all_data = {}
        nights = self.list_observation_nights()

        for night in nights:
            night_name = night.name
            sort_files = self.sort_files(night)
            if sort_files:
                all_data[night_name] = sort_files
        return all_data

    def do_photometry(self, file_path):
        # Going to call the Cepheid_aperture class and do photometry on each file from the reduction pipeline
        do_photometry = AperturePhotometry(file_path)
        return do_photometry
    
    def get_cepheid_id(self, file_path: Path):
        m = self.file_pattern.search(file_path.name)
        return int(m.group(1)) if m else None
    
    def save_photometry(self, photometry, source_file: Path, date: str, output_base: str):
        ceph_id = self.get_cepheid_id(source_file)
        out_dir = Path(output_base) / date / f"cepheid_{cepid:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{date}_cepheid{cepid:02d}_{source_file.stem}_phot.fits"
        out_path = out_dir / out_name

        return out_path


if __name__ == "__main__":

    ceph_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_test"

    output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheid_Photometry"

    data_output = output_dir / 
    
    for date_dir in Reduction_To_Photometry(ceph_dir).date_dirs():
        Reduction_To_Photometry(ceph_dir).process_date_dir(date_dir)
        for file in date_dir.glob('*.fits'):
            photometry = Reduction_To_Photometry(ceph_dir).do_photometry(file)
            save_path = r.save_photometry(photometry, file, date_dir.name, output_dir)
            