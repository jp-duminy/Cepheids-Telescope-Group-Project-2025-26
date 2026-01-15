"""
Do not run this, it is just a test idea for a full image reduction pipeline.
This is a combination  of claude, previous years and stack overflow with ideas from existing files.
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
import glob

class CalibrationSet:
    """
    Represents a matched set of calibrations (bias + flat) for a specific week/configuration.
    """
    def __init__(self, name, bias_dir, flat_dir, border=50):
        self.name = name
        self.bias_dir = Path(bias_dir) if bias_dir else None
        self.flat_dir = Path(flat_dir) if flat_dir else None
        self.border = border
        
        self._master_bias = None
        self._master_flat = None
            
    def _trim(self, data):
        """
        Trims desired number of pixels from image.
        """
        return data[self.border:-self.border, self.border:-self.border]
    
    def create_master_bias(self, bias_pattern="*.fits"):
        """
        Create master bias from directory.
        """
        if self._master_bias is not None: # if this already exists
            return self._master_bias
            
        bias_files = sorted(self.bias_dir.glob(bias_pattern))
        if len(bias_files) == 0:
            raise FileNotFoundError(f"No bias files in {self.bias_dir}") # diagnostic
        
        print(f"{len(bias_files)} bias frames in {self.bias_dir.name}...")
        bias_frames = [self._trim(fits.getdata(bf)) for bf in bias_files]
        bias_stack = np.stack(bias_frames, axis=0)
        self._master_bias = np.median(bias_stack, axis=0)
        print(f"Master bias shape: {self._master_bias.shape}")
        
        return self._master_bias
    
    def create_master_flat(self, flat_pattern="*.fits"):
        """
        Creates a master flat in the same way as create_master_bias.
        """
        if self._master_flat is not None: # if this already exists
            return self._master_flat
            
        flat_files = sorted(self.flat_dir.glob(flat_pattern))
        if len(flat_files) == 0:
            raise FileNotFoundError(f"No flat files in {self.flat_dir}") # diagnostic
        
        print(f"{len(flat_files)} flat frames in {self.flat_dir.name}...")
        flat_frames = [self._trim(fits.getdata(ff)) for ff in flat_files]
        
        stacked = np.median(flat_frames, axis=0)
        norm_factor = np.median(stacked)
        self._master_flat = stacked / norm_factor
        print(f"  Master flat shape: {self._master_flat.shape}, normalized")
        
        return self._master_flat
    
    def prepare(self):
        """
        Call this function to create the master bias and master flat frames found in specified directories.
        """
        self.create_master_bias()
        self.create_master_flat()
        return self

class CalibrationManager:
    """
    Manages calibration frames, mapping observation nights to appropriate calibrations.
    """
    def __init__(self, calibrations_dir):
        self.calib_dir = Path(calibrations_dir)
        self.calibration_sets = {}
        self.night_to_calib_map = {}

    def discover_calibrations(self, binning="1x1", filter_name="V"):
        """
        Automatically discover calibration directories.
        
        Parameters:
        -----------
        binning : str
            Binning mode, e.g., "1x1" or "2x2"
        filter_name : str
            Filter name, e.g., "V", "B", "R"
        """
        # Look for bias directories
        bias_pattern = f"*Bias*{binning}*"
        bias_dirs = sorted(self.calib_dir.glob(bias_pattern))
        
        # Look for flat directories
        flat_pattern = f"*Flat*{filter_name}*{binning}*"
        flat_dirs = sorted(self.calib_dir.glob(flat_pattern))
        
        print(f"\nDiscovered calibration directories:")
        print(f"  Bias dirs: {[d.name for d in bias_dirs]}")
        print(f"  Flat dirs: {[d.name for d in flat_dirs]}")
        
        return bias_dirs, flat_dirs
    
    def add_calibration_set(self, name, bias_dir, flat_dir):
        """Add a calibration set (e.g., for Week1, Week2, etc.)."""
        calib_set = CalibrationSet(name, bias_dir, flat_dir)
        self.calibration_sets[name] = calib_set
        return calib_set
    
    def map_night_to_calibration(self, night_name, calib_name):
        """
        Map an observation night to a calibration set.
        
        Parameters:
        -----------
        night_name : str
            Night directory name, e.g., "2025_09_22"
        calib_name : str
            Calibration set name, e.g., "Week1"
        """
        self.night_to_calib_map[night_name] = calib_name
    
    def get_calibration_for_night(self, night_name):
        """Get the appropriate calibration set for a given night."""
        if night_name not in self.night_to_calib_map:
            raise KeyError(f"No calibration mapping found for night {night_name}")
        
        calib_name = self.night_to_calib_map[night_name]
        
        if calib_name not in self.calibration_sets:
            raise KeyError(f"Calibration set '{calib_name}' not found")
        
        return self.calibration_sets[calib_name]
    
    def prepare_all(self):
        """Prepare all calibration sets."""
        print("\n" + "="*60)
        print("PREPARING CALIBRATION FRAMES")
        print("="*60)
        
        for name, calib_set in self.calibration_sets.items():
            print(f"\n{name}:")
            calib_set.prepare()


class CepheidDataOrganizer:
    """
    Organizes Cepheid files by night and Cepheid number.
    """
    def __init__(self, cepheids_dir):
        self.cepheids_dir = Path(cepheids_dir)
        self.cepheid_pattern = re.compile(r'Cepheids?_(\d+)', re.IGNORECASE)
    
    def list_observation_nights(self):
        """List all observation night directories."""
        nights = sorted([d for d in self.cepheids_dir.iterdir() if d.is_dir()])
        return nights
    
    def organize_night(self, night_dir):
        """
        Organize files for a single night.
        
        Returns:
        --------
        dict: {cepheid_num: [file_paths]}
        """
        cepheid_files = defaultdict(list)
        
        for fits_file in sorted(Path(night_dir).glob("*.fits")):
            match = self.cepheid_pattern.search(fits_file.name)
            if match:
                ceph_num = int(match.group(1))
                cepheid_files[ceph_num].append(fits_file)
        
        return dict(cepheid_files)
    
    def organize_all_nights(self):
        """
        Organize all nights.
        
        Returns:
        --------
        dict: {night_name: {cepheid_num: [file_paths]}}
        """
        all_data = {}
        nights = self.list_observation_nights()
        
        for night_dir in nights:
            night_name = night_dir.name
            cepheid_files = self.organize_night(night_dir)
            if cepheid_files:
                all_data[night_name] = cepheid_files
        
        return all_data
    
    @staticmethod
    def filter_useful_images(file_list, min_sequence=5, method='last_n'):
        """
        Identify the useful sequence of images.
        
        Parameters:
        -----------
        file_list : list
            List of FITS file paths
        min_sequence : int
            Minimum number of images to consider useful
        method : str
            'last_n': Take the last N images (assumes test images are first)
            'exposure_time': Group by exposure time, take largest group
            
        Returns:
        --------
        list: Filtered file paths
        """
        if len(file_list) < min_sequence:
            print(f"    Warning: Only {len(file_list)} images, less than minimum {min_sequence}")
            return file_list
        
        if method == 'last_n':
            # Simple approach: take last N images
            return file_list[-min_sequence:]
        
        elif method == 'exposure_time':
            # Group by exposure time, take largest consistent group
            exp_times = []
            for fits_file in file_list:
                with fits.open(fits_file) as hdul:
                    exp_time = hdul[0].header.get('EXPOSURE', None)
                    exp_times.append((fits_file, exp_time))
            
            # Group by exposure time
            exp_groups = defaultdict(list)
            for fits_file, exp_time in exp_times:
                if exp_time is not None:
                    exp_time_rounded = round(exp_time, 2)
                    exp_groups[exp_time_rounded].append(fits_file)
            
            # Find largest group
            largest_group = max(exp_groups.values(), key=len, default=[])
            
            if len(largest_group) >= min_sequence:
                return largest_group
            else:
                return file_list[-min_sequence:]  # Fallback
        
        return file_list


class CepheidImageReducer:
    """Reduces images using a specific calibration set."""
    
    def __init__(self, calibration_set, border=50):
        self.calib = calibration_set
        self.border = border
        
    def _trim(self, data):
        """Trim border pixels."""
        return data[self.border:-self.border, self.border:-self.border]
    
    def reduce_image(self, fits_path):
        """Apply reduction pipeline to single image."""
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
        
        # Trim
        trimmed = self._trim(data)
        
        # Get calibrations
        master_bias = self.calib._master_bias
        master_flat = self.calib._master_flat
        
        if master_bias is None or master_flat is None:
            raise RuntimeError("Calibrations not prepared")
        
        # Check dimensions
        if trimmed.shape != master_bias.shape:
            raise ValueError(f"Shape mismatch: {trimmed.shape} vs {master_bias.shape}")
        
        # Reduce
        bias_corrected = trimmed - master_bias
        flat_corrected = bias_corrected / master_flat
        
        return flat_corrected, header


class CepheidStacker:
    """Stacks reduced images."""
    
    @staticmethod
    def stack_images(image_list, headers_list, method='mean'):
        """Stack images and combine headers."""
        if len(image_list) == 0:
            raise ValueError("No images to stack")
        
        stack = np.stack(image_list, axis=0)
        stacked = np.mean(stack, axis=0) if method == 'mean' else np.median(stack, axis=0)
        
        base_header = headers_list[0].copy()
        base_header['TOTEXP'] = sum(h.get('EXPOSURE', 0) for h in headers_list)
        
        gains = [h.get('GAIN', 0) for h in headers_list if 'GAIN' in h]
        if gains:
            base_header['TOTGAIN'] = float(np.mean(gains))
        
        read_noises = [h.get('RDNOISE', 0) for h in headers_list if 'RDNOISE' in h]
        if read_noises:
            base_header['TOTRN'] = float(np.sqrt(np.sum(np.array(read_noises)**2)))
        
        base_header['NSTACK'] = len(image_list)
        base_header['COMMENT'] = f'Stacked {len(image_list)} images using {method}'
        
        return stacked, base_header


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_single_night_test(
    base_dir,
    night_name,
    week_name,
    cepheid_num,
    binning="1x1",
    filter_name="V",
    min_sequence=5,
    selection_method="last_n"
):
    base_path = Path(base_dir)
    cepheids_path = base_path / "Cepheids"
    calibrations_path = base_path / "Calibrations"

    print("="*70)
    print(f"TEST MODE — NIGHT {night_name}, CEPHEID {cepheid_num}")
    print("="*70)

    # ------------------------------------------------------------------
    # Locate calibration directories
    # ------------------------------------------------------------------
    bias_dir = None
    flat_dir = None

    for d in calibrations_path.iterdir():
        if d.is_dir():
            name = d.name.lower()
            if week_name.lower() in name and "bias" in name and binning in d.name:
                bias_dir = d
            if week_name.lower() in name and "flat" in name and filter_name in d.name and binning in d.name:
                flat_dir = d

    if bias_dir is None or flat_dir is None:
        raise RuntimeError("Could not locate calibration directories")

    print(f"Bias dir: {bias_dir.name}")
    print(f"Flat dir: {flat_dir.name}")

    calib = CalibrationSet(week_name, bias_dir, flat_dir)
    calib.prepare()

    # ------------------------------------------------------------------
    # Inspect master calibration frames
    # ------------------------------------------------------------------
    describe_array("MASTER BIAS", calib._master_bias)
    describe_array("MASTER FLAT", calib._master_flat)

    # ------------------------------------------------------------------
    # Locate Cepheid files
    # ------------------------------------------------------------------
    organizer = CepheidDataOrganizer(cepheids_path)
    night_dir = cepheids_path / night_name
    night_data = organizer.organize_night(night_dir)

    if cepheid_num not in night_data:
        raise RuntimeError(f"No files found for Cepheid {cepheid_num}")

    files = organizer.filter_useful_images(
        night_data[cepheid_num],
        min_sequence=min_sequence,
        method=selection_method
    )

    print(f"\nSelected {len(files)} images")

    reducer = CepheidImageReducer(calib)

    reduced_images = []
    headers = []

    for f in files:
        reduced, header = reducer.reduce_image(f)
        reduced_images.append(reduced)
        headers.append(header)

    # ------------------------------------------------------------------
    # Inspect a single reduced image
    # ------------------------------------------------------------------
    describe_array("SINGLE REDUCED IMAGE", reduced_images[0])

    # ------------------------------------------------------------------
    # Stack and inspect
    # ------------------------------------------------------------------
    stacked, _ = CepheidStacker.stack_images(reduced_images, headers, method="mean")
    describe_array("FINAL STACKED IMAGE", stacked)

    print("\nTEST COMPLETE")
def run_full_cepheid_pipeline(
    base_dir,
    night_to_week_mapping,
    binning="1x1",
    filter_name="V",
    output_dir=None,
    cepheid_nums=None,
    min_sequence=5,
    selection_method='last_n'
):
    """
    Complete pipeline for reducing Cepheid observations across multiple nights.
    
    Parameters:
    -----------
    base_dir : str or Path
        Base directory containing Cepheids/ and Calibrations/
    night_to_week_mapping : dict
        Maps night names to week names, e.g., 
        {'2025_09_22': 'Week1', '2025_09_23': 'Week1', '2025_09_29': 'Week2'}
    binning : str
        Binning mode
    filter_name : str
        Filter name
    output_dir : str or Path
        Where to save outputs (defaults to base_dir/Reduced)
    cepheid_nums : list
        List of Cepheid numbers to process (None = all)
    min_sequence : int
        Minimum number of images in useful sequence
    selection_method : str
        Method for selecting useful images ('last_n' or 'exposure_time')
    """
    base_path = Path(base_dir)
    cepheids_path = base_path / "Cepheids"
    calibrations_path = base_path / "Calibrations"
    
    if output_dir is None:
        output_dir = base_path / "Reduced"
    output_path = Path(output_dir)
    
    print("="*70)
    print("CEPHEID VARIABLE STAR REDUCTION PIPELINE")
    print("="*70)
    print(f"Base directory: {base_path}")
    print(f"Binning: {binning}, Filter: {filter_name}")
    print(f"Output directory: {output_path}")
    
    # ========================================================================
    # STEP 1: Setup calibration manager
    # ========================================================================
    calib_mgr = CalibrationManager(calibrations_path)
    
    # Get unique weeks from mapping
    weeks = sorted(set(night_to_week_mapping.values()))
    
    for week in weeks:
        # Find appropriate calibration directories
        bias_dir = None
        flat_dir = None
        
        for d in calibrations_path.iterdir():
            if d.is_dir():
                if week.lower() in d.name.lower() and 'bias' in d.name.lower() and binning in d.name:
                    bias_dir = d
                if week.lower() in d.name.lower() and 'flat' in d.name.lower() and filter_name in d.name and binning in d.name:
                    flat_dir = d
        
        if bias_dir is None or flat_dir is None:
            print(f"\nWarning: Could not find calibrations for {week}")
            print(f"  Bias dir: {bias_dir}")
            print(f"  Flat dir: {flat_dir}")
            continue
        
        calib_mgr.add_calibration_set(week, bias_dir, flat_dir)
    
    # Map nights to calibrations
    for night, week in night_to_week_mapping.items():
        calib_mgr.map_night_to_calibration(night, week)
    
    # Prepare all calibrations
    calib_mgr.prepare_all()
    
    # ========================================================================
    # STEP 2: Organize observation files
    # ========================================================================
    print("\n" + "="*70)
    print("ORGANIZING OBSERVATION FILES")
    print("="*70)
    
    organizer = CepheidDataOrganizer(cepheids_path)
    all_nights_data = organizer.organize_all_nights()
    
    print(f"\nFound {len(all_nights_data)} observation nights:")
    for night_name, ceph_data in all_nights_data.items():
        print(f"  {night_name}: {len(ceph_data)} Cepheids observed")
    
    # ========================================================================
    # STEP 3: Process each night
    # ========================================================================
    print("\n" + "="*70)
    print("REDUCING AND STACKING IMAGES")
    print("="*70)
    
    summary = defaultdict(lambda: defaultdict(int))
    
    for night_name, ceph_data in all_nights_data.items():
        print(f"\n{'='*70}")
        print(f"NIGHT: {night_name}")
        print(f"{'='*70}")
        
        # Get calibrations for this night
        try:
            calib_set = calib_mgr.get_calibration_for_night(night_name)
            print(f"Using calibrations: {calib_set.name}")
        except KeyError as e:
            print(f"Skipping night (no calibration mapping): {e}")
            continue
        
        reducer = CepheidImageReducer(calib_set)
        
        # Create output directory for this night
        night_output = output_path / night_name
        night_output.mkdir(parents=True, exist_ok=True)
        
        # Process each Cepheid
        for ceph_num, all_files in sorted(ceph_data.items()):
            # Filter to requested Cepheids
            if cepheid_nums is not None and ceph_num not in cepheid_nums:
                continue
            
            print(f"\n  Cepheid {ceph_num}:")
            print(f"    Total files found: {len(all_files)}")
            
            # Select useful images
            useful_files = organizer.filter_useful_images(
                all_files, 
                min_sequence=min_sequence,
                method=selection_method
            )
            print(f"    Selected {len(useful_files)} useful images")
            
            if len(useful_files) == 0:
                continue
            
            # Reduce each image
            reduced_images = []
            headers = []
            
            for i, fits_file in enumerate(useful_files, 1):
                try:
                    reduced, header = reducer.reduce_image(fits_file)
                    reduced_images.append(reduced)
                    headers.append(header)
                except Exception as e:
                    print(f"    Error on image {i}: {e}")
            
            if len(reduced_images) == 0:
                print(f"    No valid images after reduction")
                continue
            
            # Stack
            stacked, combined_header = CepheidStacker.stack_images(
                reduced_images, 
                headers, 
                method='mean'
            )
            
            # Save
            save_path = night_output / f"cepheid_{ceph_num:02d}_stacked.fits"
            hdu = fits.PrimaryHDU(stacked, header=combined_header)
            hdu.writeto(save_path, overwrite=True)
            
            total_exp = combined_header['TOTEXP']
            n_stacked = combined_header['NSTACK']
            
            print(f"    ✓ Stacked {n_stacked} images (total exp: {total_exp:.1f}s)")
            print(f"    ✓ Saved: {save_path.name}")
            
            # Update summary
            summary[ceph_num][night_name] = n_stacked
    
    # ========================================================================
    # STEP 4: Print summary
    # ========================================================================
    print("\n" + "="*70)
    print("REDUCTION SUMMARY")
    print("="*70)
    
    for ceph_num in sorted(summary.keys()):
        nights_data = summary[ceph_num]
        total_images = sum(nights_data.values())
        print(f"\nCepheid {ceph_num}:")
        print(f"  Total nights: {len(nights_data)}")
        print(f"  Total stacked images: {total_images}")
        for night, n_imgs in sorted(nights_data.items()):
            print(f"    {night}: {n_imgs} images")
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"Output saved to: {output_path}")
    print("="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
"""
if __name__ == "__main__":
    
    # Define which weeks' calibrations to use for each observation night
    night_to_week_mapping = {
        '2025_09_22': 'Week1',
        '2025_09_23': 'Week1',
        '2025_09_24': 'Week1',
        '2025_09_29': 'Week2',
        '2025_09_30': 'Week2',
        # Add more nights as needed
    }
"""
if __name__ == "__main__":

    run_single_night_test(
        base_dir="/storage/teaching/TelescopeGroupProject/2025-26",
        night_name="2025_09_22",
        week_name="Week1",
        cepheid_num=1,
        binning="1x1",
        filter_name="V",
        min_sequence=5,
        selection_method="last_n"
    )


from pathlib import Path

base = Path("/storage/teaching/TelescopeGroupProject/2025-26")

# List observation nights
nights = sorted((base / "Cepheids").iterdir())
print("Observation nights:")
for n in nights:
    print(f"  {n.name}")

# List calibration directories
calibs = sorted((base / "Calibrations").iterdir())
print("\nCalibration directories:")
for c in calibs:
    print(f"  {c.name}")