import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
import re
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
import warnings
    

class Calibration_Set:
    """
    Returns a master flat and bias for a given week+day configuration.
    """
    def __init__(self, name, bias_dir, flat_dir):
        self.name = name
        self.bias_dir = Path(bias_dir) if bias_dir else None
        self.flat_dir = Path(flat_dir) if flat_dir else None

        self.bias_files = []
        self.flat_files = []

        self.master_bias = None
        self.master_flat = None

    @staticmethod
    def img_trim(img):
        """
        Trims image by desired number of pixels.
        """
        trim = 50 # decided on 50 pixels
        return img[trim:-trim, trim:-trim]
    
    def create_master_bias(self):
        """
        Create master bias image from the stored bias file list.
        """
        if self.master_bias is not None:  # if this already exists
            return self.master_bias
        
        # Use the stored bias_files list instead of globbing
        if self.bias_files is None or len(self.bias_files) == 0:
            raise ValueError(f"No bias files specified for {self.name}")
        
        bias_files = sorted(self.bias_files)
        print(f"{len(bias_files)} bias files found in {self.bias_dir}.")

        bias_frames = [self.img_trim(fits.getdata(f)) for f in bias_files]
        bias_stack = np.stack(bias_frames, axis=0)
        self.master_bias = np.median(bias_stack, axis=0)
        print(f"Master bias shape: {self.master_bias.shape}")

        print(f"\nMaster bias statistics:")
        print(f"  Min: {np.min(self.master_bias):.1f} ADU")
        print(f"  Max: {np.max(self.master_bias):.1f} ADU")
        print(f"  Mean: {np.mean(self.master_bias):.1f} ADU")
        print(f"  Median: {np.median(self.master_bias):.1f} ADU")
        print(f"  Std: {np.std(self.master_bias):.1f} ADU")
        
        # Check for structure (bias should be fairly uniform)

        _, median, std_clipped = sigma_clipped_stats(self.master_bias, sigma=3.0)
        print(f"  Sigma-clipped std: {std_clipped:.1f} ADU")
        
        if std_clipped > 10:
            warnings.warn("Master bias has significant structure!")

        return self.master_bias

    def create_master_flat(self):
        """
        Create master flat image from the stored flat file list.
        """
        if self.master_flat is not None:  # if this already exists
            return self.master_flat

        if self.master_bias is None:
            self.create_master_bias() # since the flat-fielded images are bias-corrected we check whether master bias actually exists

        # Use the stored flat_files list instead of globbing
        if self.flat_files is None or len(self.flat_files) == 0:
            raise ValueError(f"No flat files specified for {self.name}")

        flat_files = sorted(self.flat_files)
        print(f"{len(flat_files)} flat files found in {self.flat_dir}.")

        flat_frames = [
        self.img_trim(fits.getdata(f)) - self.master_bias
        for f in flat_files]

        flat_frames_norm = [f / np.mean(f) for f in flat_frames]

        self.master_flat = np.median(np.stack(flat_frames_norm), axis=0)
        self.master_flat /=  np.mean(self.master_flat)
        print(f"Master flat shape: {self.master_flat.shape}")

        print(f"Master flat statistics:")
        print(f"  Min: {np.min(self.master_flat):.4f}")
        print(f"  Max: {np.max(self.master_flat):.4f}")
        print(f"  Mean: {np.mean(self.master_flat):.4f}")  # Should be ~1.0
        print(f"  Median: {np.median(self.master_flat):.4f}")  # Should be ~1.0
        
        if np.mean(self.master_flat) < 0.9 or np.mean(self.master_flat) > 1.1:
            warnings.warn("Master flat normalization is wrong!")

        return self.master_flat

    def prepare(self):
        """
        Prepare master bias and flat frames.
        """
        self.create_master_bias()
        self.create_master_flat()
        return self

class Calibration_Manager:
    """
    Organises calibration frames; maps weekly calibration frames to observation nights. 
    """
    def __init__(self, calibrations_dir):
        self.calib_dir = Path(calibrations_dir)
        self.calibration_sets = {}
        self.night_to_calib_map = {}

    def week_calibrations(self, week_name, binning="binning1x1", filter="V"):
        """
        Locate calibration file directories for a specific week.
        Binning is always 1x1 and filter is always V (for PIRATE telescope data)
        """
        week_dir = self.calib_dir / week_name / binning

        if not week_dir.exists():
            print(f"Warning: Calibration directory not found: {week_dir}")
            return None

        print(f"Located directory {week_dir}")

        bias_tag = "Bias"
        flat_tag = "flats"

        bias_files = list(week_dir.glob(f"*{bias_tag}*.fits"))
        flat_files = list(week_dir.glob(f"*{flat_tag}*{filter}*.fits"))

        if len(bias_files) == 0:
            print(f"Warning: No bias files found in {week_dir}")
            return None
    
        if len(flat_files) == 0:
            print(f"Warning: No flat files found for filter {filter} in {week_dir}")
            return None
        
        print(f"Found {len(bias_files)} bias files")
        print(f"Found {len(flat_files)} flat files for filter {filter}")

        # now create the set of calibrations for the given week
        calib_name = f"{week_name}_{filter}_{binning}"
        calib_set = Calibration_Set(calib_name, week_dir, week_dir)

        calib_set.bias_files = sorted(bias_files)
        calib_set.flat_files = sorted(flat_files)
    
        self.calibration_sets[week_name] = calib_set
    
        return calib_set
    
    def map_night_to_week(self, night_name, calib_name):
        """
        Map an observation night to its corresponding calibration set.
        Night name will be a file such as "2025_09_22" (first night)
        Calibration name will be the full calibration identifier (week, filter, binning)""
        """
        self.night_to_calib_map[night_name] = calib_name # assign the nights/weeks to a place in the dictionary in __init__

    def get_calibration_for_night(self, night_name):
        """
        Use map function to acquire the appropriate calibration set for a given night.
        Returns the index in the dictionary containing the correct calibration.
        """
        if night_name not in self.night_to_calib_map:
            raise KeyError(f"No calibration mapping found for night {night_name}")
        
        calib_name = self.night_to_calib_map[night_name]
        
        if calib_name not in self.calibration_sets:
            raise KeyError(f"Calibration set '{calib_name}' not found")
        
        return self.calibration_sets[calib_name]
    
    def prepare_all(self):
        """
        Prepares all master bias and flat images at once.
        """
        # diagnostic prints
        print("\n" + "="*60)
        print("PREPARING CALIBRATION FRAMES")
        print("="*60)
        
        # prepare all calibration frames
        for name, calib_set in self.calibration_sets.items():
            print(f"\n{name}:")
            calib_set.prepare()
 
class Cepheid_Data_Organiser:
    """
    Organises Cepheid files by number and night.
    """
    def __init__(self, cepheids_directory):
        """
        Create path to Cepheid directory and find patterns in files of "Cepheid_(#)".
        """
        self.cepheids_directory = Path(cepheids_directory)
        self.cepheid_pattern = re.compile(r'Cepheids?_(\d+).*?_Filter_V_', re.IGNORECASE)
    
    def list_observation_nights(self):
        """
        Sort all directories for nights in the Cepheids directory
        in alphabetical order
        """
        nights = sorted([night for night in self.cepheids_directory.iterdir() if night.is_dir()])
        return nights

    def organise_night(self, night_directory):
        """
        Organise each night's files based on Cepheid number
        """
        cepheid_files = defaultdict(list)
        
        for file in sorted(Path(night_directory).glob("*.fits")):
            pattern_presence = self.cepheid_pattern.search(file.name)
            if pattern_presence:
                ceph_num = int(pattern_presence.group(1))
                cepheid_files[ceph_num].append(file)
        
        return dict(cepheid_files)
    
    def organise_all_nights(self):
        """
        Organise all Cepheid files firstly by night, and then by Cepheid number. Returns
        a list of files that correspond to each night+Cepheid.
        """
        all_data = {}
        nights = self.list_observation_nights()
        
        for night in nights:
            night_name = night.name
            cepheid_files = self.organise_night(night)
            if cepheid_files:
                all_data[night_name] = cepheid_files
        
        return all_data
    
    @staticmethod
    def filter_useful_images(file_list):
        """
        Find the burst of images by grouping images by exposure time.
        """
        # extract exposure times
        exp_times = []
        for fits_file in file_list:
            with fits.open(fits_file) as hdul:
                exp_time = hdul[0].header.get('EXPTIME', None)
                exp_times.append((fits_file, exp_time))
        
        # group images by exposure time
        exp_groups = defaultdict(list)
        for fits_file, exp_time in exp_times:
            if exp_time is not None:
                exp_time_rounded = round(exp_time, 2)
                exp_groups[exp_time_rounded].append(fits_file)
        
        # now find the largest group (this is the burst)
        largest_group = max(exp_groups.values(), key=len, default=[])
        print(f"{len(largest_group)} images in largest group.")

        return largest_group

class Cepheid_Image_Reducer:
    """
    Apply image reduction to Cepheid .fits files
    """
    def __init__(self, calib_set, border=50):
        """
        Load in calibration frames and border to trim images by.
        """
        self.calib = calib_set
        self.border = border # 50px chosen

    def perform_reduction(self, filename):
        """
        Perform data reduction on a single image
        """

        with fits.open(filename) as hdul: # hdul is the header data unit list
            data = hdul[0].data # extract data from header data unit
            header = hdul[0].header.copy()

        # trim the pixels off the image
        trimmed = data[self.border:-self.border, self.border:-self.border]

        # check whether this is appropriate syntax?
        master_bias = self.calib.master_bias
        master_flat = self.calib.master_flat
        
        print(f"\n  Raw science frame:")
        print(f"    Min: {np.min(trimmed):.1f} ADU")
        print(f"    Mean: {np.mean(trimmed):.1f} ADU")
        print(f"    Negative: {np.sum(trimmed < 0)} pixels")

        if master_bias is None or master_flat is None:
            raise RuntimeError("Calibrations not prepared. Call prepare() first.")

        # check images are same size
        if trimmed.shape != master_bias.shape:
            raise ValueError(f"Shape mismatch: {trimmed.shape} vs {master_bias.shape}")

        # then apply corrections
        bias_corrected = trimmed - master_bias

            
        print(f"  After bias subtraction (bias mean={np.mean(master_bias):.1f}):")
        print(f"    Min: {np.min(bias_corrected):.1f} ADU")
        print(f"    Mean: {np.mean(bias_corrected):.1f} ADU")
        print(f"    Negative: {np.sum(bias_corrected < 0)} ({100*np.sum(bias_corrected<0)/bias_corrected.size:.1f}%)")

        flat_corrected = bias_corrected / master_flat

            
        print(f"  After flat correction:")
        print(f"    Min: {np.min(flat_corrected):.1f} ADU")
        print(f"    Mean: {np.mean(flat_corrected):.1f} ADU")
        print(f"    Negative: {np.sum(flat_corrected < 0)} ({100*np.sum(flat_corrected<0)/flat_corrected.size:.1f}%)")
        

        return flat_corrected, header
    
class Cepheid_Image_Stacker:
    """
    Stacks all science frames together to produce a single final image for each 
    night + Cepheid.
    """
    @staticmethod
    def stack_images(image_list, headers_list, method='median'):
        """
        Stacks images and combines headers for photometry.
        """
        image_stack = np.stack(image_list, axis=0)
        if method == 'mean':
            final_image = np.mean(image_stack, axis=0) # go with mean for now, potentially implement sigma clipping later
        else:
            final_image = np.median(image_stack, axis=0)

        base_header = headers_list[0].copy()
        base_header['TOTEXP'] = sum(h.get('EXPTIME', 0) for h in headers_list)
        base_header['MEANEXP'] = base_header['TOTEXP'] / len(headers_list)

        gains = [h.get('GAIN', 0) for h in headers_list if 'GAIN' in h] # check if gain is in headers
        if gains:
            base_header['MEANGAIN'] = float(np.mean(gains))

        read_noises = [h.get('RDNOISE', 0) for h in headers_list if 'RDNOISE' in h] # check if read noise is in headers
        if read_noises:
            base_header['TOTRN'] = float(np.sqrt(np.sum(np.array(read_noises)**2)))

        base_header['NSTACK'] = len(image_list)
        base_header['COMMENT'] = f'Stacked {len(image_list)} images using {method}'

        return final_image, base_header

def run_pipeline(
    base_dir,
    night_to_week_mapping,
    binning="binning1x1",
    filter_name="V",
    output_dir=None,
    cepheid_nums=None,
    visualise=True
):
    """
    Executes the complete cepheid reduction pipeline and displays final images.
    """

    # directories
    base_path = Path(base_dir)
    cepheids_path = base_path / "Cepheids"
    calibrations_path = base_path / "Calibrations"
    output_path = Path(output_dir)

    print("="*40)
    print("Beginning Cepheid Reduction")
    print("="*40)

    print("="*40)
    print(f"Creating Calibrations...")
    print("="*40)
    calib_mgr = Calibration_Manager(calibrations_path)
    
    # find unique weeks
    weeks = sorted(set(night_to_week_mapping.values()))
    print(f"\nWeeks to process: {weeks}") # expect 5 weeks!

    # create calibration for each week
    for week in weeks:
        calib_mgr.week_calibrations(week, binning=binning, filter=filter_name) 
    
    # map nights to corresponding weeks
    for night, week in night_to_week_mapping.items():
        calib_mgr.map_night_to_week(night, week)

    # prepare all calibrations
    calib_mgr.prepare_all()

    print("="*40)
    print(f"Organising files...")
    print("="*40)

    organizer = Cepheid_Data_Organiser(cepheids_path)
    all_nights_data = organizer.organise_all_nights()
    
    print(f"\nFound {len(all_nights_data)} observation nights:")
    for night_name, ceph_data in all_nights_data.items():
        mapping_status = "✓ mapped" if night_name in night_to_week_mapping else "✗ not mapped"
        print(f"  {night_name}: {len(ceph_data)} Cepheids observed [{mapping_status}]")

    print("="*40)
    print(f"Processing nights...")
    print("="*40)

    summary = defaultdict(lambda: defaultdict(int))
    stacked_images = {}

    for night_name, ceph_data in all_nights_data.items():
        # Skip nights without calibration mapping
        if night_name not in night_to_week_mapping:
            print(f"\n No calibration mapping for {night_name}")
            continue
        
        print("="*40)
        print(f"NIGHT: {night_name}")
        print("="*40)
    
        # get calibrations for night
        try:
            calib_set = calib_mgr.get_calibration_for_night(night_name)
            print(f"Using calibrations: {calib_set.name}")
        except KeyError as e:
            print(f"Skipping night: {e}")
            continue
        
        # create reducer object
        reducer = Cepheid_Image_Reducer(calib_set)
        
        # create output directory
        night_output = output_path / night_name
        night_output.mkdir(parents=True, exist_ok=True)
        
        # process each cepheid individually
        for ceph_num, all_files in sorted(ceph_data.items()):
            # only select requested cepheids
            if cepheid_nums is not None and ceph_num not in cepheid_nums:
                continue
            
            print(f"\nCepheid {ceph_num}:")
            print(f"Total files found: {len(all_files)}")
            
            # select useful images (from burst, grouped by exposure time)
            useful_files = organizer.filter_useful_images(all_files)
            print(f"Selected {len(useful_files)} useful images")
            
            if len(useful_files) == 0:
                print(f"No useful images found")
                continue
            
            # now reduce each image
            reduced_images = []
            headers = []
            
            for i, fits_file in enumerate(useful_files, 1):
                try:
                    reduced, header = reducer.perform_reduction(fits_file)
                    reduced_images.append(reduced)
                    headers.append(header)
                except Exception as e:
                    print(f"Error on image {i} ({fits_file.name}): {e}")
            
            if len(reduced_images) == 0:
                print(f"No valid images after reduction")
                continue
            
            # stack images
            stacked, combined_header = Cepheid_Image_Stacker.stack_images(
                reduced_images, 
                headers, 
                method='median'
            )
            
            # save file
            save_path = night_output / f"cepheid_{ceph_num:02d}_stacked.fits"
            hdu = fits.PrimaryHDU(stacked, header=combined_header)
            hdu.writeto(save_path, overwrite=True)
            
            total_exp = combined_header['TOTEXP']
            n_stacked = combined_header['NSTACK']
            
            print(f"✓ Stacked {n_stacked} images (total exp: {total_exp:.1f}s)")
            print(f"✓ Saved: {save_path.name}")
            
            # store data for visualisation
            if ceph_num not in stacked_images:
                stacked_images[ceph_num] = []
            stacked_images[ceph_num].append({
                'image': stacked,
                'night': night_name,
                'header': combined_header
            })
                
            # update summary
            summary[ceph_num][night_name] = n_stacked

    print("\n" + "="*70)
    print("Reduction summary diagnostics:")
    print("="*70)
    
    if len(summary) == 0:
        print("\nNo data was processed!")
    else:
        for ceph_num in sorted(summary.keys()):
            nights_data = summary[ceph_num]
            total_images = sum(nights_data.values())
            print(f"\nCepheid {ceph_num}:")
            print(f"  Total nights: {len(nights_data)}")
            print(f"  Total stacked images: {total_images}")
            for night, n_imgs in sorted(nights_data.items()):
                print(f"    {night}: {n_imgs} images")

    if visualise:
        quick_view_stacked_images(stacked_images)

    return summary, stacked_images
    
def quick_view_stacked_images(stacked_images):
    """
    Quick visualisation of stacked images without saving to disk.
    """
    if len(stacked_images) == 0:
        print("No images to display!")
        return
    
    zscale = ZScaleInterval()
    
    for ceph_num, night_data in sorted(stacked_images.items()):
        n_panels = len(night_data)
        n_cols = min(5, n_panels)  # Max 4 columns
        n_rows = (n_panels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for idx, entry in enumerate(night_data):
            image_to_show = entry['image']
            night_name = entry['night']
        
            # Apply zscale
            vmin, vmax = zscale.get_limits(image_to_show)
            
            # Plot
            ax = axes[idx]
            im = ax.imshow(image_to_show, cmap='gray', origin='lower', 
                        vmin=vmin, vmax=vmax)
            ax.set_title(f"{night_name}", fontsize=10)
            ax.axis('off')
            
            # Add colourbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide extra subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Cepheid {ceph_num}", fontsize=16)
        plt.tight_layout()
        plt.show()


# Usage:

if __name__ == "__main__":

    night_to_week_mapping = {
    '2025-09-22': 'week1',
    '2025-09-24': 'week1',
    '2025-09-29': 'week2',
    '2025-10-01': 'week2',
    '2025-10-06': 'week3',
    '2025-10-07': 'week3',
    '2025-10-08': 'week3',
    '2025-10-09': 'week3',
    '2025-10-13': 'week4',
    '2025-10-14': 'week4',
    '2025-10-19': 'week4',
    '2025-10-21': 'week5',
    '2025-10-22': 'week5',
    '2025-10-23': 'week5',
    # should be 14 observation nights (includes random TA observations)
    }

    cepheid_nums= [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    ]

    summary, images = run_pipeline(
    base_dir="/storage/teaching/TelescopeGroupProject/2025-26",
    night_to_week_mapping=night_to_week_mapping,
    output_dir = "/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids",
    cepheid_nums=cepheid_nums,
    visualise=True
    )








