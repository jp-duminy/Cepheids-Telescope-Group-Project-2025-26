import numpy as np
from astropy.io import fits
from pathlib import Path
from collections import defaultdict
import re
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
import warnings
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
    

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
        self.cepheid_pattern = re.compile(r'Standard\d+_\d+_([^_]+(?:_\d+)?)', re.IGNORECASE)
    
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
        standard_files = defaultdict(list)
        
        for file in sorted(Path(night_directory).glob("*.fits")):
            pattern_presence = self.cepheid_pattern.search(file.name)
            if pattern_presence:
                standard_name = pattern_presence.group(1)
                standard_files[standard_name].append(file)
        
        return dict(standard_files)
    
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
    def align_images(image_list, reference_idx=0, max_shift_threshold=50):
        """
        Align images using cross-correlation to a reference image.
        
        Parameters
        ----------
        image_list : list
            List of 2D numpy arrays to align
        reference_idx : int
            Index of the reference image (default: 0, first image)
        max_shift_threshold : float
            Maximum allowed shift in pixels. Images with larger shifts are rejected.
            
        Returns
        -------
        aligned_images : list
            List of aligned images
        shifts : list
            List of (y_shift, x_shift) tuples for each image
        """
        reference = image_list[reference_idx]
        aligned_images = []
        shifts_list = []
        rejected_count = 0
        
        print(f"  Aligning {len(image_list)} images to reference (image {reference_idx+1})...")
        
        for i, image in enumerate(image_list):
            if i == reference_idx:
                # Reference image doesn't need alignment
                aligned_images.append(image)
                shifts_list.append((0.0, 0.0))
                continue
            
            try:
                # Calculate shift using phase cross-correlation with lower upsampling
                shift_yx, error, diffphase = phase_cross_correlation(
                    reference, image, upsample_factor=2  # Reduced from 10 to avoid over-interpolation
                )
                
                shift_magnitude = np.sqrt(shift_yx[0]**2 + shift_yx[1]**2)
                
                # Check if shift is reasonable
                if shift_magnitude > max_shift_threshold:
                    print(f"    Image {i+1}: REJECTED - shift too large ({shift_magnitude:.2f} > {max_shift_threshold} pixels)")
                    rejected_count += 1
                    continue
                
                # Apply shift with higher-order interpolation for better quality
                aligned = shift(image, shift_yx, mode='constant', cval=0.0, order=3)
                aligned_images.append(aligned)
                shifts_list.append(tuple(shift_yx))
                
                print(f"    Image {i+1}: shift = ({shift_yx[0]:.2f}, {shift_yx[1]:.2f}) pixels (magnitude: {shift_magnitude:.2f})")
                
            except Exception as e:
                print(f"    Warning: Failed to align image {i+1}: {e}")
                print(f"    Skipping this image.")
                rejected_count += 1
                continue
        
        if rejected_count > 0:
            print(f"  ⚠ {rejected_count} image(s) rejected due to alignment issues")
        
        return aligned_images, shifts_list
    
    @staticmethod
    def stack_images(image_list, headers_list, method='median', align=True):
        """
        Stacks images and combines headers for photometry.
        
        Parameters
        ----------
        image_list : list
            List of 2D numpy arrays to stack
        headers_list : list
            List of FITS headers
        method : str
            Stacking method: 'median' or 'mean'
        align : bool
            Whether to align images before stacking (default: True)
        """
        if len(image_list) == 0:
            raise ValueError("No images to stack!")
        
        original_count = len(image_list)
        
        if align and len(image_list) > 1:
            print(f"\n  Aligning images before stacking...")
            aligned_images, shifts = Cepheid_Image_Stacker.align_images(image_list)
            
            if len(aligned_images) == 0:
                print("  ⚠ All images rejected during alignment. Falling back to unaligned stacking.")
                aligned_images = image_list
            elif len(aligned_images) < original_count:
                print(f"  → Using {len(aligned_images)}/{original_count} images after alignment filtering")
                # Adjust headers list to match aligned images
                # This is tricky - we need to track which images were kept
                # For now, just use the first N headers
                headers_list = headers_list[:len(aligned_images)]
            else:
                max_shift = max([np.sqrt(s[0]**2 + s[1]**2) for s in shifts])
                print(f"  ✓ Maximum shift: {max_shift:.2f} pixels")
            
            image_list = aligned_images
        
        if len(image_list) == 0:
            raise ValueError("No valid images remaining after alignment!")
        
        image_stack = np.stack(image_list, axis=0)
        if method == 'mean':
            final_image = np.mean(image_stack, axis=0)
        else:
            final_image = np.median(image_stack, axis=0)

        base_header = headers_list[0].copy()
        base_header['TOTEXP'] = sum(h.get('EXPTIME', 0) for h in headers_list)
        base_header['MEANEXP'] = base_header['TOTEXP'] / len(headers_list)

        gains = [h.get('GAIN', 0) for h in headers_list if 'GAIN' in h]
        if gains:
            base_header['MEANGAIN'] = float(np.mean(gains))

        read_noises = [h.get('RDNOISE', 0) for h in headers_list if 'RDNOISE' in h]
        if read_noises:
            base_header['TOTRN'] = float(np.sqrt(np.sum(np.array(read_noises)**2)))

        base_header['NSTACK'] = len(image_list)
        base_header['ALIGNED'] = align
        base_header['COMMENT'] = f'Stacked {len(image_list)} images using {method}'
        if align:
            base_header['COMMENT'] = 'Images aligned before stacking'

        return final_image, base_header


def select_images_interactively(reduced_images, headers, fits_file_names=None):
    """
    Interactively select which images to include in the stack.
    
    Parameters
    ----------
    reduced_images : list
        List of reduced image arrays
    headers : list
        List of FITS headers corresponding to images
    fits_file_names : list, optional
        List of file names for reference
        
    Returns
    -------
    selected_images : list
        Filtered list of reduced images
    selected_headers : list
        Filtered list of headers
    """
    if len(reduced_images) == 0:
        return reduced_images, headers
    
    selected_indices = []
    
    for i, (image, header) in enumerate(zip(reduced_images, headers)):
        # Display image info
        fname = fits_file_names[i].name if fits_file_names else f"Image {i+1}"
        exp_time = header.get('EXPTIME', 'N/A')
        
        print(f"\n[{i+1}/{len(reduced_images)}] {fname}")
        print(f"  Exposure time: {exp_time}s")
        print(f"  Image min/max: {np.min(image):.0f} / {np.max(image):.0f} ADU")
        print(f"  Image mean: {np.mean(image):.0f} ADU, std: {np.std(image):.0f} ADU")
        
        # Ask user
        while True:
            response = input("  Include in stack? (y/n/skip): ").lower().strip()
            if response in ['y', 'yes', 'n', 'no', 's', 'skip']:
                break
            print("  Please enter 'y', 'n', or 'skip'")
        
        if response in ['y', 'yes']:
            selected_indices.append(i)
        elif response == 's':
            # 'skip' still includes the image (continue to next)
            selected_indices.append(i)
    
    # Filter based on selections
    selected_images = [reduced_images[i] for i in selected_indices]
    selected_headers = [headers[i] for i in selected_indices]
    
    print(f"\n→ Selected {len(selected_images)} out of {len(reduced_images)} images for stacking")
    
    return selected_images, selected_headers

def run_pipeline(
    base_dir,
    night_to_week_mapping,
    binning="binning1x1",
    filter_name="V",
    output_dir=None,
    cepheid_nums=None,
    visualise=True,
    target_week=None,
    interactive_selection=False,
    align_images=True
):
    """
    Executes the complete cepheid reduction pipeline and displays final images.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing Cepheids and Calibrations folders
    night_to_week_mapping : dict
        Mapping of night dates to week labels
    binning : str
        Binning mode (default: "binning1x1")
    filter_name : str
        Filter name (default: "V")
    output_dir : str
        Output directory for stacked images
    cepheid_nums : list
        List of cepheid numbers to process. If None, process all.
    visualise : bool
        Whether to visualize final stacked images (default: True)
    target_week : str
        Only process nights from this week (e.g., "week1"). If None, process all weeks.
    interactive_selection : bool
        If True, allow user to include/exclude images before stacking (default: False)
    align_images : bool
        If True, align images before stacking to avoid doubled stars (default: True)
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
    
    # Filter weeks if target_week specified
    if target_week is not None:
        if target_week not in weeks:
            raise ValueError(f"Target week '{target_week}' not found in mapping. Available: {weeks}")
        weeks = [target_week]
        print(f"Filtering to target week: {target_week}")

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
        
        # Filter by week if target_week specified
        current_week = night_to_week_mapping[night_name]
        if target_week is not None and current_week != target_week:
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
            # Check if any file contains the standard designation (e.g., "Standard1", "Standard2")
            if cepheid_nums is not None:
                should_process = False
                for fits_file in all_files:
                    filename = fits_file.name
                    # Check if this file matches one of the requested standards
                    for ceph in cepheid_nums:
                        if f"Standard{ceph}" in filename:
                            should_process = True
                            break
                    if should_process:
                        break
                
                if not should_process:
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
            
            # Interactive image selection if enabled
            if interactive_selection:
                print(f"\nInteractive mode: Select images to include in stack")
                print(f"Total reduced images: {len(reduced_images)}")
                reduced_images, headers = select_images_interactively(
                    reduced_images, headers, fits_file_names=useful_files
                )
            
            if len(reduced_images) == 0:
                print(f"No images selected for stacking")
                continue
            
            # stack images
            stacked, combined_header = Cepheid_Image_Stacker.stack_images(
                reduced_images, 
                headers, 
                method='median',
                align=align_images
            )
            
            # save file
            save_path = night_output / f"standard_{ceph_num}_stacked.fits"
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

    # EXAMPLE 1: Standard 1 only, week 1 only, with interactive image selection
    print("\n" + "="*70)
    print("EXAMPLE 1: Standard 1, Week 1, with interactive selection")
    print("="*70)
    summary, images = run_pipeline(
        base_dir="/storage/teaching/TelescopeGroupProject/2025-26",
        night_to_week_mapping=night_to_week_mapping,
        output_dir="/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids",
        cepheid_nums=[1],  # Only standard 1
        target_week='week1',  # Only week 1
        interactive_selection=True,  # Allow include/exclude before stacking
        align_images=False,  # Set to True to enable alignment (may blur if shifts are large)
        visualise=True
    )

    # EXAMPLE 2: Standard 1, week 1, automatic (non-interactive)
    # Uncomment to use instead of Example 1:
    # print("\n" + "="*70)
    # print("EXAMPLE 2: Standard 1, Week 1, automatic")
    # print("="*70)
    # summary, images = run_pipeline(
    #     base_dir="/storage/teaching/TelescopeGroupProject/2025-26",
    #     night_to_week_mapping=night_to_week_mapping,
    #     output_dir="/storage/teaching/TelescopeGroupProject/2025-26/student-work/",
    #     cepheid_nums=[1],  # Only standard 1
    #     target_week='week1',  # Only week 1
    #     interactive_selection=False,  # Automatic stacking
    #     visualise=True
    # )






