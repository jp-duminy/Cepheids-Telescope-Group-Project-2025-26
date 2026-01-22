import os
import re
from collections import defaultdict

class CepheidFileGrouper:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.pattern = re.compile(r"Cepheids_(\d+)_00")
        self.groups = defaultdict(list)

    def find_ceph(folder, nums):
        """
        Find all FITS files for the given Cepheid numbers.
    
        nums = list of numbers as strings or ints, e.g. ["7"] or [7]
        """
        nums = set(str(n) for n in nums)  # convert numbers to strings for matching
    
        pattern = re.compile(r"Cepheids_(\d+)_00")   # match only numeric Cepheids
        matches = []

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".fits"):
                continue

            m = pattern.search(fname)
            if not m:
                continue

            cepheid_number = m.group(1)

            # Keep only desired numeric Cepheids
            if cepheid_number in nums:
                matches.append(os.path.join(folder, fname))

        # sort for predictable order
        matches.sort()

        return matches
