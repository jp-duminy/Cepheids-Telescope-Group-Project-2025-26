import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
with fits.open("/storage/teaching/TelescopeGroupProject/2025-26/student-work/Cepheids/2025-09-22/cepheid_11_stacked.fits") as hdul:
            data_s = hdul[0].data
            header_s = hdul[0].header 

with fits.open("/storage/teaching/TelescopeGroupProject/2025-26/Cepheids/2025-10-07/PIRATE_165221_OSL_ROE_Cepheids_6_00_Z_Lac_00_Filter_V_00_2025_10_07_20_28_30.fits") as hdul:
        data0 = hdul[0].data
        header0 = hdul[0].header

print(header_s)


