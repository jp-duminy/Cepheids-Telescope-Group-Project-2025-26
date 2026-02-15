### References
Standard star SA 111775 is a model star for the photometry pipeline:
- Crowded field, yet still located with DAOStarFind
- Centroiding works perfectly
- Beautiful CoG/PSF
- Final aperture is visibly correct

CP Cep is also nice; Z Lac has the best-looking centroid.

### General Notes
- F108 has a very noisy background but demonstrates sub-pixel centroiding.
- Initial guess is wrong for SA 112 595 (no starfind)
- G93 48 identified as troublesome star (removed automatically in airmass correction step)
- G156 31 image is wrong (translated image in stack, used different night for plots)
- Pulled out a sky-subtracted, stacked + reduced MW Cyg from 2025-10-06 for the presentation/report
- I have saved a Cepheid 2 (V520 Cyg) reference stars image to demonstrate our flipping works
- Cepheid 7 (V Lac) & Cepheid 8 (SW Cas) have horrendous PSFs

### For Methodology
- Centroiding uses a more aggressive sigma-clipped background subtraction to determine the centroid with sub-pixel accuracy (this is demonstrated in the centroid plots)
- There is no plot of the sigma-clipped sky annulus; this must be explicitly described but is not visualised.
- Reference night is 2025-10-06, this determines the 'default' FLIPSTAT direction
- Want a finder chart to compare to DAOStarFinder to demonstrate our methodology works
