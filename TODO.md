## To-do for period-fitting update

This branch is intended to improve, refine and debug the period-amplitude relation fitting functions. sinusoid_period_fitting has been improved upon and outputs pleasing graphs and values when run. Changes were merged into main and a new branch has been created for further improvements following the exam period and Christmas break.

Targets:

- sawtooth_period_fitting must be updated to reflect sinusoid_period_fitting changes
- run_period_fit must be updated to compare the two models and pick the best one
- Need to write functions to parse data from photometry
- Refine error analysis
- Update plots to incorporate errors
- Start thinking about applying the data analysis to Andromeda CV1

Bug fixing:

- Rewrite code to take periods instead of frequency (ugh)