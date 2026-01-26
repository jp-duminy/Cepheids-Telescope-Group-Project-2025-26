## To-do for period-fitting update

This branch is intended to improve, refine and debug the period-amplitude relation fitting functions. sinusoid_period_fitting has been improved upon and outputs pleasing graphs and values when run. Changes were merged into main and a new branch has been created for further improvements following the exam period and Christmas break.

Update 12/01: Branch work is underway, starting with refining sinusoid fitting and patching over changes into sawtooth fitting. 

Targets:

- ~~sawtooth_period_fitting must be updated to reflect sinusoid_period_fitting changes~~ 14/01
- run_period_fit must be updated to compare the two models and pick the best one (implement Bayesian Information Criterion)
- Need to write functions to parse data from photometry
- ~~Refine error analysis~~ 13/01
- Update plots to incorporate errors
- Start thinking about applying the data analysis to Andromeda CV1

Bug fixing:

- ~~Rewrite code to take periods instead of frequency (ugh)~~ 13/01
- ~~Clip walker positions to avoid walkers being initialised outside of parameter space~~ 15/01
- ~~Investigate and fix median sampling error~~ 13/01

Mimi test