## Update 27/01

### Small Tasks:

- run_period_fit must be updated to compare the two models and pick the best one (implement Bayesian Information Criterion)
- Need to write functions to parse data from photometry
- Update plots to incorporate errors (can extract from chains!)
- Start thinking about applying the data analysis to Andromeda CV1

### Big Ones:

- BIC is useful but also compare to literature & eyeball the fit; allow flexibility (reduced chi-squares)
- Once best model is picked, run emcee with desired model
- Store chains (as dictionaries) and check whether they converged
- Sample randomly from cepheid chains; period and magnitude (midline)
- Marginalised likelihood model for P-L relation (use literature values)
- Run emcee on P-L in a similar way to cepheid
