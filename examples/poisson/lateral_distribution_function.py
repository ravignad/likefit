# Fit the number of particles measured by the muon detectors of Auger as
# function of the distance to the shower axis
# This example corresponds to a cosmic ray observed by three detectors

import numpy as np

# Install likefit if not available with: pip -m install likefit
import likefit

# Distance of the detectors to the shower axis in metres
distance = np.array([191, 263, 309])

# Number of measured particles in each detector
particles = np.array([33, 19, 11])


# Fit model
def fit_model(x, par):
    x0 = 250  # Reference distance in metres
    return par[0] * np.power(x/x0, -par[1])


# Fit data
fitter = likefit.Poisson(distance, particles, fit_model)
seed = np.array([30, 2])

# Reduce the tolerance to converge the minimization
fitter.fit(seed)
fitter.print_results()

# Plot data and fit
fitter.plot_fit(xlabel="Distance (m)", ylabel="Particles")

# Plot the 1σ, 2σ, and 3σ confidence regions
fitter.plot_confidence_regions(parx_index=0, pary_index=1, xlabel="Shower size", ylabel="Slope")
