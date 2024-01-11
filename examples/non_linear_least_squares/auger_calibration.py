# A simplified energy calibration of the Auger surface detector
# Data taken from Pierre Auger Collaboration, Auger Open Data, DOI:10.5281/zenodo.4487612

import numpy as np
import pandas as pd

# likefit can be installed with: pip -m install likefit
import likefit

# Import the Auger calibration data
data = pd.read_csv("auger_calibration.csv")

# In this example both the shower size and the energy have errors
# We choose the energy as the x variable because its errors are smaller than the shower size ones
xdata = data["energy"]
ydata = data["shower_size"]
yerror = data["shower_size_error"]


# Calculate the energy from the shower size
# par[0]: energy at the reference shower size
# par[1]: index of the power law used for the calibration
def fit_model(energy, par):
    size_0 = 30          # Reference shower size
    energy_0 = par[0]
    power_law_index = par[1]
    return size_0 * np.power(energy / energy_0, 1 / power_law_index)


'''
    By the selection of the energy as the x variable and the shower size as y variable, 
    the fit model is the inverse of the calibration function
'''

# We fit with non-linear least squares because the fit model is not linear in the parameters
# and the shower size assigned to the y variable follows a normal distribution
fitter = likefit.NonLinearLeastSquares(xdata, ydata, yerror, fit_model)

# The convergence of the fit depends heavily on choosing a seed close to the minimum
seed = np.array([7, 1])
fitter.fit(seed)
fitter.print_results()

# Plot the data and the fit
fitter.plot_fit(xlabel="Energy (EeV)", ylabel="Shower size (VEM)")

# Plot the 1σ and 2σ confidence regions
fitter.plot_confidence_regions(parx_index=0, pary_index=1, xlabel="${E}_0$ (EeV)", ylabel="B")
