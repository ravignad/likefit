# Plot the likelihood cost as function of two parameters
# Two parameters are scanned and the others are left at the maximum likelihood estimator values
# (i. e. not the profile likelihood)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

from likelihood import LeastSquares


# fit_model vectorized in x
def fit_model(x, theta):
    return theta[0] + theta[1] * (x-1.2) + theta[2] * (x-1.2) ** 2


input_file = "least_squares.csv"
data = pd.read_csv(input_file)

# Fit data
fit = LeastSquares(data["x"], data["y"], data["dy"], fit_model)
seed = np.array([1, 1, 1])
fit(seed)

# Indexes of the parameters to plot
parx_index = 0
pary_index = 1

# Confidence levels to plot
nsigma = 2

# Calculate coordinates of the points to plot
estimators = fit.get_estimators()
errors = fit.get_errors()

parx_min = estimators[parx_index] - nsigma*errors[parx_index]
parx_max = estimators[parx_index] + nsigma*errors[parx_index]
parx = np.linspace(parx_min, parx_max, num=50)

pary_min = estimators[pary_index] - nsigma*errors[pary_index]
pary_max = estimators[pary_index] + nsigma*errors[pary_index]
pary = np.linspace(pary_min, pary_max, num=50)

x, y = np.meshgrid(parx, pary)
cost = fit.vcost_function(parx_index, pary_index, parx, pary)
z = cost - cost.min()

# Plot
fig = plt.figure(figsize=(5, 4))
ax = fig.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel(f"Parameter {parx_index}")
ax.set_ylabel(f"Parameter {pary_index}")
ax.set_zlabel(r"$-2\log(L/L_{max})$")

# Levels of the countour lines
sigma_levels = np.arange(0, 7)
bounds = sigma_levels**2
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=128)
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, norm=norm)

# Plot colorbar
clb = plt.colorbar(surf, shrink=0.5, location='left')
clb.ax.set_title(r"$\sigma^2$")

plt.show()
