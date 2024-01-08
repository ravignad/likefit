# Plot the likelihood cost as function of two parameters
# Two parameters are scanned and the others are left at the maximum likelihood estimator values
# (i. e. not the profile likelihood)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

import likefit


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])

fitter = likefit.NonLinearLeastSquares(xdata, ydata, ysigma, fit_model)
seed = np.array([0, 0])
fitter.fit(seed)

# Indexes of the parameters to plot
parx_index = 0
pary_index = 1

# Confidence levels to plot
nsigma = 2

# Calculate coordinates of the points to plot
estimators = fitter.get_estimators()
errors = fitter.get_errors()

parx_min = estimators[parx_index] - nsigma*errors[parx_index]
parx_max = estimators[parx_index] + nsigma*errors[parx_index]
parx = np.linspace(parx_min, parx_max, num=50)

pary_min = estimators[pary_index] - nsigma*errors[pary_index]
pary_max = estimators[pary_index] + nsigma*errors[pary_index]
pary = np.linspace(pary_min, pary_max, num=50)

x, y = np.meshgrid(parx, pary)
cost = fitter.vcost_function(parx_index, pary_index, parx, pary)
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

# plt.savefig("likelihood.png")
