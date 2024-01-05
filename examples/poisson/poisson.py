# Example of fitting data with the least squares method

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import likefit

xdata = np.linspace(start=-2.9, stop=2.9, num=30)
nevents = np.array([0, 2, 5, 8, 7, 18, 15, 27, 34, 51, 55, 63, 67, 75, 90, 78, 73, 70, 62, 51, 33, 26, 30, 17, 15, 14, 5,
                    4, 1, 0])


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * norm.pdf(x, loc=par[1], scale=par[2])


fitter = likefit.Poisson(xdata, nevents, fit_model)
seed = np.array([1, 0, 1])
fitter.fit(seed)

print(f"Estimators: {fitter.get_estimators()}")
print(f"Errors: {fitter.get_errors()}")
print(f"Covariance matrix: {fitter.get_covariance_matrix()}")
print(f"Correlation matrix: {fitter.get_correlation_matrix()}")
print(f"Deviance: {fitter.get_deviance()}")
print(f"Degrees of freedom: {fitter.get_ndof()}")
print(f"Pvalue: {fitter.get_pvalue()}")

# Plot
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")

# Plot data
ax.plot(xdata, nevents, ls='none', marker='o', label="Data")

# Plot fitter
xmin = xdata.min()
xmax = xdata.max()
xfit = np.linspace(start=xmin, stop=xmax, num=100)
yfit = fitter.get_yfit(xfit)
ax.plot(xfit, yfit, ls='--', label="Fit")

# Plot error band
yfit_error = fitter.get_yfit_error(xfit)
ax.fill_between(xfit, yfit - yfit_error, yfit + yfit_error, color='tab:orange', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()
