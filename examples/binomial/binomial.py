# Example of fitting data with the least squares method

import numpy as np
import matplotlib.pyplot as plt
from likefit import Binomial

xdata = np.arange(start=0.05, stop=1.05, step=0.05)
ntrials = np.full(xdata.shape, 30)
nsuccess = np.array([0, 0, 0, 3, 3, 2, 8, 5, 4, 11, 18, 15, 19, 20, 26, 24, 26, 29, 30, 30])


# fit_model is sigmoid function vectorized in x
def fit_model(x, par):
    return 1 / (1+np.exp(-(x-par[0])/par[1]))


fitter = Binomial(xdata, ntrials, nsuccess, fit_model)
seed = np.array([0.5, 1])
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
ydata = nsuccess / ntrials
ax.plot(xdata, ydata, ls='none', marker='o', label="Data")

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
