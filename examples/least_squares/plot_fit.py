# Plot least squares fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")

# Plot data
ax.errorbar(fit.x, fit.y, fit.ysigma, ls='none', marker='o', label="Data")
xmin = 1
xmax = 1.35

# Plot fit
xfit = np.linspace(start=xmin, stop=xmax, num=100)
yfit = fit.get_yfit(xfit)
ax.plot(xfit, yfit, ls='--', label="Fit")

# Plot error band
yfit_error = fit.get_yfit_error(xfit)
ax.fill_between(xfit, yfit - yfit_error, yfit + yfit_error, color='tab:orange', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()
