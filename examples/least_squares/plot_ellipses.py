# Plot confidence ellipses as function of two parameters

import numpy as np
import matplotlib.pyplot as plt

import likefit


# fit_model vectorized in x
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])

fitter = likefit.LeastSquares(xdata, ydata, ysigma, fit_model)
seed = np.array([0, 0])
fitter.fit(seed)

# Indexes of the parameters to plot
parx_index = 0
pary_index = 1

# Plot
fig, ax = plt.subplots()
ax.set_xlabel(f"Parameter {parx_index}")
ax.set_ylabel(f"Parameter {pary_index}")

estimators = fitter.get_estimators()
plt.plot(estimators[parx_index], estimators[pary_index], 'o', label="Estimator")

ellipse1_x, ellipse1_y = fitter.get_confidence_ellipse(parx_index, pary_index, nsigma=1)
plt.plot(ellipse1_x, ellipse1_y, label=r"1σ")
ellipse2_x, ellipse2_y = fitter.get_confidence_ellipse(parx_index, pary_index, nsigma=2)
plt.plot(ellipse2_x, ellipse2_y, label=r"2σ")

plt.legend()
plt.tight_layout()
plt.show()

# plt.savefig("ellipses.png")
