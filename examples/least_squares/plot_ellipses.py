# Plot confidence ellipses as function of two parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from likelihood import LeastSquares


# fit_model vectorized in x
def fit_model(x, theta):
    return theta[0] * np.exp(theta[1]*x)


input_file = "least_squares.csv"
data = pd.read_csv(input_file)

fit = LeastSquares(data["x"], data["y"], data["dy"], fit_model)
seed = np.array([0, 0])
fit(seed)

# Indexes of the parameters to plot
parx_index = 0
pary_index = 1

# Plot
fig, ax = plt.subplots()
ax.set_xlabel(f"Parameter {parx_index}")
ax.set_ylabel(f"Parameter {pary_index}")

estimators = fit.get_estimators()
plt.plot(estimators[parx_index], estimators[pary_index], 'o', label="Estimator")

ellipse1_x, ellipse1_y = fit.get_confidence_ellipse(parx_index, pary_index, nsigma=1)
plt.plot(ellipse1_x, ellipse1_y, label=r"1σ")
ellipse2_x, ellipse2_y = fit.get_confidence_ellipse(parx_index, pary_index, nsigma=2)
plt.plot(ellipse2_x, ellipse2_y, label=r"2σ")

plt.legend()
plt.tight_layout()
plt.show()
