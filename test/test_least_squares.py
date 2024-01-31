import pytest
import numpy as np
import likefit

xdata = np.array([0., 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
ydata = np.array([0.92, 0.884, 0.626, 0.504, 0.481, 0.417, 0.288, 0.302, 0.177, 0.13, 0.158])
ysigma = np.array([0.1, 0.082, 0.067, 0.055, 0.045, 0.037, 0.03, 0.025, 0.02, 0.017, 0.014])


# fit_model vectorized in xdata
def fit_model(x, par):
    return par[0] * np.exp(par[1] * x)


@pytest.fixture
def least_squares_fitter():
    return likefit.LeastSquares(xdata, ydata, ysigma, fit_model)


def test_initialization(least_squares_fitter):
    assert np.array_equal(least_squares_fitter.xdata, xdata)
    assert np.array_equal(least_squares_fitter.ydata, ydata)
    assert np.array_equal(least_squares_fitter.ydata_error, ysigma)
    assert least_squares_fitter.model is fit_model


def test_fit(least_squares_fitter):
    seed = np.array([0, 0])
    assert least_squares_fitter.fit(seed) == 0
