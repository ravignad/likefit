# test_linear_least_squares.py
import numpy as np
import pytest
from likefit import LinearLeastSquares

# Sample data for testing
xdata = np.array([-0.18, -0.14, -0.1, -0.06, -0.02,  0.02,  0.06,  0.1,  0.14])
ydata = np.array([2.243, 2.217, 2.201, 2.175, 2.132, 2.116, 2.083, 2.016, 2.004])
ysigma = np.array([0.008, 0.008, 0.01, 0.009, 0.011, 0.016, 0.018, 0.021, 0.017])


# Model linear in the parameters
def fit_model(x, par):
    return par[0] + par[1] * x


@pytest.fixture
def linear_least_squares_fitter():
    return LinearLeastSquares(xdata, ydata, ysigma, fit_model)


def test_initialization(linear_least_squares_fitter):
    assert np.array_equal(linear_least_squares_fitter.xdata, xdata)
    assert np.array_equal(linear_least_squares_fitter.ydata, ydata)
    assert np.array_equal(linear_least_squares_fitter.ydata_error, ysigma)
    assert linear_least_squares_fitter.model is fit_model
    model_matrix = np.array([np.ones_like(xdata), xdata])
    assert np.array_equal(linear_least_squares_fitter.model_matrix, model_matrix)


def test_count_parameters(linear_least_squares_fitter):
    npar = linear_least_squares_fitter._LinearLeastSquares__count_parameters()
    assert npar == 2  # Assuming a simple linear model with two parameters (slope and intercept)


def test_fit(linear_least_squares_fitter):
    linear_least_squares_fitter.fit()
    estimators = np.array([2.1192891, -0.7269067])
    assert np.array_equal(linear_least_squares_fitter.estimators.round(7), estimators)
