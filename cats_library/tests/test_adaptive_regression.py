from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import numpy
from cats.adaptive_regression import AdaptiveRegression
from . import generate_dataset
from sklearn.metrics import mean_squared_error


def test_regression():
    X, y = generate_dataset()

    lr = AdaptiveRegression()
    lr.fit(X, y, X, y)

    assert mean_squared_error(lr.predict(X), y) < numpy.var(y)
