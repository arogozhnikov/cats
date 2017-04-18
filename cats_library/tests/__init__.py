from __future__ import division, print_function, absolute_import
import numpy

__author__ = 'Alex Rogozhnikov'


def generate_dataset(n_samples=10000, n_features=3, n_categories=10):
    data = numpy.random.randint(0, n_categories, size=[n_samples, n_features])
    cat_predictions = numpy.random.normal(size=n_categories)
    y = cat_predictions[data].sum(axis=1)
    return data, y

