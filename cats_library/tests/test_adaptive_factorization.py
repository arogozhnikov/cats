from __future__ import division, print_function, absolute_import

__author__ = 'Alex Rogozhnikov'

import numpy
from cats.fm_fast import DatasetRepresentation, FastFM
from sklearn.metrics import mean_squared_error


def test_representation():
    # region tests
    _n_samples = 10000
    _n_columns = 5
    _n_cats = 10
    _n_units = 0
    _cat_biases = numpy.random.normal(size=[_n_columns, _n_cats])
    _cat_reprs = numpy.random.normal(size=[_n_columns, _n_units, _n_cats])
    _X = numpy.random.randint(0, _n_cats, size=[_n_samples, _n_columns])
    _y = numpy.random.normal(size=_n_samples)
    _cat_biases_original = _cat_biases.copy()
    _cat_reprs_original = _cat_reprs.copy()
    # create dataset representation
    dataset = DatasetRepresentation(_X, _y, _n_units, biases=_cat_biases, reprs=_cat_reprs)
    initial_predictions = dataset.get_predictions().copy()
    initial_biass = dataset.biass.copy()

    # trivial check with removing single element
    if _n_units > 1 and _n_columns > 1:
        dataset.remove_factor(_cat_reprs, column_id=1, unit_id=1)
        dataset.add_factor(_cat_reprs, column_id=1, unit_id=1)
    assert numpy.allclose(initial_predictions, dataset.get_predictions())

    # delete all categories and biases
    for column_id in range(_n_columns):
        dataset.remove_bias(_cat_biases, column_id=column_id)
        _cat_biases[column_id] -= _cat_biases_original[column_id]
        dataset.check_consistency(reprs=_cat_reprs)
        for unit_id in range(_n_units):
            dataset.remove_factor(_cat_reprs, column_id=column_id, unit_id=unit_id)
            _cat_reprs[column_id, unit_id] -= _cat_reprs_original[column_id, unit_id]
            dataset.check_consistency(reprs=_cat_reprs)

    # check that we have 0 prediction
    assert numpy.allclose(dataset.biass, 0)
    assert numpy.allclose(dataset.get_predictions(), 0)
    assert numpy.allclose(dataset.reprs, 0)
    assert numpy.allclose(dataset.reprs_sq, 0)
    # restore, check that we have the same prediction as in the beginning
    for column_id in range(_n_columns):
        _cat_biases[column_id] += _cat_biases_original[column_id]
        dataset.add_bias(_cat_biases, column_id=column_id)
        dataset.check_consistency(reprs=_cat_reprs)
        _ = dataset.grad_hess_bias_total(column_id)

        for unit_id in range(_n_units):
            dataset.activate_unit(unit_id=unit_id, reprs=_cat_reprs)

            _cat_reprs[column_id, unit_id] += _cat_reprs_original[column_id, unit_id]
            print('column', column_id, 'unit', unit_id)
            dataset.add_factor(_cat_reprs, column_id=column_id, unit_id=unit_id)

            dataset.check_consistency(reprs=_cat_reprs)
            _ = dataset.grad_hess_repr_total(_cat_reprs, column_id=column_id, unit_id=unit_id)

    assert numpy.allclose(initial_biass, dataset.biass)
    print(initial_predictions)
    print(dataset.get_predictions())
    assert numpy.allclose(initial_predictions, dataset.get_predictions())
    print("checks passed")


def test_fastfm(n_samples=10000, n_categories=10):
    X = numpy.random.randint(0, n_categories, size=[n_samples, 2])
    y = numpy.array(X.sum(axis=1) % 2)
    clf = FastFM(n_iterations=10)
    clf.fit(X, y, X, y)
    pred = clf.decision_function(X)
    print(numpy.var(y) ** 0.5)
    print(mean_squared_error(y, pred) ** 0.5)

    assert mean_squared_error(y, pred) < numpy.var(pred) / 3
    print('FastFM successfully checked')
