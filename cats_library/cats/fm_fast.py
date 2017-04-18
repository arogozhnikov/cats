"""
This modification should use less memory
- hashing trick for categories for cross-terms (or simply ignoring those?)
+ keeping track of only one column at a time
- speed up with pythran (?)

"""
from __future__ import division, print_function, absolute_import

import numpy
import scipy.optimize
from six import print_ as print
from .fm_python_utils import  _compute_linear_factor


# temporarily here
def _compute_grad_hess_repr(grad, multiplier, column, n_cats):
    return numpy.bincount(column, weights=grad * multiplier, minlength=n_cats), \
           numpy.bincount(column, weights=multiplier ** 2, minlength=n_cats)

def _activate_unit(self_reprs, self_reprs_sq, reprs, unit_id, X):
    self_reprs[:] = 0
    self_reprs_sq[:] = 0
    for column_id in range(X.shape[1]):
        column = X[:, column_id]
        t = reprs[column_id][unit_id][column]
        self_reprs += t
        self_reprs_sq += t ** 2

def _compute_test_loss(train_gradient_sums, train_hessian_sums, test_gradient_sums, test_hessian_sums,
                       regularization):
    """ Computes optimal loss of second order, given regularization """
    steps = train_gradient_sums / (train_hessian_sums + regularization)
    losses = - test_gradient_sums * steps + test_hessian_sums * steps ** 2 / 2.
    return numpy.sum(losses)

def _del_factor(cat_lookup, column, self_reprs, self_reprs_sq, self_y_minus_pred):
    cat_change = cat_lookup[column]
    self_reprs_sq -= cat_change ** 2
    self_reprs -= cat_change
    self_y_minus_pred += 2 * self_reprs * cat_change


def _add_factor(cat_lookup, column, self_reprs, self_reprs_sq, self_y_minus_pred):
    cat_change = cat_lookup[column]
    self_reprs_sq += cat_change ** 2
    self_y_minus_pred -= 2 * self_reprs * cat_change
    self_reprs += cat_change

# dtype for all the floats
_float_type = 'float64'

__author__ = 'Alex Rogozhnikov'


class DatasetRepresentation(object):
    def __init__(self, X, y, n_units, biases, reprs):
        """
        biases[column_id][categories]
        reprs[column_id][unit_id][categories]
        """
        self.X = numpy.require(X, requirements='F', dtype='int32')
        self.y = y.copy().astype(_float_type)
        self.n_units = n_units
        self.n_columns = X.shape[1]
        n_samples = len(self.X)

        self.n_categories = [len(b) for b in biases]

        self.biass = numpy.zeros(n_samples, dtype=_float_type)
        # only one at a time
        self.reprs = numpy.zeros(n_samples, dtype=_float_type)
        self.reprs_sq = numpy.zeros(n_samples, dtype=_float_type)

        self.activated_unit = None

        for column in range(self.n_columns):
            self.biass += biases[column][self.X[:, column]]

        predictions = self.biass.copy()

        for unit_id in range(self.n_units):
            self.activate_unit(unit_id, reprs)
            predictions += self.reprs ** 2 - self.reprs_sq

        self.y_minus_pred = self.y - predictions

    def check_consistency(self, reprs):
        alternative_predictions = self.biass.copy()
        for unit_id in range(self.n_units):
            self.activate_unit(unit_id, reprs)
            alternative_predictions += self.reprs ** 2 - self.reprs_sq

        assert numpy.allclose(self.get_predictions(), alternative_predictions, atol=1e-5)

    def activate_unit(self, unit_id, reprs):
        """Computes reprs and reprs_sq for particular unit"""
        if self.activated_unit == unit_id:
            # nothing to do!
            return
        else:
            self.activated_unit = unit_id
            print("activate_unit", flush=True)
            _activate_unit(self.reprs, self.reprs_sq, reprs, unit_id, self.X)

    def get_predictions(self):
        return self.y - self.y_minus_pred

    def remove_bias(self, biases, column_id):
        column = self.X[:, column_id]
        bias_change = biases[column_id][column]
        self.biass -= bias_change
        self.y_minus_pred += bias_change

    def add_bias(self, biases, column_id):
        column = self.X[:, column_id]
        bias_change = biases[column_id][column]
        self.biass += bias_change
        self.y_minus_pred -= bias_change

    def _grad_hess(self):
        return self.y_minus_pred, numpy.ones(len(self.y))

    def grad_hess_bias(self):
        return self._grad_hess()

    def remove_factor(self, reprs, column_id, unit_id):
        self.activate_unit(unit_id, reprs)
        column = self.X[:, column_id]
        cat_lookup = reprs[column_id][unit_id]
        print("del factor", flush=True)
        _del_factor(cat_lookup, column, self.reprs, self.reprs_sq, self.y_minus_pred)

    def add_factor(self, reprs, column_id, unit_id):
        assert self.activated_unit == unit_id
        column = self.X[:, column_id]

        cat_lookup = reprs[column_id][unit_id]
        print("add factor", flush=True)
        _add_factor(cat_lookup, column, self.reprs, self.reprs_sq, self.y_minus_pred)

    def _compute_linear_factor(self, reprs, column_id, unit_id):
        self.activate_unit(unit_id, reprs)
        column = self.X[:, column_id]
        column_unit_reprs = reprs[column_id][unit_id]
        # return self.reprs - column_unit_reprs[column]
        print("linear factor", flush=True)
        return _compute_linear_factor(self.reprs, column_unit_reprs, column)

    def grad_hess_bias_total(self, column_id):
        n_cats = self.n_categories[column_id]
        column = self.X[:, column_id]
        grad, hess = self.grad_hess_bias()
        return [numpy.bincount(column, weights=grad, minlength=n_cats),
                numpy.bincount(column, weights=hess, minlength=n_cats)]

    def grad_hess_repr_total(self, reprs, column_id, unit_id):
        n_cats = self.n_categories[column_id]
        column = self.X[:, column_id]
        grad = self.y_minus_pred

        multiplier = 2 * self._compute_linear_factor(reprs, column_id=column_id, unit_id=unit_id)
        print("gradhess", flush=True)
        return _compute_grad_hess_repr(grad, multiplier, column, n_cats)

    def rmse(self):
        return (self.y_minus_pred ** 2).mean() ** 0.5


# endregion


# region optimization

def fm_optimize_regularization(train_gradient_sums, train_hessian_sums, test_gradient_sums, test_hessian_sum,
                               current_regularization):
    print("minimize", flush=True)
    reg_log = numpy.log2(current_regularization)
    test_loss = lambda r: _compute_test_loss(train_gradient_sums, train_hessian_sums,
                                             test_gradient_sums, test_hessian_sum, 2 ** r)
    optimize_result = scipy.optimize.minimize_scalar(test_loss, bounds=[reg_log - 1, reg_log + 1],
                                                     options={'maxiter': 3}, method='bounded')
    return 2 ** optimize_result['x']


# endregion optimization


class FastFM:
    def __init__(self, learning_rate=1.0, n_units=10, n_iterations=30, sign=+1):
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.n_iterations = n_iterations
        self.sign = sign
        self.initialized = False

    def initialize_parameters(self, X, X_test):
        if self.initialized:
            print('continue training')
        else:
            self.n_columns = X.shape[1]
            self.cat_biases = []
            self.cat_representations = []
            self.repr_regularizations = []
            self.bias_regularizations = []
            for column_id in range(self.n_columns):
                n_cats = max(max(X[:, column_id]), max(X_test[:, column_id])) + 1
                self.cat_biases.append(numpy.zeros(n_cats, dtype=_float_type))
                _cat_reprs = numpy.random.normal(size=[self.n_units, n_cats]) * 0.1
                self.cat_representations.append(_cat_reprs.astype(_float_type))

                self.repr_regularizations.append(numpy.zeros(self.n_units) + 100.)
                self.bias_regularizations.append(10.)
            print('starting training')

    def staged_fit(self, X, y, X_test, y_test):
        X = numpy.require(X, requirements='F', dtype='int32')
        X_test = numpy.require(X_test, requirements='F', dtype='int32')

        self.mean = numpy.mean(y)
        y = self.sign * (y - self.mean)
        y_test = self.sign * (y_test - self.mean)

        self.initialize_parameters(X, X_test)

        train_repr = DatasetRepresentation(X, y, self.n_units, self.cat_biases, self.cat_representations)
        test_repr = DatasetRepresentation(X_test, y_test, self.n_units, self.cat_biases, self.cat_representations)

        for iteration in range(self.n_iterations):
            for column_id in range(self.n_columns):
                self.update_bias_for_column(column_id, test_repr, train_repr)

            for unit_id in range(self.n_units):
                for column_id in range(self.n_columns):
                    self.update_repr_for_column_and_unit(column_id, test_repr, train_repr, unit_id)

            yield train_repr, test_repr

    def fit(self, X, y, X_test, y_test):
        for i, (train_repr, test_repr) in enumerate(self.staged_fit(X, y, X_test, y_test)):
            print(test_repr.rmse())
            print(self.repr_regularizations)

        return self

    def update_bias_for_column(self, column_id, test_repr, train_repr):
        # fitting biases with exact step
        train_repr.remove_bias(self.cat_biases, column_id=column_id)
        test_repr.remove_bias(self.cat_biases, column_id=column_id)
        train_grads, train_hesss = train_repr.grad_hess_bias_total(column_id)
        test_grads, test_hesss = test_repr.grad_hess_bias_total(column_id)

        current_reg = self.bias_regularizations[column_id]
        l2_reg = fm_optimize_regularization(train_grads, train_hesss, test_grads, test_hesss, current_reg)
        self.bias_regularizations[column_id] = l2_reg = (self.bias_regularizations[column_id] + l2_reg) / 2.
        new_values = train_grads / (train_hesss + l2_reg)

        self.cat_biases[column_id] = new_values
        train_repr.add_bias(self.cat_biases, column_id=column_id)
        test_repr.add_bias(self.cat_biases, column_id=column_id)

    def update_repr_for_column_and_unit(self, column_id, test_repr, train_repr, unit_id):
        train_repr.remove_factor(self.cat_representations, column_id=column_id, unit_id=unit_id)
        test_repr.remove_factor(self.cat_representations, column_id=column_id, unit_id=unit_id)
        train_grads, train_hesss = train_repr.grad_hess_repr_total(
            self.cat_representations, column_id=column_id, unit_id=unit_id)
        test_grads, test_hesss = test_repr.grad_hess_repr_total(
            self.cat_representations, column_id=column_id, unit_id=unit_id)

        current_reg = self.repr_regularizations[column_id][unit_id]
        l2_reg = fm_optimize_regularization(train_grads, train_hesss, test_grads, test_hesss, current_reg)
        self.repr_regularizations[column_id][unit_id] = l2_reg = \
            (self.repr_regularizations[column_id][unit_id] + l2_reg + 10) / 2.
        new_values = train_grads / (train_hesss + l2_reg)

        self.cat_representations[column_id][unit_id] = (new_values + self.cat_representations[column_id][unit_id]) / 2.
        train_repr.add_factor(self.cat_representations, column_id=column_id, unit_id=unit_id)
        test_repr.add_factor(self.cat_representations, column_id=column_id, unit_id=unit_id)

    def decision_function(self, X):
        representation = DatasetRepresentation(X, numpy.zeros(len(X)), self.n_units,
                                               biases=self.cat_biases, reprs=self.cat_representations)
        return representation.get_predictions() * self.sign + self.mean
