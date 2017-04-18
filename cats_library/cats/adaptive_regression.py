from __future__ import division, print_function, absolute_import

import numpy
import scipy.optimize
from sklearn.metrics import mean_squared_error

__author__ = 'Alex Rogozhnikov'


def optimization_target(train_gradient_sums, train_hessian_sums, test_gradient_sums,
                        test_hessian_sums, current_values, regularization):
    steps = (train_gradient_sums - regularization * current_values) / (train_hessian_sums + regularization)
    losses = - test_gradient_sums * steps + test_hessian_sums * steps ** 2 / 2.
    return losses.sum()


def optimize_regularization(train_gradient_sums, train_hessian_sums,
                            test_gradient_sums, test_hessian_sum,
                            current_values, current_reg):
    func = lambda r: optimization_target(train_gradient_sums, train_hessian_sums,
                                         test_gradient_sums, test_hessian_sum, current_values, 2 ** r)
    logreg = numpy.log2(current_reg).clip(1, 7)
    optimize_result = scipy.optimize.minimize_scalar(func, bounds=[logreg - 1, logreg + 1],
                                                     options={'maxiter': 5}, method='bounded')
    return 2 ** optimize_result['x']


class AdaptiveRegression:
    def __init__(self, n_iterations=10, learning_rate=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y, X_test, y_test):
        X = numpy.require(X, dtype='int32', requirements='F')
        X_test = numpy.require(X_test, dtype='int32', requirements='F')
        self.mean = numpy.mean(y)
        y = y - self.mean
        y_test = y_test - self.mean

        self.cat_biases = []
        for column in range(X.shape[1]):
            max_cats = max(numpy.max(X[:, column]), numpy.max(X_test[:, column])) + 1
            self.cat_biases.append(numpy.zeros(max_cats, dtype='float'))

        predictions = numpy.zeros(len(X))
        test_predictions = numpy.zeros(len(X_test))

        regularizations = numpy.zeros(X.shape[1]) + 20.

        train_hess = []
        test_hess = []
        for column in range(X.shape[1]):
            max_cats = len(self.cat_biases[column])
            inds = X[:, column]
            inds_test = X_test[:, column]
            train_hess_sums = numpy.bincount(inds, minlength=max_cats)
            test_hess_sums = numpy.bincount(inds_test, minlength=max_cats)
            train_hess.append(train_hess_sums)
            test_hess.append(test_hess_sums)

        for stage in range(self.n_iterations):
            for column in range(X.shape[1]):
                max_cats = len(self.cat_biases[column])
                grads = y - predictions
                grads_test = y_test - test_predictions

                inds = X[:, column]
                inds_test = X_test[:, column]

                train_grad_sums = numpy.bincount(inds, weights=grads, minlength=max_cats)
                test_grad_sums = numpy.bincount(inds_test, weights=grads_test, minlength=max_cats)
                train_hess_sums = train_hess[column]
                test_hess_sums = test_hess[column]

                current_values = self.cat_biases[column]
                new_regularization = optimize_regularization(train_grad_sums, train_hess_sums,
                                                             test_grad_sums, test_hess_sums, current_values,
                                                             current_reg=regularizations[column])
                reg = (regularizations[column] + new_regularization) / 2.
                regularizations[column] = reg

                updates = self.learning_rate * (train_grad_sums - reg * current_values) / (train_hess_sums + reg)
                self.cat_biases[column] += updates
                predictions += updates[inds]
                test_predictions += updates[inds_test]

            print (stage,
                   mean_squared_error(predictions, y) ** 0.5,
                   mean_squared_error(test_predictions, y_test) ** 0.5)

            print(regularizations)

            self.regularizations = regularizations
        return self

    def predict(self, X):
        X = numpy.array(X)
        predictions = numpy.zeros(len(X)) + self.mean
        for column in range(X.shape[1]):
            predictions += self.cat_biases[column][X[:, column]]
        return predictions
