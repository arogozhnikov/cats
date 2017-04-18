import numpy
__author__ = 'Alex Rogozhnikov'

#pythran export _compute_linear_factor(float32[], float32[], int32[])
#pythran export _compute_linear_factor(float64[], float64[], int32[])


def _compute_linear_factor(self_reprs, column_unit_reprs, column):
    return self_reprs - column_unit_reprs[column]

#pythran export _del_factor(float32[], int32[], float32[], float32[], float32[])
#pythran export _del_factor(float64[], int32[], float64[], float64[], float64[])
#pythran export _add_factor(float32[], int32[], float32[], float32[], float32[])
#pythran export _add_factor(float64[], int32[], float64[], float64[], float64[])


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


# #pythran export _activate_unit(float32[], float32[], float32[:, :] list, int, int32[:, :])
# #pythran export _activate_unit(float64[], float64[], float64[:, :] list, int, int32[:, :])
#
# def _activate_unit(self_reprs, self_reprs_sq, reprs, unit_id, X):
#     self_reprs[:] = 0
#     self_reprs_sq[:] = 0
#     for column_id in range(X.shape[1]):
#         column = X[:, column_id]
#         t = reprs[column_id][unit_id][column]
#         self_reprs += t
#         self_reprs_sq += t ** 2


# #pythran export _compute_grad_hess_repr(float32[], float32[], int32[], int)
# #pythran export _compute_grad_hess_repr(float64[], float64[], int32[], int)
#
# def _compute_grad_hess_repr(grad, multiplier, column, n_cats):
#     # return numpy.bincount(column, weights=grad * multiplier, minlength=n_cats), \
#     #        numpy.bincount(column, weights=multiplier ** 2, minlength=n_cats)
#     grads = numpy.zeros(n_cats, dtype='float32')
#     hesss = numpy.zeros(n_cats, dtype='float32')
#     for i in range(len(grad)):
#         index = column[i]
#         grads[index] += grad[i] * multiplier[i]
#         hesss[index] += multiplier[i] * multiplier[i]
#     return grads, hesss
