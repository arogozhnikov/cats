import numpy
import pandas
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.special 
from sklearn.metrics import log_loss
from hep_ml.losses import LogLossFunction

def generate_connections(n_categories, n_units, n_overlap=6):
    max_connections = int(numpy.ceil( (n_units * n_overlap) / float(n_categories) ))
    connections = numpy.zeros([n_categories, n_units], dtype=int)
    for unit in range(n_units):
        p = max_connections * (unit + 2.) / n_units - connections.sum(axis=1)
        p = p.clip(0)
        p = p / numpy.sum(p)
        categories = numpy.random.choice(n_categories, size=n_overlap, p=p, replace=False)
        connections[categories, unit] = 1
    return connections

class StallsFM(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=1.0, regularization=100., n_units=10, iterations=30, 
                 n_thresholds=10, max_overlap=20, sign=+1):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.bias_regularization = regularization * 0.1
        self.n_units = n_units
        self.iterations = iterations
        self.n_thresholds = n_thresholds
        self.loss = LogLossFunction()
        self.max_overlap = max_overlap
        self.sign = sign
        self.unit_signs = numpy.ones(n_units) * sign
        
    def decompose_data(self, X, fit=False):
        # hack to support both pandas and numpy.arrays
        X = pandas.DataFrame(X)
        
        if fit:
            self.is_sequential = numpy.array([column.dtype == 'float' for name, column in X.iteritems()])
            self.codings = []
            self.codings.append([0])
            for name, column in X.iteritems():
                if column.dtype == 'float':
                    self.codings.append(numpy.percentile(column, numpy.linspace(0, 100, self.n_thresholds + 1)[1:-1]))
                else:
                    self.codings.append(numpy.unique(column))

        X_categoricals = []
        X_categoricals.append(numpy.zeros(len(X), dtype=int))
        for is_seq, coding, (name, column) in zip(self.is_sequential, self.codings[1:], X.iteritems()):
            if is_seq:
                X_categoricals.append(numpy.searchsorted(coding, column))
            else:
                X_categoricals.append((numpy.searchsorted(coding, column) + 1) * numpy.in1d(column, coding))                
        return numpy.array(X_categoricals).T
    
    def compute_grad_hess(self, predictions):
        return self.loss.negative_gradient(predictions), self.loss.hessian(predictions)
            
    def fit(self, X, y):
        self.classes_, y = numpy.unique(y, return_inverse=True)
        assert len(self.classes_) == 2, 'only two classes supported'
        X_cat = self.decompose_data(X, fit=True)
        
        self.cat_biases = [numpy.zeros(len(coding) + 1) for coding in self.codings]
        self.cat_representations = [numpy.random.normal(size=[len(coding) + 1, self.n_units]) * 0.1 
                                    for coding in self.codings]
        self.connections = numpy.zeros([X_cat.shape[1], self.n_units])
        max_overlap = min(self.max_overlap, X_cat.shape[1])
        self.connections[:] = generate_connections(X_cat.shape[1], self.n_units, n_overlap=max_overlap)
        
        return self.partial_fit(X, y, restart=True)
    
    def partial_fit(self, X, y, restart=False):
        assert isinstance(X, pandas.DataFrame), 'only pandas.DataFrames are accepted'
        assert numpy.in1d(y, self.classes_).all()
        y = numpy.searchsorted(self.classes_, y)

        assert len(X) == len(y)
        
        self.loss.fit(X, y, sample_weight=numpy.ones_like(y))
        X_cat = self.decompose_data(X, fit=False)

        unit_signs = self.unit_signs
        self.losses = []

        for iteration in range(self.iterations):
            if iteration % 1 == 0:
                biases, representations, representations_sq = self.compute_representations(X_cat)
                new_predictions = self.compute_prediction(biases, representations, representations_sq, unit_signs)
                if iteration > 0:
                    assert numpy.allclose(predictions, new_predictions)
                predictions = new_predictions

            for category_biases, category_representations, column, connection in \
                    zip(self.cat_biases, self.cat_representations, X_cat.T, self.connections):

                # fitting biases with exact step
                minlen = len(category_biases)
                grads, hesss = self.compute_grad_hess(predictions)
                total_grads = numpy.bincount(column, weights=grads, minlength=minlen)
                total_hesss = numpy.bincount(column, weights=hesss, minlength=minlen)
                updates = (total_grads - self.bias_regularization * category_biases) / (total_hesss + self.bias_regularization)
                category_biases[:] += updates
                biases += updates[column]
                predictions += updates[column]
                
                for unit in numpy.arange(self.n_units):
                    unit_sign = unit_signs[unit]
                    if unit_sign == 0 or connection[unit] == 0:
                        continue
                    grads, hesss = self.compute_grad_hess(predictions)
                    predictions -= unit_sign * representations[:, unit] ** 2
                    predictions += unit_sign * category_representations[column, unit] ** 2
                    representations[:, unit] -= category_representations[column, unit]

                    total_grads = numpy.bincount(column, weights=(2 * unit_sign) * representations[:, unit] * grads, minlength=minlen)
                    total_hesss = numpy.bincount(column, weights=4 * representations[:, unit] ** 2 * hesss, minlength=minlen)
                    nominator = total_grads - self.regularization * category_representations[:, unit]
                    denominator = total_hesss + self.regularization

                    # TODO iterative update here with penalty for is_seq
                    unit_update = self.learning_rate * nominator / denominator
                    category_representations[:, unit] += unit_update
                    category_representations[:, unit] = numpy.clip(category_representations[:, unit], -1, 1)

                    representations[:, unit] += category_representations[column, unit]
                    predictions += unit_sign * representations[:, unit] ** 2
                    predictions -= unit_sign * category_representations[column, unit] ** 2
                    
                self.losses.append(self.loss(predictions))
            print(iteration, self.losses[-1])
        return self
                
    def compute_prediction(self, biases, representations, representations_sq, unit_signs):
        return biases + (representations ** 2).dot(unit_signs) - representations_sq.dot(unit_signs)
        
    def compute_representations(self, X_cat):
        biases = numpy.zeros(len(X_cat), dtype='float')
        representations = numpy.zeros([len(X_cat), self.n_units], dtype='float')
        representations_sq = numpy.zeros([len(X_cat), self.n_units], dtype='float')
        for cat_biases, cat_representations, column, connection in \
                zip(self.cat_biases, self.cat_representations, X_cat.T, self.connections):
            biases += cat_biases[column]
            representations += cat_representations[column] * connection[None, :]
            representations_sq += (cat_representations ** 2)[column] * connection[None, :]
        return biases, representations, representations_sq
        
    def decision_function(self, X):
        X_cat = self.decompose_data(X, fit=False)
        biases, representations, representations_sq = self.compute_representations(X_cat)
        return self.compute_prediction(biases, representations, representations_sq, self.unit_signs)
        
    def predict_proba(self, X):
        result = numpy.zeros([len(X), 2])
        result[:, 1] = scipy.special.expit(self.decision_function(X))
        result[:, 0] = 1 - result[:, 1]
        return result
        
    def predict(self, X):
        return numpy.argmax(self.predict_proba(X), axis=1)
    
    
from hep_ml.losses import LogLossFunction


class Logistic_My:
    def __init__(self, regularization, n_iterations=10):
        self.regularization = regularization
        self.n_iterations = n_iterations
        
    def fit(self, X, y):
        X = numpy.array(X)
        self.loss = LogLossFunction()
        self.loss.fit(X, y, sample_weight=y*0 + 1)
        max_cats = numpy.max(X) + 1
        self.cat_biases = numpy.zeros([max_cats, X.shape[1]], dtype='float')
        predictions = numpy.zeros(len(X))
        for stage in range(self.n_iterations):
            for column in range(X.shape[1]):
                grads = self.loss.negative_gradient(predictions)
                hesss = self.loss.hessian(predictions)
                inds = X[:, column]
                nominator = numpy.bincount(inds, weights=grads, minlength=max_cats) - self.regularization * self.cat_biases[:, column]
                denominator = numpy.bincount(inds, weights=hesss, minlength=max_cats) + self.regularization
                predictions -= self.cat_biases[inds, column]
                self.cat_biases[:, column] += nominator / denominator
                predictions += self.cat_biases[inds, column]
            print stage, self.loss(predictions)
        return self
    
    def predict_proba(self, X):
        X = numpy.array(X)
        predictions = numpy.zeros(len(X))
        for column in range(X.shape[1]):
            predictions += self.cat_biases[X[:, column], column]
        return predictions
    
    def predict_train(self, X):
        X = numpy.array(X)
        predictions = self.predict_proba(X)
        
        grads = self.loss.negative_gradient(predictions)
        hesss = self.loss.hessian(predictions)
        prediction_shift = numpy.zeros(len(X))
        for column in range(X.shape[1]):
            inds = X[:, column]
            cum_grads = numpy.bincount(inds, weights=grads)[inds] 
            cum_hess = numpy.bincount(inds, weights=hesss)[inds] + self.regularization
            prediction_shift += - grads / cum_hess 
        
        return predictions + prediction_shift
    
 

from hep_ml.losses import LogLossFunction
from scipy.special import expit

class CategoricalNN:
    def __init__(self, n_units=10, regularization=100., n_iterations=10):
        self.n_units = n_units
        self.n_iterations = n_iterations
        self.regularization = regularization
        
        
    def fit(self, X_cat, y):
        X_cat = numpy.array(X_cat)
        self.unit_weights = numpy.random.normal(size=self.n_units)
        self.cat_weights = []
        for column in X_cat.T:
            self.cat_weights.append(numpy.random.normal(size=[numpy.max(column) + 1, self.n_units]) * 0.1)
        loss = LogLossFunction()
        loss.fit(X_cat, y, y * 0 + 1.)
        
        unit_predictions, predictions = self.compute_all(X_cat)
        
        # Training process
        for iteration in range(self.n_iterations):
            new_unit_predictions, new_predictions = self.compute_all(X_cat)
            assert numpy.allclose(predictions, new_predictions)
            predictions = new_predictions
            assert numpy.allclose(unit_predictions, new_unit_predictions)
            unit_predictions = new_unit_predictions

            for unit in range(self.n_units):
                # updating coefficient for unit
                
                for updated_unit in [unit]:
                    grads = loss.negative_gradient(predictions)
                    hesss = loss.hessian(predictions)
                    unit_outputs = self.activation(unit_predictions[:, updated_unit])
                    nom = numpy.dot(grads, unit_outputs)
                    denom = (numpy.dot(hesss, unit_outputs ** 2) + self.regularization)
                    step = 0.5 * nom / denom
                    self.unit_weights[updated_unit] += step
                    predictions += step * unit_outputs

                for column in range(X_cat.shape[1]):
                    inds = X_cat[:, column]
                    # updating with respect to column and unit
                    unit_outputs, unit_derivs, unit_hesss = self.act_grad_hess(unit_predictions[:, unit])
                    
                    unit_weight = self.unit_weights[unit]
                    grads = loss.negative_gradient(predictions) * unit_weight
                    hesss = loss.hessian(predictions) * unit_weight ** 2
                    
                    cat_grads = grads * unit_derivs
                    cat_hesss = hesss * (unit_derivs ** 2) + grads * unit_hesss
                    
                    max_cats = self.cat_weights[column].shape[0]
                    
                    nominator = numpy.bincount(inds, weights=cat_grads, minlength=max_cats)
                    nominator -= self.regularization * self.cat_weights[column][:, unit]

                    cat_steps =  nominator/ \
                        (numpy.bincount(inds, weights=cat_hesss.clip(0), minlength=max_cats) + self.regularization)
                    cat_steps *= 1.5
                        
                    self.cat_weights[column][:, unit] += cat_steps
                    predictions -= self.unit_weights[unit] * unit_outputs
                    unit_predictions[:, unit] += cat_steps[inds]
                    unit_outputs = self.activation(unit_predictions[:, unit])
                    predictions += self.unit_weights[unit] * unit_outputs
                
                    print iteration, unit, column, loss(predictions)
                   
        return self
    
#     def activation(self, unit_input):
#         return numpy.tanh(unit_input)
    
#     def act_grad_hess(self, unit_input):
#         unit_outputs = numpy.tanh(unit_input)
#         unit_derivs =  (1 - unit_outputs ** 2)
#         unit_hesss =  - 2 * unit_outputs * unit_derivs        
#         return unit_outputs, unit_derivs, unit_hesss
    
#     def activation(self, unit_input):
#         return unit_input ** 2
    
#     def act_grad_hess(self, unit_input):
#         unit_outputs = unit_input ** 2
#         unit_derivs =  2 * unit_input 
#         unit_hesss =   2. + 0 * unit_input
#         return unit_outputs, unit_derivs, unit_hesss
    
    def activation(self, unit_input):
        return numpy.logaddexp(0, unit_input)
    
    def act_grad_hess(self, unit_input):
        unit_outputs = numpy.logaddexp(0, unit_input)
        unit_derivs =  expit(unit_input)
        unit_hesss =   unit_derivs * (1. - unit_derivs)
        return unit_outputs, unit_derivs, unit_hesss
    
    def compute_all(self, X_cat):
        X_cat = numpy.array(X_cat)
        unit_predictions = numpy.zeros([len(X_cat), self.n_units])
        for column, column_weights in enumerate(self.cat_weights):
            for unit in range(self.n_units):
                unit_predictions[:, unit] += column_weights[X_cat[:, column], unit]
        predictions = self.activation(unit_predictions).dot(self.unit_weights)
        return unit_predictions, predictions
    
    def decision_function(self, X_cat):
        unit_predictions, predictions = self.compute_all(X_cat)
        return predictions
            
        
        
    
