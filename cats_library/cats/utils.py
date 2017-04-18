"""
Tools to deal with categories.
"""
from __future__ import division, print_function, absolute_import

import numpy

from collections import OrderedDict
from sklearn.base import TransformerMixin, BaseEstimator

__author__ = 'Alex Rogozhnikov'


class CategoryMapper:
    def fit(self, categories):
        self.lookup = numpy.unique(categories)
        return self

    def transform(self, categories):
        """
        Converts categories to numbers, 0 is reserved for new values (not present in fitted data)
        """
        return (numpy.searchsorted(self.lookup, categories) + 1) * numpy.in1d(categories, self.lookup)


class EncodingTransformer(BaseEstimator, TransformerMixin):
    """
    Each of features is treated as categorical and replaced with small integers.
    """

    def __init__(self, zero_for_new_values=True):
        self.zero_for_new_values = zero_for_new_values

    def fit(self, X, y=None):
        X = numpy.require(X)
        self.features = numpy.arange(X.shape[1])
        self.mappers = []
        for feature in self.features:
            self.mappers.append(CategoryMapper().fit(X[:, feature]))
        return self

    def transform(self, X):
        assert X.shape[1] == len(self.features), 'wrong number of features'
        X = numpy.require(X)
        result = numpy.empty([len(X), len(self.features)], dtype='int32', order='F')
        for feature, mapper in zip(self.features, self.mappers):
            result[:, feature] = mapper.transform(X[:, feature])


def compute_combination_column(train, test, features, new_name=None, unfrequent_threshold=0):
    """
    Adds a new column - category over others. Modifies original dataframes.
    :param features: names of features to be combined into new feature.
    :param train: pandas.DataFrame
    :param test: pandas.DataFrame
    :param unfrequent_threshold: unfrequent categories (having no more than threshold)
    will be first mapped to a single category, then product is computed.
    """
    feature_train = numpy.zeros(len(train), dtype='int')
    feature_test = numpy.zeros(len(test), dtype='int')

    if new_name is None:
        new_name = '_'.join(features)

    for feature in features:
        train_f = train[feature]
        test_f = test[feature]
        feature_max = max(numpy.max(train_f), numpy.max(test_f))
        if unfrequent_threshold > 0:
            counts = numpy.bincount(numpy.hstack([train_f, test_f]))
            is_suppressed = (counts <= unfrequent_threshold)
            train_f[is_suppressed[train_f]] = 0
            test_f[is_suppressed[test_f]] = 0

        feature_train *= feature_max
        feature_test *= feature_max
        feature_train += train[feature]
        feature_test += test[feature]

    mapper = CategoryMapper().fit(feature_train)
    feature_train = mapper.transform(feature_train)
    feature_test = mapper.transform(feature_test)

    train[new_name] = feature_train
    test[new_name] = feature_test
    return new_name


class ColumnCombiner(BaseEstimator, TransformerMixin):
    """
    Adds a new column - categories based on different subsets.
    Works better if done using whole sample.
    Warning: modification is inplace!
    """

    def __init__(self, features_combinations, unfrequent_threshold=0):
        """
        :param features_combinations: dict: new_name -> list of columns to be combined
        """
        self.unfrequent_threshold = unfrequent_threshold
        self.features_combinations = features_combinations
        self.feature_descriptions = None

    def fit(self, X, y=None):
        self.feature_descriptions = OrderedDict()
        for new_feature, combined_features in self.features_combinations:
            for feature in combined_features:
                if feature not in self.feature_descriptions.keys():
                    train_f = X[feature]
                    feature_max = numpy.max(train_f) + 1
                    counts = numpy.bincount(train_f)
                    is_suppressed = (counts <= self.unfrequent_threshold)

                    self.feature_descriptions[feature] = feature_max, is_suppressed
        return self

    def transform(self, X, inplace=True):
        if not inplace:
            X = X.copy()
        for new_feature_name, combined_features in self.features_combinations:
            new_feature = numpy.zeros(len(X), dtype='int')

            for combined_feature in combined_features:
                features_max, is_suppressed = self.feature_descriptions[combined_feature]
                column = X[combined_feature]
                column[is_suppressed[column]] = 0

                new_feature *= features_max
                new_feature += column

            X[new_feature_name] = new_feature
        return X


# random stored trash

def preprocess_categories(train, test, category_columns):
    for column in category_columns:
        mapper = CategoryMapper().fit(train[column].copy())
        train[column] = mapper.transform(train[column])
        test[column] = mapper.transform(test[column])


def compute_combination_column(train, test, features, new_name=None):
    feature_train = numpy.zeros(len(train), dtype='int')
    feature_test = numpy.zeros(len(test), dtype='int')

    if new_name is None:
        new_name = '_'.join(features)

    for feature in features:
        feature_max = max(numpy.max(feature_train), numpy.max(feature_test))
        feature_train *= feature_max
        feature_test *= feature_max
        feature_train += train[feature]
        feature_test += test[feature]

    train[new_name] = feature_train
    test[new_name] = feature_test
    preprocess_categories(train, test, [new_name])
    return new_name


def aggregate_over_categories(train, test, trainY, features, add_mean=False, add_sum=False, add_frequencies=False,
                              threshold_for_statistics=2):
    mean = numpy.mean(trainY)
    for feature in features:
        sums_over_category = numpy.bincount(train[feature], weights=trainY)
        frequencies = numpy.bincount(train[feature])
        means_over_category = sums_over_category / frequencies
        means_over_category[frequencies < threshold_for_statistics] = mean

        if add_mean:
            train[feature + '_mean'] = means_over_category[train[feature]]
            test[feature + '_mean'] = means_over_category[test[feature]]

        if add_sum:
            train[feature + '_sum'] = sums_over_category[train[feature]]
            test[feature + '_sum'] = sums_over_category[test[feature]]

        if add_frequencies:
            train[feature + '_freq'] = frequencies[train[feature]]
            test[feature + '_freq'] = frequencies[test[feature]]


## useful for large datasets
def reduce_categories(data, features, threshold=1):
    for feature in features:
        lookup, new_column, counts = numpy.unique(data[feature], return_inverse=True, return_counts=True)
        nulled = counts <= threshold
        mapping = numpy.cumsum(1 - nulled).astype('int32')
        mapping[nulled] = 0
        data[feature] = mapping[new_column]


def add_combination_column(data, features, new_name=None):
    if new_name is None:
        new_name = '_'.join(features)

    new_column_values = data[features[0]]
    for feature in features[1:]:
        feature_max = numpy.max(data[feature])
        new_column_values = new_column_values * feature_max + data[feature]

    data[new_name] = new_column_values
    reduce_categories(data, [new_name])
