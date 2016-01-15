import numpy
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_problem_flight(large=False, convert_to_ints=False):
    '''
    Dataset used in common ML benchmarks: https://github.com/szilard/benchm-ml
    links to files:
    https://s3.amazonaws.com/benchm-ml--main/test.csv
    https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv
    https://s3.amazonaws.com/benchm-ml--main/train-1m.csv
    https://s3.amazonaws.com/benchm-ml--main/train-10m.csv
    '''
    if large:
        trainX = pandas.read_csv('../data/flight_train-10m.csv')
    else:
        trainX = pandas.read_csv('../data/flight_train-1m.csv')
    testX  = pandas.read_csv('../data/flight_test.csv')
    
    trainY = (trainX.dep_delayed_15min.values == 'Y') * 1
    testY  = (testX.dep_delayed_15min.values == 'Y') * 1
    
    trainX = trainX.drop('dep_delayed_15min', axis=1)
    testX  = testX.drop('dep_delayed_15min', axis=1)
    if convert_to_ints:
        categoricals = ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest',]
        continous = ['DepTime', 'Distance']
        
        trainX, testX = process_categorical_features(trainX, testX, columns=categoricals)
        trainX, testX = process_continuous_features(trainX, testX, columns=continous)
    
    return trainX, testX, trainY, testY


def load_problem_movielens_100k(all_features=False):
    '''Standard test dataset for recommendation systems
    From http://grouplens.org/datasets/movielens/
    '''
    folder = '../data/ml-100k'
    ratings = pandas.read_csv(folder + '/u.data', sep='\t', 
                              names=['user', 'movie', 'rating', 'timestamp'], header=None)
    ratings = ratings.drop('timestamp', axis=1)
    if all_features:
        users   = pandas.read_csv(folder + '/u.user', sep='|', 
                                  names=['user', 'age', 'gender', 'occupation', 'zip'], header=None)
        movies  = pandas.read_csv(folder + '/u.item', sep='|',
           names=['movie', 'title','released','video_release', 'IMDb URL','unknown','Action','Adventure','Animation',
            'Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir',
            'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'], header=None)
        
        movies = movies.drop(['title', 'IMDb URL', 'video_release'], axis=1)
        movies['released'] = pandas.to_datetime(movies['released']).map(lambda z: z.year)
        ratings = pandas.merge(pandas.merge(ratings, users, on='user'), movies, on='movie')

    answers = ratings['rating'].values
    ratings = ratings.drop('rating', axis=1)

    for feature in ratings.columns:
        _, ratings[feature] = numpy.unique(ratings[feature], return_inverse=True)
        
    trainX, testX, trainY, testY = train_test_split(ratings, answers, train_size=0.75, random_state=42)
    return trainX, testX, trainY, testY


def load_problem_movielens_1m(all_features=False):
    '''
    Standard test dataset for recommendation systems
    From http://grouplens.org/datasets/movielens/
    '''
    folder = '../data/ml-1m'
    ratings = pandas.read_csv(folder + '/ratings.dat', sep='::', 
                              names=['user', 'movie', 'rating', 'timestamp'], header=None)
    ratings = ratings.drop('timestamp', axis=1)
    
    if all_features:
        users = pandas.read_csv(folder + '/users.dat', sep='::', 
                                names=['user', 'gender', 'age', 'occupation', 'zip'], header=None)
        movies = pandas.read_csv(folder + '/movies.dat', sep='::', names=['movie', 'title', 'genres'], header=None)
        sparse_genres = CountVectorizer().fit_transform(movies.genres.map(lambda x: x.replace('|', ' ')))
        sparse_genres = pandas.DataFrame(sparse_genres.todense())
        movies = pandas.concat([movies[['movie']], sparse_genres], axis=1)    
        ratings = pandas.merge(pandas.merge(ratings, users, on='user'), movies, on='movie')

    answers = ratings['rating'].values
    ratings = ratings.drop('rating', axis=1)

    for feature in ratings.columns:
        _, ratings[feature] = numpy.unique(ratings[feature], return_inverse=True)
        
    trainX, testX, trainY, testY = train_test_split(ratings, answers, train_size=0.75, random_state=42)
    return trainX, testX, trainY, testY


def preprocess_ad_problem():
    """
    Kaggle competition on CTR prediction: https://www.kaggle.com/c/avazu-ctr-prediction
    """
    av_train = pandas.read_csv('../data/ad_train.csv')
    for column in av_train.columns:
        if column != 'hour':        
            av_train[column] = numpy.unique(av_train[column], return_inverse=True)[1].astype('uint16')
            
    for column in av_train.columns:
        if numpy.max(av_train[column]) < 250:
            av_train[column] = av_train[column].astype('uint8')
        elif numpy.max(av_train[column]) < 65000:
            av_train[column] = av_train[column].astype('uint16')
        else:
            av_train[column] = av_train[column].astype('uint32')            
            
    av_train.to_hdf('../data/ad_updated_train.hdf5', 'data')
            
def load_problem_ad():
    """
    Kaggle competition on CTR prediction: https://www.kaggle.com/c/avazu-ctr-prediction
    First use preprocess ad.
    """
    data = pandas.read_hdf('../data/ad_updated_train.hdf5', 'data')
    data['day'] = (data['hour'] // 100) % 100
    data['hour'] = data['hour'] % 100
    answers = data['click'].values
    data = data.drop('click', axis=1)
    trainX, testX, trainY, testY = train_test_split(data, answers, train_size=0.75, random_state=42)
    return trainX, testX, trainY, testY

def remap(column, lookup):
    return (numpy.searchsorted(lookup, column) + 1) * numpy.in1d(column, lookup)

def process_categorical_features(trainX, testX, columns, copy=True):
    if copy:
        trainX = trainX.copy()
        testX = testX.copy()
    
    for column in columns:
        lookup = numpy.unique(trainX[column])
        trainX[column] = remap(trainX[column], lookup)
        testX[column] = remap(testX[column], lookup)
    
    return trainX, testX
        
def process_continuous_features(trainX, testX, columns, copy=True):
    if copy:
        trainX = trainX.copy()
        testX = testX.copy()
    
    for column in columns:
        percentiles = numpy.percentile(trainX[column], [10, 20, 30, 40, 50, 60, 70, 80, 90])
        trainX[column] = numpy.searchsorted(percentiles, trainX[column])
        testX[column]  = numpy.searchsorted(percentiles, testX[column])
    
    return trainX, testX