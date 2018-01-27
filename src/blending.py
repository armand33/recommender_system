import numpy as np
import os
import pickle
import scipy.sparse as sp

from matrix_factorization import matrix_factorization_als, write_als_prediction


def build_matrix(nnz, methods, ratings, g_mean, user_means, item_means, sgd, als, cf_item, cf_user, predict=False):
    """
    Build the matrix for blending (shape = (number of predictions, number of models))
    :param nnz: list of tuples item, user where each tuple is a prediction
    :param methods: list of the models to put in the matrix (should be in 'global_mean', 'user_mean', 'item_mean'
                    'sgd', 'als', 'cf_item', 'cf_user').
    :param ratings: initial sparse matrix of shape (number of items, number of users)
    :param g_mean: global mean of the training set
    :param user_means: user means (from training set)
    :param item_means: item means (from training set)
    :param sgd: predictions coming from the sgd matrix factorization
    :param als: predictions coming from the als matrix factorization
    :param cf_item: predictions coming from the item-based collaborative filtering
    :param cf_user: predictions coming from the user-based collaborative filtering
    :param predict: Boolean, to know if currently training/testing or predicting
    :return: matrix for blending (shape = (number of predictions, number of models))
    """
    m = np.zeros(shape=(len(nnz), len(methods)))

    if not predict:
        y = []
    else:
        y = None

    for i, (item, user) in enumerate(nnz):
        if not predict:
            y.append(ratings[item, user])
        if 'global_mean' in methods:
            k = methods.index('global_mean')
            m[i, k] = g_mean
        if 'user_mean' in methods:
            k = methods.index('user_mean')
            m[i, k] = user_means[user]
        if 'item_mean' in methods:
            k = methods.index('item_mean')
            m[i, k] = item_means[item]
        if 'sgd' in methods:
            k = methods.index('sgd')
            m[i, k] = sgd[item, user]
        if 'als' in methods:
            k = methods.index('als')
            m[i, k] = als[item, user]
        if 'cf_item' in methods:
            k = methods.index('cf_item')
            m[i, k] = cf_item[item, user]
        if 'cf_user' in methods:
            k = methods.index('cf_user')
            m[i, k] = cf_user[item, user]

    return m, y


def blend(methods, alpha, nnz_train, nnz_test, train, test,
          g_mean, user_means, item_means, sgd, als, cf_item, cf_user, predict=False):
    """

    :param methods: list of methods to blend
    :param alpha: penalization parameter for the ridge regression
    :param nnz_train: exhaustive list of (i, j) such that train[i, j] in non zero
    :param nnz_test: exhaustive list of (i, j) such that test[i, j] in non zero
    :param train: training set (sparse matrix of shape (number of items, number of users))
    :param test: test set (sparse matrix of shape (number of items, number of users))
    :param g_mean: global mean of the training set
    :param user_means: user_means computed on the training set
    :param item_means: item_means computed on the training set
    :param sgd: sgd prediction on the training set (sparse matrix of shape (number of items, number of users))
    :param als: als predictions on the training set (sparse matrix of shape (number of items, number of users))
    :param cf_item: predictions on the training set (sparse matrix of shape (number of items, number of users))
    :param cf_user: predictions on the training set (sparse matrix of shape (number of items, number of users))
    :param predict: Boolean, if True, no ground truth vector y is return
    :return: matrix of shape (number of predictions, number of models), vector of length number of predictions
    """
    m_train, y_train = build_matrix(nnz_train, methods, train, g_mean,
                                    user_means, item_means, sgd, als, cf_item, cf_user, predict=False)

    # Ridge Regression
    w = np.linalg.solve(m_train.T.dot(m_train) + alpha * np.eye(m_train.shape[1]), m_train.T.dot(y_train))
    y_predict_train = m_train.dot(w)

    # Cut predictions that are too high and too low
    for i in range(len(y_predict_train)):
        y_predict_train[i] = min(5, y_predict_train[i])
        y_predict_train[i] = max(1, y_predict_train[i])

    m_test, y_test = build_matrix(nnz_test, methods, test, g_mean,
                                  user_means, item_means, sgd, als, cf_item, cf_user, predict)
    y_predict_test = m_test.dot(w)

    # Cut predictions that are too high and too low
    for i in range(len(y_predict_test)):
        y_predict_test[i] = min(5, y_predict_test[i])
        y_predict_test[i] = max(1, y_predict_test[i])

    if y_test is not None:
        return w, np.sqrt(np.mean((y_train - y_predict_train) ** 2)), np.sqrt(np.mean((y_test - y_predict_test) ** 2))
    else:
        with open('../data/predictions/merge_prediction.csv', 'w') as output:
            output.write('Id,Prediction\n')
            for i, (row, col) in enumerate(nnz_test):
                output.write('r{}_c{},{}\n'.format(row + 1, col + 1, y_predict_test[i]))


def train_als(ratings):
    """
    Check if the training files are present, if not, the ALS model is trained and the files are written to disk.
    :param ratings: initial training matrix (sparse matrix of shape (number of items, number of users)
    :return: _
    """
    if not os.path.isfile('../data/pickle/data_train_als.pickle') or \
            not os.path.isfile('../data/predictions/als_prediction.csv'):
        print('Files are not present, training the model now.')
        # In this case we train the ALS model
        _, user_features, item_features = matrix_factorization_als(ratings, None, verbose=True, stop_criterion=0.00001,
                                                                   lambda_user=0.014, lambda_item=0.575,
                                                                   num_features=20)
        write_als_prediction(user_features, item_features)

        nnz_row, nnz_col = ratings.nonzero()
        nnz_ratings = list(zip(nnz_row, nnz_col))
        als_train = np.zeros(shape=ratings.shape)
        for i, (item, user) in enumerate(nnz_ratings):
            als_train[item, user] = min(5, max(1, user_features[:, user].T.dot(item_features[:, item])))

        with open(b'../data/pickle/data_train_als.pickle', 'wb') as f:
            pickle.dump(sp.lil_matrix(als_train), f)
    else:
        print('Files are present, no need to train the model.')
