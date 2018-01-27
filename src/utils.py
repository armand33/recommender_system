# !/usr/bin/env python

import numpy as np
import scipy.sparse as sp


def split_data(ratings, min_num_ratings, p_test=0.1, verbose=False, seed=988):
    """
    Splits the data set (ratings) to training data and test data
    :param ratings: initial data set (sparse matrix of dimensions n items and p users)
    :param min_num_ratings: all users and items must have at least min_num_ratings per user and per item to be kept
    :param p_test: proportion of the data dedicated to test
    :param verbose: True if user wants to print details of computation
    :param seed: random seed
    :return:    - valid_ratings (initial data set where some items and users where dropped)
                - train train data (same shape as valid_ratings but with 1-p_test non_zero values)
                - test data (same shape as valid_ratings but with p_test non zero values
    """

    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

    # set seed
    np.random.seed(seed)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    # define the sparse matrix that will contain train and test data
    train = sp.lil_matrix(valid_ratings.shape)
    test = sp.lil_matrix(valid_ratings.shape)

    # get the index of non zero elements of the valid_ratings
    non_zero_item, non_zero_users = valid_ratings.nonzero()

    # for each item, select p_test percent of users to put in test and put the rest in train
    for item in set(non_zero_item):

        _, indexes = valid_ratings[item].nonzero()
        test_ind = np.random.choice(indexes, size=int(len(indexes) * p_test))
        train_ind = list(set(indexes) - set(test_ind))

        train[item, train_ind] = valid_ratings[item, train_ind]
        test[item, test_ind] = valid_ratings[item, test_ind]

    if verbose:
        print('Shape of original ratings : {}'.format(ratings.shape))
        print('Shape of valid ratings (and of train and test data) : {}'.format(valid_ratings.shape))
        print("Total number of nonzero elements in original data : {v}".format(v=ratings.nnz))
        print("Total number of nonzero elements in train data : {v}".format(v=train.nnz))
        print("Total number of nonzero elements in test data : {v}".format(v=test.nnz))
    return valid_ratings, train, test


def compute_error(data, user_features, item_features, nz):
    """
    Returns the error of the prediction using matrix factorization data = transpose(user_features) x item_features.
    The error is only computed on non zero elements.
    :param data: sparse matrix of shape (num_items, num_users)
    :param user_features: matrix of shape (num_features, num_users)
    :param item_features: matrix of shape (num_features, num_items)
    :param nz: list of non zero entries of matrix data
    :return: the RMSE corresponding to the approximation of data by transpose(user_features) x item_features
    """
    mse = 0
    for row, col in nz:
        current_item = item_features[:, row]
        current_user = user_features[:, col]
        prediction = current_user.T.dot(current_item)
        prediction = min(5, prediction)
        prediction = max(1, prediction)
        mse += (data[row, col] - prediction) ** 2
    return np.sqrt(1.0 * mse / len(nz))


def init_mf(train, num_features):
    """
    Initialize the empty matrices for matrix factorization.
    As indicated in lab 10, the item_features matrix is initialized by assigning the average rating for that movie
    as the first row, and small random numbers for the remaining entries.
    :param train: training data set (sparse matrix of size nxp, n items and p users)
    :param num_features: number of latent features wanted in the factorization
    :return: two randomly initialized matrices of shapes (num_features, num_users) and (num_features, num_items)
    """
    num_items, num_users = train.shape

    user_features = np.random.rand(num_features, num_users) / num_users
    user_features[0, :] = np.ones((num_users,))

    item_features = np.random.rand(num_features, num_items) / num_items
    item_features[0, :] = sp.csr_matrix.mean(train, axis=1).reshape(num_items, )

    return user_features, item_features


def build_prediction_factorization(item_features, user_features, test):
    """
    Build the prediction using matrix factorization
    :param item_features:
    :param user_features:
    :param test:
    :return:
    """
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    for row, col in nnz_test:
        current_item = item_features[:, row]
        current_user = user_features[:, col]
        prediction = current_user.T.dot(current_item)
        prediction = min(5, prediction)
        prediction = max(1, prediction)
        test[row, col] = prediction

    return test


def write_matrix_to_file(matrix, path):
    with open(path, 'w') as output:
        output.write('Id,Prediction\n')
        nnz_row, nnz_col = matrix.nonzero()
        nnz = list(zip(nnz_row, nnz_col))
        for row, col in nnz:
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, matrix[row, col]))


def get_indices(matrix):
    """
    :param matrix: sparse matrix of any shape
    :return: list of (i, j) such as matrix[i, j] is non zero
    """
    nnz_row, nnz_col = matrix.nonzero()
    nnz = list(zip(nnz_row, nnz_col))
    return nnz
