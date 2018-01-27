# !/usr/bin/env python

import numpy as np

from helpers import load_data
from utils import init_mf, compute_error


def update_user_feature(train, num_user, item_features, lambda_user, nz_items_indices, robust=False):
    """
    Update user feature matrix
    :param train: training data, sparse matrix of shape (num_item, num_user)
    :param num_user: number of users
    :param item_features: factorized item features, dense matrix of shape (num_feature, num_item)
    :param lambda_user: ridge regularization parameter
    :param nz_items_indices: list of arrays, contains the non-zero indices of each column (user) in train
    :param robust: True to enable robustsness against singular matrices
    :return: user_features : updated factorized user features, dense matrix of shape (num_feature, num_user)
    """
    num_features = item_features.shape[0]
    user_features = np.zeros((num_features, num_user))

    for user in range(num_user):
        y = train[nz_items_indices[user], user].todense()  # non-zero elements of col n° user of train
        x = item_features[:, nz_items_indices[user]]       # corresponding columns of item_features
        nnz = nz_items_indices[user].shape[0]

        # Solution to ridge problem min(|x.T @ w - y|^2 + lambda * |w|^2)
        wy = x.dot(y)
        if not robust:
            w = np.linalg.solve(x.dot(x.T) + lambda_user * nnz * np.identity(num_features), wy)
        else:
            w = np.linalg.lstsq(x.dot(x.T) + lambda_user * nnz * np.identity(num_features), wy)[0]
        user_features[:, user] = w.ravel()

    return user_features


def update_item_feature(train, num_item, user_features, lambda_item, nz_users_indices, robust=False):
    """
    Update item feature matrix
    :param train: training data, sparse matrix of shape (num_item, num_user)
    :param num_item: number of items
    :param user_features: factorized user features, dense matrix of shape (num_feature, num_user)
    :param lambda_item: ridge regularization parameter
    :param nz_users_indices: list of arrays, contains the non-zero indices of each row (item) in train
    :param robust: True to enable robustsness against singular matrices
    :return: item_features: updated factorized item features, dense matrix of shape (num_feature, num_item)
    """
    num_features = user_features.shape[0]
    item_features = np.zeros((num_features, num_item))

    for item in range(num_item):
        y = train[item, nz_users_indices[item]].todense().T  # non-zero elements of line n° item of train
        x = user_features[:, nz_users_indices[item]]   # corresponding columns of user_features
        nnz = nz_users_indices[item].shape[0]

        # Solution to ridge problem min(|X.T @ w - y|^2 + lambda * |w|^2)
        wy = x.dot(y)
        if not robust:
            w = np.linalg.solve(x.dot(x.T) + lambda_item * nnz * np.identity(num_features), wy)
        else:
            w = np.linalg.lstsq(x.dot(x.T) + lambda_item * nnz * np.identity(num_features), wy)[0]
        item_features[:, item] = w.ravel()

    return item_features


def matrix_factorization_als(train, test, num_features=20, lambda_user=0.014, lambda_item=0.575,
                             stop_criterion=1e-5, max_iter=100, min_iter=25, verbose=False, robust=False):
    """
    Computes the matrix factorization on train data using alternating least squares
    and returns RMSE computed on train and test data.
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: testing data, sparse matrix of shape (num_items, num_users)
                 if None, only the training part is done and the matrix factorization is returned
    :param num_features: number of latent features in the factorization
    :param lambda_user: size of penalization coefficient for the optimization of user features
    :param lambda_item: size of penalization coefficient for the optimization of item features
    :param stop_criterion: computation continues until an optimization step does not improve the error by more than
                           this value
    :param max_iter: maximum number of iterations
    :param min_iter: minimum number of iterations
    :param verbose: True if user wants to print details of the computation
    :param robust: True to enable robustsness against singular matrices
    :return: training RMSE, testing RMSE, user_features and item_features
             or user_features and item_features only if test is None
    """
    change = 1
    error_list = [0, 0]

    num_item = train.shape[0]
    num_user = train.shape[1]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_mf(train, num_features)

    nz_items, nz_users = train.nonzero()
    nz_users_indices = [nz_users[nz_items == i] for i in range(num_item)]
    nz_items_indices = [nz_items[nz_users == u] for u in range(num_user)]
    nz_train = list(zip(nz_items, nz_users))

    # run ALS
    if verbose:
        print('Learning the matrix factorization using ALS...')
    n_iter = 0
    while change > stop_criterion and n_iter < max_iter or n_iter < min_iter:
        n_iter += 1

        user_features = update_user_feature(train, num_user, item_features, lambda_user, nz_items_indices, robust)
        item_features = update_item_feature(train, num_item, user_features, lambda_item, nz_users_indices, robust)

        error = compute_error(train, user_features, item_features, nz_train)
        if False:
            print("Iteration {} : RMSE on training set: {}.".format(n_iter, error))
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])

    if test is not None:
        # evaluate the test error
        nnz_row, nnz_col = test.nonzero()
        nnz_test = list(zip(nnz_row, nnz_col))
        test_rmse = compute_error(test, user_features, item_features, nnz_test)
        if verbose:
            print('Final RMSE on train data: {}'.format(error_list[-1]))
            print('Final RMSE on test data: {}.'.format(test_rmse))
        return error_list[-1], test_rmse, user_features, item_features

    if test is None:
        return error_list[-1], user_features, item_features


def write_als_prediction(user_features, item_features, input_path='../data/sample_submission.csv',
                         output_path='../data/predictions/als_prediction.csv', verbose=False):
    """
    Writes a prediction based on matrix factorization given in argument
    :param user_features: sparse matrix of shape (num_features, num_users)
    :param item_features: sparse matrix of shape (num_features, num_items)
    :param input_path: path to the sample submission provided by Kaggle
    :param output_path: path to output the submission
    :param verbose: if True, details about computation are printed
    """
    test = load_data(input_path, verbose=verbose)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            item_info = item_features[:, row]
            user_info = user_features[:, col]
            prediction = user_info.T.dot(item_info)
            prediction = min(5, prediction)
            prediction = max(1, prediction)
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, prediction))


def matrix_factorization_sgd(train, test, gamma=0.01, num_features=25,
                             lambda_user=0.011, lambda_item=0.25, num_epochs=50, verbose=False):
    """
    Computes the matrix factorization on train data using stochastic gradient descent
    and returns RMSE computed on train and test data.
    :param train: training data, sparse matrix of shape (num_items, num_users)
    :param test: testing data, sparse matrix of shape (num_items, num_users)
                 if None, only the training is performed and the factorization is returned
    :param gamma: size of each step of the gradient descent
    :param num_features: number of latent features in the factorization
    :param lambda_user: size of penalization coefficient for the optimization of user features
    :param lambda_item: size of penalization coefficient for the optimization of item features
    :param num_epochs: number of epochs for the optimization
    :param verbose: True if user wants to print details of the computation
    :return: training RMSE, testing RMSE, user_features and item_features
             or user_features and item_features only if test is None
    """
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_mf(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    if verbose:
        print('Learning the matrix factorization using SGD...')
    for epoch in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.1

        for d, n in nz_train:
            current_item = item_features[:, d]
            current_user = user_features[:, n]
            error = train[d, n] - current_user.T.dot(current_item)

            # gradient
            item_features[:, d] += gamma * (error * current_user - lambda_item * current_item)
            user_features[:, n] += gamma * (error * current_item - lambda_user * current_user)

        rmse = compute_error(train, user_features, item_features, nz_train)
        if False:
            print("epoch: {}, RMSE on training set: {}.".format(epoch + 1, rmse))

        errors.append(rmse)

    if test is not None:
        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        rmse = compute_error(test, user_features, item_features, nz_test)
        if verbose:
            print('Final RMSE on train data: {}'.format(errors[-1]))
            print('Final RMSE on test data: {}.'.format(rmse))
        return errors[-1], rmse, user_features, item_features

    if test is None:
        return errors[-1], user_features, item_features


def write_sgd_prediction(user_features, item_features, input_path='../data/sample_submission.csv',
                         output_path='../data/predictions/sgd_prediction.csv', verbose=False):
    """
    Writes a prediction based on matrix factorization given in argument
    :param user_features: sparse matrix of shape (num_features, num_users)
    :param item_features: sparse matrix of shape (num_features, num_items)
    :param input_path: path to the sample submission provided by Kaggle
    :param output_path: path to output the submission
    :param verbose: if True, details about computation are printed
    """
    test = load_data(input_path, verbose=verbose)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            item_info = item_features[:, row]
            user_info = user_features[:, col]
            prediction = user_info.T.dot(item_info)
            prediction = min(5, prediction)
            prediction = max(1, prediction)
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, prediction))
