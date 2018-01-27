# !/usr/bin/env python
import numpy as np

from utils import split_data
from helpers import calculate_mse, load_data


def global_mean(ratings):
    """
    :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
    :return: the global mean of the non zero ratings of the matrix
    """
    # compute mean over non zero ratings of the train
    mean = ratings[ratings.nonzero()].mean()
    return mean


def compute_rmse_global_mean(train, test):
    """
    Compute the RMSE obtained by using global mean of non zero value of train to predict values of test
    :param train: training data set (sparse matrix of size nxp, n items and p users)
    :param test: testing data set (sparse matrix of size nxp, n items and p users)
    :return: RMSE value of the prediction
    """
    mean = global_mean(train)
    mse = calculate_mse(test[test.nonzero()].todense(), mean)
    rmse = np.sqrt(float(mse) / test.nnz)
    return rmse


def global_mean_test(ratings, min_num_ratings, verbose=False):
    """
    Splits the data set in train and test and compute the RMSE using the global mean as prediction.
    :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
    :param min_num_ratings: all users and items must have at least min_num_ratings per user and per item to be kept
    :param verbose: True if user wants details to be printed
    :return: RMSE value of the prediction using global mean as a prediction
    """
    # split the data
    _, train, test = split_data(ratings, min_num_ratings, verbose=verbose)
    # compute the RMSE
    error = compute_rmse_global_mean(train, test)

    return float(error)


def write_global_mean_prediction(mean, input_path='../data/sample_submission.csv',
                                 output_path='../data/global_mean_prediction.csv', verbose=False):
    """
    Write a prediction file for Kaggle submission using global mean.
    :param mean: global mean that is the prediction
    :param input_path: path to the sample submission provided by the Kaggle page (entries we have to predict)
    :param output_path: path to output the prediction file
    :param verbose: if True, details of computation are printed
    """
    mean = min(5, mean)
    mean = max(1, mean)
    test = load_data(input_path, verbose=verbose)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, mean))


def compute_item_means(ratings):
    """
        Computes the mean of ratings for each item (row)
        :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
        :return: List of length n. means[i] is the mean of the ratings of item i.
    """
    means = [0 for _ in range(ratings.shape[0])]
    for item in range(ratings.shape[0]):

        # compute the mean of non-zero rating for current user
        current_ratings = ratings[item]
        current_non_zero_ratings = current_ratings[current_ratings.nonzero()]

        if current_non_zero_ratings.shape[1] != 0:
            mean = current_non_zero_ratings.mean()
            mean = min(5, mean)
            mean = max(1, mean)
            means[item] = mean

    return means


def item_mean_test(ratings, min_num_ratings, verbose=False, p_test=0.1):
    """
    Splits the data set in train and test and compute the RMSE using as prediction the item mean.
    :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
    :param min_num_ratings: all users and items must have at least min_num_ratings per user and per item to be kept
    :param verbose: True if user wants details to be printed
    :param p_test share of the data set to be dedicated to test set
    :return: RMSE value of the prediction using item means as a predictions
b    """
    _, train, test = split_data(ratings, min_num_ratings, verbose=verbose, p_test=p_test)
    cumulated_rmse = 0

    # find the RMSE share due to all users
    for item in range(train.shape[0]):

        # compute the mean of non-zero rating for current user
        current_train_ratings = train[item]
        current_non_zero_train_ratings = current_train_ratings[current_train_ratings.nonzero()]

        if current_non_zero_train_ratings.shape[1] != 0:
            mean = current_non_zero_train_ratings.mean()
            # compute the rmse with all non-zero ratings of current user in test set
            current_test_ratings = test[item]
            current_non_zero_test_ratings = current_test_ratings[current_test_ratings.nonzero()].todense()
            cumulated_rmse += calculate_mse(current_non_zero_test_ratings, mean)

    cumulated_rmse = np.sqrt(float(cumulated_rmse) / test.nnz)

    return cumulated_rmse


def write_item_mean_prediction(means, input_path='../data/sample_submission.csv',
                               output_path='../data/item_mean_prediction.csv', verbose=False):
    """
    Write a prediction file for Kaggle submission using item mean.
    :param means: list of user means
    :param input_path: path to the sample submission provided by the Kaggle page (entries we have to predict)
    :param output_path: path to output the prediction file
    :param verbose: if True, details of computation are printed
    """
    test = load_data(input_path, verbose=verbose)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, means[row]))


def compute_user_means(ratings):
    """
    Computes the mean of ratings for each user (column)
    :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
    :return: List of length p. means[i] is the mean of the ratings of user i.
    """
    means = [0 for _ in range(ratings.shape[1])]
    for user in range(ratings.shape[1]):

        # compute the mean of non-zero rating for current user
        current_ratings = ratings[:, user]
        current_non_zero_ratings = current_ratings[current_ratings.nonzero()]

        if current_non_zero_ratings.shape[0] != 0:
            mean = current_non_zero_ratings.mean()
            mean = min(5, mean)
            mean = max(1, mean)
            means[user] = mean

    return means


def user_mean_test(ratings, min_num_ratings, verbose=False, p_test=0.1):
    """
    Splits the data set in train and test and compute the RMSE using as prediction the user mean.
    :param ratings: initial data set (sparse matrix of size nxp, n items and p users)
    :param min_num_ratings: all users and items must have at least min_num_ratings per user and per item to be kept
    :param verbose: True if user wants details to be printed
    :param p_test share of the data set to be dedicated to test set
    :return: RMSE value of the prediction using user means as a predictions
    """
    _, train, test = split_data(ratings, min_num_ratings, verbose=verbose, p_test=p_test)
    cumulated_rmse = 0

    # find the RMSE share due to all users
    for user in range(train.shape[1]):

        # compute the mean of non-zero rating for current user
        current_train_ratings = train[:, user]
        current_non_zero_train_ratings = current_train_ratings[current_train_ratings.nonzero()]

        if current_non_zero_train_ratings.shape[0] != 0:
            mean = current_non_zero_train_ratings.mean()
            # compute the rmse with all non-zero ratings of current user in test set
            current_test_ratings = test[:, user]
            current_non_zero_test_ratings = current_test_ratings[current_test_ratings.nonzero()].todense()
            cumulated_rmse += calculate_mse(current_non_zero_test_ratings, mean)

    cumulated_rmse = np.sqrt(float(cumulated_rmse) / test.nnz)

    return cumulated_rmse


def write_user_mean_prediction(means, input_path='../data/sample_submission.csv',
                               output_path='../data/user_mean_prediction.csv', verbose=False):
    """
    Write a prediction file for Kaggle submission using user mean.
    :param means: list of user means
    :param input_path: path to the sample submission provided by the Kaggle page (entries we have to predict)
    :param output_path: path to output the prediction file
    :param verbose: if True, details of computation are printed
    """
    test = load_data(input_path, verbose=verbose)
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    with open(output_path, 'w') as output:
        output.write('Id,Prediction\n')
        for row, col in nnz_test:
            output.write('r{}_c{},{}\n'.format(row + 1, col + 1, means[col]))
