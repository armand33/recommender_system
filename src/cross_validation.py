# !/usr/bin/env python

import numpy as np
import scipy.sparse as sp

from matrix_factorization import matrix_factorization_als, matrix_factorization_sgd


def k_fold_split(ratings, min_num_ratings=10, k=4):
    """
    Creates the k (training set, test_set) used for k_fold cross validation
    :param ratings: initial sparse matrix of shape (num_items, num_users)
    :param min_num_ratings: all users and items must have at least min_num_ratings per user and per item to be kept
    :param k: number of fold
    :return: a list fold of length k such that
                - fold[l][0] is a list of tuples (i,j) of the entries of 'ratings' that are the l-th testing set
                - fold[l][1] is a list of tuples (i,j) of the entries of 'ratings' that are the l-th training set
    """
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()

    # set seed
    np.random.seed(988)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]

    nnz_row, nnz_col = valid_ratings.nonzero()
    nnz = list(zip(nnz_row, nnz_col))

    nnz = np.random.permutation(nnz)

    len_splits = int(len(nnz) / k)
    splits = []
    for i in range(k):
        splits.append(nnz[i * len_splits: (i + 1) * len_splits])

    splits = [f.tolist() for f in splits]

    folds = []
    for i in range(k):
        tmp = []
        for j in range(k):
            if j != i:
                tmp = tmp + splits[j]
        folds.append([splits[i], tmp])

    return folds


def split_matrix(ratings, indices):
    """
    Splits the initial ratings matrix in train and test using the indices provided
    :param ratings: initial sparse matrix of shape (num_items, num_features)
    :param indices: list of length 2 such that:
                - indices[0] is a list of tuples (i,j) of the entries of 'ratings' that are the testing set
                - indices[1] is a list of tuples (i,j) of the entries of 'ratings' that are the training set
            Typically we have indices = k_fold_split(ratings, k)[l] for l in range(k)/
    :return: train and test matrices (both sparse matrices of shape (num_items, num_features))
    """
    train = sp.lil_matrix(ratings.shape)
    test = sp.lil_matrix(ratings.shape)

    for i, j in indices[0]:
        test[i, j] = ratings[i, j]

    for i, j in indices[1]:
        train[i, j] = ratings[i, j]
    return train, test


def cross_validation_step_sgd(ratings, k_fold=4, gamma=0.01, num_features=25,
                              lambda_user=0.1, lambda_item=0.7, num_epochs=20, verbose=False):
    """
    Performs a cross validation given the parameters and return the average training RMSE and testing RMSE.
    :param ratings: initial sparse matrix of shape (num_items, num_users)
    :param k_fold: number of folds wanted
    :param gamma: step size
    :param num_features: used for matrix factorization
    :param lambda_user: penalization coefficient for user features optimization
    :param lambda_item: penalization coefficient for item features optimization
    :param num_epochs: number of epochs to perform is the SGD
    :param verbose: if True, details of the computation are printed
    :return: mean train error, mean test error
    """
    tr_errors = []
    te_errors = []
    folds = k_fold_split(ratings, k=k_fold)
    for fold in range(k_fold):
        print('Fold {}/{}'.format(fold+1, k_fold))
        train, test = split_matrix(ratings, folds[fold])
        train_rmse, test_rmse, _, _ = matrix_factorization_sgd(train, test, gamma=gamma, num_features=num_features,
                                                               lambda_user=lambda_user, lambda_item=lambda_item,
                                                               num_epochs=num_epochs, verbose=verbose)
        tr_errors.append(train_rmse)
        te_errors.append(test_rmse)
    return np.mean(tr_errors), np.mean(te_errors)


def cross_validation_step_als(ratings, k_fold=4, stop_criterion=1e-4, num_features=25,
                              lambda_user=0.1, lambda_item=0.7, verbose=False):
    """
    Performs a cross validation given the parameters and return the average training RMSE and testing RMSE.
    :param ratings: initial sparse matrix of shape (num_items, num_users)
    :param k_fold: number of folds wanted
    :param stop_criterion: computation continues until an optimization step does not improve the error by more than
                           this value
    :param num_features: used for matrix factorization
    :param lambda_user: penalization coefficient for user features optimization
    :param lambda_item: penalization coefficient for item features optimization
    :param verbose: if True, details of the computation are printed
    :return: mean train error, mean test error
    """
    tr_errors = []
    te_errors = []
    folds = k_fold_split(ratings, k=k_fold)
    for fold in range(k_fold):
        print('Fold {}/{}'.format(fold+1, k_fold))
        train, test = split_matrix(ratings, folds[fold])
        train_rmse, test_rmse, _, _ = matrix_factorization_als(train, test, num_features=num_features,
                                                               lambda_user=lambda_user, lambda_item=lambda_item,
                                                               stop_criterion=stop_criterion, verbose=verbose)
        tr_errors.append(train_rmse)
        te_errors.append(test_rmse)
    return np.mean(tr_errors), np.mean(te_errors)
