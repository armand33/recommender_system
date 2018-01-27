import numpy as np


def booleanViewings(ratings):
    """
    Converts the information on which coefficients are non-zero into a boolean matrix
    :param ratings: Sparse matrix of the ratings, with non-zero coefficients when a rating is available
    :return: "Boolean" matrix with a 1 at position (f, u) if the rating (f, u) is available in the ratings matrix ; a 0
              otherwise
    """
    nF, nU = ratings.shape
    viewings = np.zeros((nF, nU))
    nnz_rows, nnz_cols = ratings.nonzero()
    for f, u in list(zip(nnz_rows, nnz_cols)):
        viewings[f, u] = 1
    return viewings


def sparseToDense(S):
    """
    Converts a scipy.sparse sparse matrix into a numpy dense array
    :param S: Input sparse matrix
    :return: Corresponding numpy array
    """
    return np.asarray(S.toarray())


def normalizeRatings(denseRatings, boolViewings):
    """
    Normalizes the ratings on a per-user basis
    :param denseRatings: The ratings matrix, given as a numpy array
    :param boolViewings: A boolean (0 or 1) matrix indicating the disponibility of the ratings (result of
           booleanViewings())
    :return: A 2D numpy array containing the normalized ratings, along with two 1D arrays, containing respectively the
             mean rating and the standard deviation of the ratings of each user
    The normalization process consists in replacing the rating r of a (f, u) film-user couple by (r - m)/s, where m, s
    are the mean/standard deviation of the ratings of u
    """
    user_means = np.sum(denseRatings, axis=0) / np.sum(boolViewings, axis=0)
    user_stdDevs = (np.sum(denseRatings * denseRatings, axis=0) / np.sum(boolViewings,
                                                                         axis=0)) - user_means * user_means
    normalized_ratings = boolViewings * (denseRatings - user_means) / user_stdDevs
    return normalized_ratings, user_means, user_stdDevs


def buildUsersGraph(normRatings, boolViewings, film_weights=[], verbose=True):
    """
    Builds a graph structure between the users, measuring the similarity of ratings of users who have seen at least one
    common film
    :param normRatings: The normalized ratings, result of normalizeRatings()
    :param boolViewings: A boolean (0 or 1) matrix indicating the disponibility of the ratings
                         (result of booleanViewings())
    :param film_weights: Optional weights, one for each film, applied to the terms concerning this film
                         in the similarity sum
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: A similarity matrix between users, and a common viewings matrix indicating for users (u, v) the number of
             common films graded by both u and v
    The two output matrices describe the structure of a graph, whose vertices are the users, and an edge connects two
    users if their number of common viewings is > 0
    The similarity matrix provides weights for these edges, corresponding to a measure of how similar are the gradings
    of the connected users
    The similarity between two users is computed as the sum, over the films f they have both seen, of the product of
    their two grades of f, multiplied optionally by the weight of f
    This corresponds to a non-normalized correlation measure. Possible film weights are for instance the standard
    deviation of the gradings of each film, to give less weight to the films on which all users agree (this has not
    proved useful as of now)
    """
    nF, nU = normRatings.shape
    if len(film_weights) != nF:
        film_weights = np.ones(nF)
    similarities = np.zeros((nU, nU))
    user_commonViewings = np.zeros((nU, nU))
    for f in range(nF):
        if verbose and f % 100 == 0:
            print(f)
        users = np.where(boolViewings[f, :] == 1)[0]
        nRatings = len(users)
        for i in range(nRatings):
            for j in range(i + 1, nRatings):
                similarities[users[i], users[j]] += normRatings[f, users[i]] * normRatings[f, users[j]] * film_weights[
                    f]
                similarities[users[j], users[i]] = similarities[users[i], users[j]]
                user_commonViewings[users[i], users[j]] += 1
                user_commonViewings[users[j], users[i]] += 1
    return similarities, user_commonViewings


def buildFilmsGraph(normRatings, boolViewings, user_weights=[], verbose=True):
    """
    Builds a graph structure between the films, measuring the similarity of ratings of films who have been viewed by
    at least a common user
    :param normRatings: The normalized ratings, result of normalizeRatings()
    :param boolViewings: A boolean (0 or 1) matrix indicating the disponibility of the ratings
    (result of booleanViewings())
    :param user_weights: Optional weights, one for each user, applied to the terms concerning this user in the
                         similarity sum
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: A similarity matrix between films, and a common viewers matrix indicating for films (f, g) the number of
    common users having graded both f and g
    The return arguments contain the structure of a graph, whose vertices are the films, and an edge connects two films
    if their number of common viewers is > 0
    The similarity matrix provides weights for these edges, corresponding to a measure of how similar are the gradings
    of the connected films
    The similarity between two films is computed as the sum, over the users u that have both seen them, of the product
    of his two grades, multiplied optionally by the weight of u
    This corresponds to a non-normalized correlation measure. Using user weights has not been tested as of now.
    """
    return buildUsersGraph(normRatings.transpose(), boolViewings.transpose(), user_weights, verbose)


def buildNeighbors(user_commonViewings):
    """
    Extracts the list of neighbors of each user from the commonViewings matrix
    :param user_commonViewings: Matrix indicating the number of common viewed films for each two users, as returned by
                                buildUsersGraph()
    :return: A list of lists, indicating for each user his neighbors in the graph, ie the users which have at least one
    film in common with him
    Note : can be used in the same way for the films point of view based model
    """
    nU = user_commonViewings.shape[0]
    neighbors = []
    for ui in range(nU):
        n = []
        for uj in range(nU):
            if user_commonViewings[ui, uj] > 0:
                n.append(uj)
        neighbors.append(n)
    return neighbors


def sortNeighbors(neighbors, similarities, useTuples=True, verbose=True):
    """
    Sorts each list of neighbors in decreasing order according to similarities between users
    :param neighbors: List of lists indicating the neighbors of each user, as returned by buildNeighbors()
    :param similarities: Similarity matrix between users, as returned by buildUsersGraph()
    :param useTuples: Boolean parameter, indicating if the function should return a list of list of tuples.
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: A list of lists of tuples, with for each user its neighbors in decreasing order of similarity, in the form
    of (neighbor index, similarity) tuples.
    Only neighbors with positive similarity are keeped. If useTuples is set to False, the output is a simple list of
    lists, containing only neighbor indices instead of tuples to save memory
    Note : can be used in the same way for the films point of view based model
    """
    sorted_neighbors = []
    nU = similarities.shape[0]
    for ui in range(nU):
        if verbose and ui % 100 == 0:
            print(ui)
        n = [(uj, similarities[ui, uj]) for uj in neighbors[ui] if similarities[ui, uj] > 0]
        n = sorted(n, key=lambda x: x[1], reverse=True)
        if not useTuples:
            n1 = []
            for i, j in n:
                n1.append(i)
            sorted_neighbors.append(n1)  # To save memory
        else:
            sorted_neighbors.append(n)
    return sorted_neighbors


def buildUsersModel(ratings, verbose=True):
    """
    Does the whole model building process from the users point of view given the initial sparse ratings matrix
    :param ratings: The initial sparse ratings matrix
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: The built model, as a dictionary containing the following entries:
        - 'nF' : The number of films (ie number of rows in the ratings matrix)
        - 'nU' : The number of users (ie number of columns in the ratings matrix)
        - 'boolViewings' : The boolean (0 or 1) matrix indicating which user has seen which film
        - 'denseRatings' : The ratings matrix in a dense form (numpy array)
        - 'normRatings' : A dense matrix containing the normalized (per user) ratings
        - 'userMeans' : The vector of mean grades given by users
        - 'userStdDevs' : The vector of standard deviations of the grades given by users
        - 'sortedNeighbors' : The list of lists of the neighbors of each user, along with similarities, in decreasing
                              order of similarity
    """
    nF, nU = ratings.shape
    boolViewings = booleanViewings(ratings)
    denseRatings = sparseToDense(ratings)
    normRatings, userMeans, userStdDevs = normalizeRatings(denseRatings, boolViewings)
    similarities, user_commonViewings = buildUsersGraph(normRatings, boolViewings, verbose=verbose)
    neighbors = buildNeighbors(user_commonViewings)
    sorted_neighbors = sortNeighbors(neighbors, similarities, useTuples=True, verbose=verbose)
    model = {'nF': nF, 'nU': nU, 'boolViewings': boolViewings, 'denseRatings': denseRatings, 'normRatings': normRatings,
             'userMeans': userMeans, 'userStdDevs': userStdDevs, 'sortedNeighbors': sorted_neighbors}
    return model


def buildFilmsModel(ratings, verbose=True):
    """
    Does the whole model building process from the films point of view given the initial sparse ratings matrix
    :param ratings: The initial sparse ratings matrix
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: The built model, as a dictionary containing the following entries:
        - 'nF' : The number of films (ie number of rows in the ratings matrix)
        - 'nU' : The number of users (ie number of columns in the ratings matrix)
        - 'boolViewings' : The boolean (0 or 1) matrix indicating which user has seen which film
        - 'denseRatings' : The ratings matrix in a dense form (numpy array)
        - 'normRatings' : A dense matrix containing the normalized (per user) ratings
        - 'userMeans' : The vector of mean grades given by users
        - 'userStdDevs' : The vector of standard deviations of the grades given by users
        - 'similarities' : The matrix of similarities between films
        - 'sortedNeighbors' : The list of lists of the neighbors of each film, in decreasing order of similarity
    """
    nF, nU = ratings.shape
    if verbose:
        print('Preprocessing data')
    boolViewings = booleanViewings(ratings)
    denseRatings = sparseToDense(ratings)
    normRatings, userMeans, userStdDevs = normalizeRatings(denseRatings, boolViewings)
    if verbose:
        print('Building graph')
    similarities, film_commonViewings = buildFilmsGraph(normRatings, boolViewings, verbose=verbose)
    if verbose:
        print('Extracting neighbors')
    neighbors = buildNeighbors(film_commonViewings)
    if verbose:
        print('Sorting neighbors')
    sorted_neighbors = sortNeighbors(neighbors, similarities, useTuples=False, verbose=verbose)
    model = {'nF': nF, 'nU': nU, 'boolViewings': boolViewings, 'denseRatings': denseRatings, 'normRatings': normRatings,
             'userMeans': userMeans, 'userStdDevs': userStdDevs, 'similarities': similarities,
             'sortedNeighbors': sorted_neighbors}
    if verbose:
        print('Model built')
    return model


def usersPrediction(f, u, k, model, verbose=True):
    """
    Predicts the grade of a user for one film given the (users point of view based) model and the parameter k
    :param f: The film index
    :param u: The user index
    :param k: The parameter k of the k nearest neighbors algorithm
    :param model: The prediction model, as built by buildUsersModel()
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: An estimation as a floating point number of the grade the user would give to the film
    The prediction is made by taking as many neighbors as possible of u, but at most k, who have seen the film f, chosen
    by decreasing order of similarity with u.
    The average of their normalized ratings, weighted by their similarities with u, is then computed as a prediction of
    the normalized grade of u for f.
    The grade is then multiplied by u's standard deviation, added to u's mean grade, and culled if necessary between 1
    and 5, to give the final estimate.
    If no neighbors of u with a positive similarity with u have seen the film f, the prediction is simply the average
    grade given by u.
    """
    count = 0
    ref_neighbors = []
    n_neighbors = len(model['sortedNeighbors'][u])
    while len(ref_neighbors) < k and count < n_neighbors:
        n, sim = model['sortedNeighbors'][u][count]
        if model['boolViewings'][f, n] == 1 and sim > 0:
            ref_neighbors.append((n, sim))
        count += 1
    if len(ref_neighbors) == 0:
        if verbose:
            print("No (correlated) neighbors have seen the film, returning average grade")
        return model['userMeans'][u]
    meanGrade = 0
    totWeights = 0
    for n, sim in ref_neighbors:
        meanGrade += (model['userMeans'][u] + model['userStdDevs'][u] * model['normRatings'][f, n]) * sim
        totWeights += sim
    meanGrade /= totWeights
    meanGrade = min(meanGrade, 5)
    meanGrade = max(meanGrade, 1)
    return meanGrade


def filmsPrediction(f, u, k, model, verbose=True):
    """
    Predicts the grade of a user for one film given the (films point of view based) model and the parameter k
    :param f: The film index
    :param u: The user index
    :param k: The parameter k of the k nearest neighbors algorithm
    :param model: The prediction model, as built by buildFilmsModel()
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: An estimation as a floating point number of the grade the user would give to the film
    The prediction is made by taking as many neighbors as possible of f, but at most k, who have been graded by u,
    chosen by decreasing order of similarity with f.
    The average of their normalized grades by u, weighted by their similarities with f, is then computed as a prediction
    of the normalized grade of u for f.
    The grade is then multiplied by u's standard deviation, added to u's mean grade, and culled if necessary between 1
    and 5, to give the final estimate.
    If no neighbors of f with a positive similarity with f have been graded by u, the prediction is simply the average
    grade given by u.
    """
    count = 0
    ref_neighbors = []
    n_neighbors = len(model['sortedNeighbors'][f])
    while len(ref_neighbors) < k and count < n_neighbors:
        n = model['sortedNeighbors'][f][count]
        sim = model['similarities'][f, n]
        if model['boolViewings'][n, u] == 1 and sim > 0:
            ref_neighbors.append((n, sim))
        count += 1
    if len(ref_neighbors) == 0:
        if verbose:
            print("No (correlated) neighbors have been seen by the user")
        return model['userMeans'][u]
    meanGrade = 0
    totWeights = 0
    for n, sim in ref_neighbors:
        meanGrade += model['denseRatings'][n, u] * sim
        totWeights += sim
    meanGrade /= totWeights
    meanGrade = min(meanGrade, 5)
    meanGrade = max(meanGrade, 1)
    return meanGrade


def usersModel_predictionErrorsOverk(k_list, model, testSet=[], verbose=True):
    """
    Computes the prediction train error, and optionally test error, for different values of the parameter k using a user
    point of view based model.
    :param k_list: List of  values of k to use
    :param model: Users based model for making predictions
    :param testSet: Optional testSet, given as a sparse matrix whose non zero values are used as ground truth
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: The list of train errors over k, and the (possibly empty) list of test errors over k
    """
    tr_errors = []
    te_errors = []

    for k in k_list:

        if verbose:
            print('k={}:'.format(k))

        count = 0
        mse = 0
        for u in range(model['nU']):
            if verbose and u % 100 == 0:
                print('User #{}'.format(u + 1))
            for f in range(model['nF']):
                if model['boolViewings'][f, u] == 1:
                    mse += (model['denseRatings'][f, u] - usersPrediction(f, u, k, model, verbose=verbose)) ** 2
                    count += 1
        tr_rmse = np.sqrt(mse / count)
        tr_errors.append(tr_rmse)
        if verbose:
            print("Train RMSE : {}".format(tr_rmse))

        if len(testSet) > 0:
            count = 0
            mse = 0
            nnz_rows, nnz_cols = testSet.nonzero()
            for f, u in list(zip(nnz_rows, nnz_cols)):
                mse += (testSet[f, u] - usersPrediction(f, u, k, model, verbose=verbose)) ** 2
                count += 1
            te_rmse = np.sqrt(mse / count)
            te_errors.append(te_rmse)
            if verbose:
                ("Test RMSE : {}".format(te_rmse))

    return tr_errors, te_errors


def filmsModel_predictionErrorsOverk(k_list, model, testSet=[], verbose=False):
    """
    Computes the prediction train error, and optionnaly test error, for different values of the parameter k using a
    film point of view based model.
    :param k_list: List of  values of k to use
    :param model: Films based model for making predictions
    :param testSet: Optional testSet, given as a sparse matrix whose non zero values are used as ground truth
    :param verbose: If set to True, status prints are made during the execution of the code
    :return: The list of train errors over k, and the (possibly empty) list of test errors over k
    """
    tr_errors = []
    te_errors = []

    for k in k_list:

        if verbose:
            print('k={}:'.format(k))

        count = 0
        mse = 0
        for u in range(model['nU']):
            if verbose and u % 100 == 0:
                print('User #{}'.format(u + 1))
            for f in range(model['nF']):
                if model['boolViewings'][f, u] == 1:
                    mse += (model['denseRatings'][f, u] - filmsPrediction(f, u, k, model, verbose=verbose)) ** 2
                    count += 1
        tr_rmse = np.sqrt(mse / count)
        tr_errors.append(tr_rmse)
        if verbose:
            print("Train RMSE : {}".format(tr_rmse))

        if len(testSet) > 0:
            count = 0
            mse = 0
            nnz_rows, nnz_cols = testSet.nonzero()
            for f, u in list(zip(nnz_rows, nnz_cols)):
                mse += (testSet[f, u] - filmsPrediction(f, u, k, model, verbose=verbose)) ** 2
                count += 1
            te_rmse = np.sqrt(mse / count)
            te_errors.append(te_rmse)
            if verbose:
                ("Test RMSE : {}".format(te_rmse))

    return tr_errors, te_errors


def create_prediction_file_usersModel(outputPath, testSet, model, k, verbose=True):
    """
    Creates a CSV output file containing predictions on the test set made by a user based model
    :param outputPath: Path of the file to create
    :param testSet: Sparse matrix, whose non zero entries indicate the predictions to make
    :param model: User based model to use for making the predictions
    :param k: Parameter k of the k nearest neighbors algorithm for making predictions
    :param verbose: If set to True, status prints are made during the execution of the code
    """
    with open(outputPath, 'w') as output:
        output.write('Id,Prediction\n')
        nnz_row, nnz_col = testSet.nonzero()
        for f, u in list(zip(nnz_row, nnz_col)):
            pred = usersPrediction(f, u, k, model, verbose=verbose)
            output.write('r{}_c{},{}\n'.format(f + 1, u + 1, pred))


def create_prediction_file_filmsModel(outputPath, testSet, model, k, verbose=False):
    """
    Creates a CSV output file containing predictions on the test set made by a film based model
    :param outputPath: Path of the file to create
    :param testSet: Sparse matrix, whose non zero entries indicate the predictions to make
    :param model: Film based model to use for making the predictions
    :param k: Parameter k of the k nearest neighbors algorithm for making predictions
    :param verbose: If set to True, status prints are made during the execution of the code
    """
    with open(outputPath, 'w') as output:
        output.write('Id,Prediction\n')
        nnz_row, nnz_col = testSet.nonzero()
        for f, u in list(zip(nnz_row, nnz_col)):
            pred = filmsPrediction(f, u, k, model, verbose=verbose)
            output.write('r{}_c{},{}\n'.format(f + 1, u + 1, pred))


def writeCSV(outputPath, ratings):
    """
    Creates a CSV output file from a ratings sparse matrix
    :param outputPath: Path of the file to create
    :param ratings: Sparse matrix, whose non zero entries indicate the entries to write into the CSV file
    """
    with open(outputPath, 'w') as output:
        output.write('Id,Prediction\n')
        nnz_row, nnz_col = ratings.nonzero()
        for f, u in list(zip(nnz_row, nnz_col)):
            pred = ratings[f, u]
            output.write('r{}_c{},{}\n'.format(f + 1, u + 1, pred))


def RMSE(ratings, groundTruth):
    """
    Computes the Root Mean Squared Error of a set of ratings given the ground truth:
    :param ratings: The under evaluation ratings sparse matrix
    :param groundTruth: The ground truth ratings sparse matrix
    :return: The RMSE of the ratings as compared to ground truth
    """
    nnz_row, nnz_col = ratings.nonzero()
    mse = 0
    count = 0
    for f, u in list(zip(nnz_row, nnz_col)):
        mse += (ratings[f, u] - groundTruth[f, u]) ** 2
        count += 1
    rmse = np.sqrt(mse / count)
    return rmse
