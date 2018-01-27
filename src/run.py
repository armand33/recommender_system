import pickle
import numpy as np

from blending import build_matrix, train_als
from helpers import load_data
from utils import get_indices

# load training set
print('Loading training set')
train = load_data('../data/data_train.csv')
nnz_train = get_indices(train)


"""
After some tests, the best score is achieved by only blending the als predictions (that means dilating them).
Another good performance was achieved by blending ALS, user-based  and item based collaborative filtering. Details about 
the tests can be found in the notebook 'src/notebook/blending.ipynb'.
"""


methods = ['als']

# Train the model (actually if the files are present, nothing is done.
print('Training the model')
train_als(train)

# Load trained model and predictions
print('Now loading the pre trained model')
with open(b'../data/pickle/data_train_als.pickle', 'rb') as f:
    als_train = pickle.load(f)
als_predictions = load_data('../data/predictions/als_prediction.csv')

nnz_sub = get_indices(als_predictions)

# Train the blending with Ridge Regression
print('Training the blending with ridge regression')
alpha = 0.1
m_train, y_train = build_matrix(nnz_train, methods, train, None, None, None, None, als_train, None, None, predict=False)
w = np.linalg.solve(m_train.T.dot(m_train) + alpha * np.eye(m_train.shape[1]), m_train.T.dot(y_train))
print('Dilatation coefficient (single ridge coefficient) : ')
print(w)

# Compute training RMSE
print('Train RMSE : {}'.format(np.sqrt(np.mean((y_train - m_train.dot(w)) ** 2))))


# Apply weight vector to final predictions (and not to training ratings anymore)
print('Blending the predictions')
m_sub, _ = build_matrix(nnz_sub, methods, None, None, None, None, None, als_predictions, None, None, predict=True)
y_predict_sub = m_sub.dot(w)

for i in range(len(y_predict_sub)):
    y_predict_sub[i] = min(5, max(1, y_predict_sub[i]))

# Write prediction to file
path = '../data/predictions/blended_prediction.csv'
print("Writing prediction to file : '{}'".format(path))
with open(path, 'w') as output:
    output.write('Id,Prediction\n')
    for i, (row, col) in enumerate(nnz_sub):
        output.write('r{}_c{},{}\n'.format(row + 1, col + 1, y_predict_sub[i]))
