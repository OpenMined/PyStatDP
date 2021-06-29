import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath('../'))

from diffprivlib.models import GaussianNB as IBMGaussianNB
from pydp.ml.naive_bayes import GaussianNB as PyDPGaussianNB
from pystatdp import pystatdp
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB as SKLGaussianNB


class Iris:
    def __init__(self, privacy, lower, upper, probability, var_smoothing, X, y, pred_args, pred_len, train_privacy, impl):
        self.X = X
        self.y = y
        self.pred_args = pred_args
        self.pred_len = pred_len
        if impl == 'ibm':
            clf = IBMGaussianNB(epsilon=train_privacy, bounds=(lower, upper), var_smoothing=var_smoothing)
        elif impl == 'pydp':
            clf = PyDPGaussianNB(epsilon=train_privacy, bounds=(lower, upper), probability=probability, var_smoothing=var_smoothing)
        else:  # Non-private baseline.
            clf = SKLGaussianNB(var_smoothing=var_smoothing)
        self.clf = clf

    def quick_result(self, data):
        # Get row of data to remove.
        i = data[0]
        if i != -1:
            # Construct adjacent database.
            self.X = np.append(self.X[:i-1], self.X[i:], axis=0)
            self.y = np.append(self.y[:i-1], self.y[i:], axis=0)
        # Train (private) model.
        self.clf.fit(self.X, self.y)
        # Return model prediction.
        return self.clf.predict(self.pred_args[:self.pred_len]).tolist()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure privacy of DP Gaussian NB on Iris dataset.')
    parser.add_argument('--privacy', type=float, default=0.1, help='midpoint privacy budget for test range')
    parser.add_argument('--train_privacy', type=float, default=1.5, help='claimed privacy budget')
    parser.add_argument('--test_range', type=float, default=0.1, help='test range')
    parser.add_argument('--n_checks', type=int, default=3, help='number of tests to run')
    parser.add_argument('--e_iter', type=int, default=1000, help='number of iterations to run event selection')
    parser.add_argument('--d_iter', type=int, default=5000, help='number of iterations to run hypothesis test')
    parser.add_argument('--pred_len', type=int, default=3, help='length of prediction output')
    parser.add_argument('--impl', default='pydp', help='which NB implementation to use')
    args = parser.parse_args()

    print(vars(args))

    psd = pystatdp()

    # Load data.
    dataset = datasets.load_iris()

    # Use likely misclassified points as prediction data points for the models.
    # These points are those that lie close to the classification decision boundary.
    likely_misclassified = [70, 72, 83, 84, 106, 119, 133, 138]
    pred_args = np.take(dataset.data, likely_misclassified, axis=0)

    lower = np.array([4.3, 2. , 1. , 0.1]) # lower bound of each feature's values
    upper = np.array([7.5, 4. , 6. , 2.]) # upper bound of each feature's values
    probability = 0.002 # probability for geometric distribution
    var_smoothing = 1e-4 # variance smoothing

    algo_params = tuple((lower, upper, probability, var_smoothing, dataset.data, dataset.target, pred_args, args.pred_len, args.train_privacy, args.impl))

    results = psd.main(Iris, algo_params, tuple((args.privacy,)), e_iter=args.e_iter, d_iter=args.d_iter, test_range=args.test_range, n_checks=args.n_checks)