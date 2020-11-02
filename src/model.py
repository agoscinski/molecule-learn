import numpy as np
from copy import deepcopy
from timeit import default_timer as timer
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin

from src.utils import class_name


class store_timings:
    '''
    Small context manager to get timing info on various steps of the KRR
    '''
    def __init__(self, timings, name):
        self.timings = timings
        self.name = name

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        stop = timer()
        self.timings[self.name] = 1000 * (stop - self.start)



class KernelRidgeRegresion(BaseEstimator, RegressorMixin):
    def __init__(self, kernel='linear', sigma=0.01):
        '''
        feature: class computing features from frames
        kernel: sklearn style kernel between features
        regularizer: regularization parameter for KRR
        '''
        self.sigma = sigma
        self.kernel = kernel
        if self.kernel == 'linear':
            self.kernel_function = lambda x,y=None: x.dot(x.T) if y is None else x.dot(y.T)
        if self.kernel == 'GAP':
            self.kernel_function = lambda x,y=None: x.dot(x.T)**2 if y is None else x.dot(y.T)**2
        # store training features, weights & timings
        self.train_features = None
        self.weights = None
        self.timings = {}
    
    # TODO
    #def compute_primal_weights(self):

    def get_params(self, deep=True):
        return {
            'kernel': self.kernel,
            'sigma': self.sigma
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_metadata(self, deep=False):
        return {
            'class': class_name(self),
            'kernel': self.kernel,
            'sigma': self.sigma,
            'timings': self.timings,
        }

    def fit(self, train_features, train_properties, jitter=1e-8):
        self.train_features = train_features
        with store_timings(self.timings, 'train_kernel'):
            kernel = self.kernel_function(self.train_features)

        with store_timings(self.timings, 'invert_kernel'):
            kernel[np.diag_indices_from(kernel)] += self.sigma**2 * np.trace(kernel) / (np.var(train_properties) * len(kernel)) + jitter
            self.weights = np.linalg.solve(kernel, train_properties)
        self.X_fit_ = train_features
        return self

    def predict(self, test_features):
        with store_timings(self.timings, 'predict_kernel'):
            kernel = self.kernel_function(self.train_features, test_features)
        with store_timings(self.timings, 'predict_values'):
            predicted = np.dot(self.weights, kernel)
        return predicted
