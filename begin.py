
# import system packages
import os
import pickle
import itertools
import logging
import re
import random
import time
import math
from sklearn.cluster import KMeans
import glob
import inspect

# used to parallelize evaluation
from joblib import Parallel, delayed

# numerical methods and arrays
import numpy as np
import pandas as pd

# import packages used for the implementation of sampling methods
from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
                                     cross_val_score, StratifiedKFold, train_test_split)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# some statistical methods
from scipy.stats import skew
import scipy.signal as ssignal
import scipy.spatial as sspatial
import scipy.optimize as soptimize
import scipy.special as sspecial
from scipy.stats.mstats import gmean


__author__ = "György Kovács"
__license__ = "MIT"
__email__ = "gyuriofkovacs@gmail.com"

# for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)


class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = ("The number of minority samples (%d) is not enough "
                 "for sampling")
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True


class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))


class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None)
                or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations


class NoiseFilter(StatisticsMixin,
                  ParameterCheckingMixin,
                  ParameterCombinationsMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
        """

        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self


class TomekLinkRemoval(NoiseFilter):
    """
    Tomek link removal

    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, strategy='remove_majority', n_jobs=1):
        """
        Constructor of the noise filter.

        Args:
            strategy (str): noise removal strategy:
                            'remove_majority'/'remove_both'
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_isin(strategy, 'strategy', [
                        'remove_majority', 'remove_both'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # using 2 neighbors because the first neighbor is the point itself
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        distances, indices = nn.fit(X).kneighbors(X)
        # indices are the index that contain nn
        # array([[2, 1]],[[1,3]])

        # identify links
        links = []
        for i in range(len(indices)):
            if indices[indices[i][1]][1] == i:
                if not y[indices[i][1]] == y[indices[indices[i][1]][1]]:
                    links.append((i, indices[i][1]))

        # determine links to be removed
        to_remove = []
        for li in links:
            if self.strategy == 'remove_majority':
                if y[li[0]] == self.min_label:
                    to_remove.append(li[1])
                else:
                    to_remove.append(li[0])
            elif self.strategy == 'remove_both':
                to_remove.append(li[0])
                to_remove.append(li[1])
            else:
                m = 'No Tomek link strategy %s implemented' % self.strategy
                raise ValueError(self.__class__.__name__ + ": " + m)

        to_remove = list(set(to_remove))

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)


class OverSampling(StatisticsMixin,
                   ParameterCheckingMixin,
                   ParameterCombinationsMixin,
                   RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()

    def sample_between_points_componentwise(self, x, y, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x)*self.random_state.random_sample()
        else:
            return x + (y - x)*self.random_state.random_sample()*mask

    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5)*2.0*std

    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x))-0.5)*2.0 * std

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def sample_with_timing(self, X, y):
        begin = time.time()
        X_samp, y_samp = self.sample(X, y)
        _logger.info(self.__class__.__name__ + ": " +
                     ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()


class SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            # _logger.warning(self.__class__.__name__ +
            #                ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neigh)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbor = X_min[ind[base_indices, neighbor_indices]]

        samples = X_base + np.multiply(self.random_state.rand(n_to_sample,
                                                              1),
                                       X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_TomekLinks(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_tomeklinks_enn,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA},
                    }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations(raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        smote = SMOTE(self.proportion,
                      self.n_neighbors,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)
        X_new, y_new = smote.sample(X, y)

        t = TomekLinkRemoval(strategy='remove_both', n_jobs=self.n_jobs)

        X_samp, y_samp = t.remove_noise(X_new, y_new)

        if len(X_samp) == 0:
            m = ("All samples have been removed, "
                 "returning the original dataset.")
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return X_samp, y_samp

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class CCR(OverSampling):
    """
    References:
        * BibTex::

            @article{ccr,
                    author = {Koziarski, Michał and Wozniak, Michal},
                    year = {2017},
                    month = {12},
                    pages = {727–736},
                    title = {CCR: A combined cleaning and resampling algorithm
                                for imbalanced data classification},
                    volume = {27},
                    journal = {International Journal of Applied Mathematics
                                and Computer Science}
                    }

    Notes:
        * Adapted from https://github.com/michalkoziarski/CCR
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 energy=1.0,
                 scaling=0.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            energy (float): energy parameter
            scaling (float): scaling factor
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(energy, "energy", 0)
        self.check_greater_or_equal(scaling, "scaling", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.energy = energy
        self.scaling = scaling
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'energy': [0.001, 0.0025, 0.005,
                                             0.01, 0.025, 0.05, 0.1,
                                             0.25, 0.5, 1.0, 2.5, 5.0,
                                             10.0, 25.0, 50.0, 100.0],
                                  'scaling': [0.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

        
    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        def taxicab_sample(n, r, d):
            sample = []
            random_numbers = self.random_state.rand(n)

            for i in range(n):
                # spread = r - np.sum(np.abs(sample))
                spread = r
                if len(sample) > 0:
                    spread -= abs(sample[-1])
                sample.append(spread * (2 * random_numbers[i] - 1))

            unit_vector = d / np.linalg.norm(d)
            return self.random_state.permutation(sample)* unit_vector

        minority = X[y == self.min_label]
        majority = X[y == self.maj_label]

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy
            r = 0.0
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            while True:
                if current_majority > len(majority):
                    break

                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / \
                            (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change
                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                dist = distances[i, sorted_distances[current_majority]]
                if dist >= r + radius_change:
                    r += radius_change
                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        cm1 = current_majority - 1
                        last_distance = distances[i, sorted_distances[cm1]]

                    curr_maj_idx = sorted_distances[current_majority]
                    radius_change = distances[i, curr_maj_idx] - last_distance
                    r += radius_change
                    decrease = radius_change * (current_majority + 1.0)
                    remaining_energy -= decrease
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]].astype(float)
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    n_maj_point = len(majority_point)
                    r_num = self.random_state.rand(n_maj_point)
                    r_num = 1e-6 * r_num + 1e-6
                    r_sign = self.random_state.choice([-1.0, 1.0], n_maj_point)
                    majority_point += r_num * r_sign
                    d = np.sum(np.abs(minority_point - majority_point))

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority = majority.astype(float)
        majority += translations

        kmeans = KMeans(n_clusters=1).fit(minority)
        center = kmeans.cluster_centers_
        center_dis = []
        for i in range(len(minority)):
            minority_point = minority[i]
            d = euclidean_distances(minority_point.reshape(1,-1), center)
            center_dis.append(d)
        d_means = sum(center_dis)/len(center_dis)


        appended = []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = n_to_sample / (radii[i] * np.sum(1.0 / radii))
            synthetic_samples = int(np.round(synthetic_samples))
            r = radii[i]

            d = euclidean_distances(minority_point.reshape(1,-1), center)
            ratio = d_means/d
            for _ in range(synthetic_samples):
                rand = np.random.random()*ratio
                if rand * d > r:
                    appended.append(minority_point + taxicab_sample(len(minority_point), r, d))

                else:
                    sample = minority_point + np.multiply(rand, center - minority_point)
                    appended.append(sample)

        if len(appended) == 0:
            _logger.info("No samples were added")
            return X.copy(), y.copy()

        return (np.vstack([X, np.vstack(appended)]),
                np.hstack([y, np.repeat(self.min_label, len(appended))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'energy': self.energy,
                'scaling': self.scaling,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


def perf_measure(self, y_actual, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            fp += 1
        if y_actual[i] == y_pred[i] == 0:
            tn += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            fn += 1
    return tp,fp,tn,fn


def Mcc(tp, fp, tn, fn):
    denominator = math.sqrt(tn * fn + tn * fp + tp * fn + tp * fp)
    numerator = tp * tn - fp * fn
    MCC = numerator / denominator
    return MCC


class Hybrid_sampling(OverSampling):
    def __init__(self,
                 proportion=1.0,
                 energy=1.0,
                 scaling=0.0,
                 n_jobs=1,
                 random_state=None):
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(energy, "energy", 0)
        self.check_greater_or_equal(scaling, "scaling", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.energy = energy
        self.scaling = scaling
        self.n_jobs = n_jobs
        self.MCC = 0
        self.eval = []
        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'energy': [0.001, 0.0025, 0.005,
                                             0.01, 0.025, 0.05, 0.1,
                                             0.25, 0.5, 1.0, 2.5, 5.0,
                                             10.0, 25.0, 50.0, 100.0],
                                  'scaling': [0.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        oversample = CCR()
        X, y = oversample.sample(X, y)
        t = TomekLinkRemoval(strategy='remove_majority', n_jobs=self.n_jobs)

        X_samp, y_samp = t.remove_noise(X, y)

        if len(X_samp) == 0:
            m = ("All samples have been removed, "
                 "returning the original dataset.")
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return X_samp, y_samp




























def get_all_oversamplers():
    """
    Returns all oversampling classes

    Returns:
        list(OverSampling): list of all oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers()
    """

    return OverSampling.__subclasses__()