from datetime import datetime
import numpy as np
import pandas as pd
import os

from external.bayesian_benchmarks.data import get_regression_data, Dataset, add_regression


def latex_name(name: str) -> str:
    """
    Converts a dataset string into a latex command.
    :param name:
        name of the dataset
    :return:
        a latex command for the dataset
    """
    if name.startswith('tamilnadu_electricity'):
        name = 'tamilnadu'
    elif name.startswith('pm25'):
        name = 'pm'
    return '\\' + name.upper() + '{}'


def load_dataset(name: str) -> (np.ndarray, np.ndarray):
    """
    Loads a dataset and returns a standardized numpy array.
    :param name:
        name of the dataset
    :return:
        standardized dataset as (N, D) array, where N is the size
    """

    # the zeros_ dataset can not be standardized
    if name.startswith('zeros_'):
        _, N = name.split('zeros_')
        return np.zeros((int(N), 1)), np.zeros((int(N), 1))
    X, y = load_raw_dataset(name)
    return standardize(X)[0], standardize(y)[0]

# Copyright 2021 The CGLB Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std, mu, std

# Copyright 2021 The CGLB Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def norm_dataset(data: Dataset) -> Dataset:
    """Normalise dataset tuple."""
    return norm(data[0]), norm(data[1])


def get_train_test_dataset(name: str, seed: int = 0) -> (np.ndarray, np.ndarray):
    if name.startswith('toydata'):
        import torch
        _, N = name.split('toydata')
        if N == '':
            X, y, X_star, y_star = generate_toy_data(seed=seed)
        else:
            X, y, X_star, y_star = generate_toy_data(int(N), seed=seed)
        return X, y, X_star, y_star

    # first we see if the name corresponds to one of our datasets
    try:
        X, y = load_raw_dataset(name)
        name_ = name
        @add_regression
        class OurDataset(Dataset):
            N, D = X.shape
            name = name_
            url = None

            @property
            def needs_download(self):
                return False

            def read_data(self):
                return X, y
    except:
        pass
    data: Dataset = get_regression_data(name, split=seed, prop=2./3.)

    #return data.X_train, data.Y_train, data.X_test, data.Y_test
    #raise NotImplementedError("We are standardizing twice here!")
    # # TODO: even though Artemev's standardization has data leakage use his--for reproducibility
    # X_train, m, s = standardize(dataset.X_train)
    # X_test = standardize(dataset.X_test, mean=m, std=s)[0]
    # Y_train, m, s = standardize(dataset.Y_train)
    # Y_test = standardize(dataset.Y_test, mean=m, std=s)[0]
    # return X_train, Y_train, X_test, Y_test

    train, test = (data.X_train, data.Y_train), (data.X_test, data.Y_test)
    (x_train, x_mu, x_std), (y_train, y_mu, y_std) = norm_dataset(train)
    x_test = (test[0] - x_mu) / x_std
    y_test = (test[1] - y_mu) / y_std
    return x_train, y_train, x_test, y_test


def load_raw_dataset(name: str) -> (np.ndarray, np.ndarray):
    """
    Loads a dataset without standardizing it.
    :param name:
        name of the dataset
    :return:
        (N, D) array, where N is the size
    """
    #data_dir = os.path.dirname(__file__)
    # TODO: bad hack. How to avoid for ACGP as a library? (need to distinguish development mode and normal mode)
    data_dir = "datasets"
    if name == 'pm25':
        def parser(y, m, d, h):
            def check(x):
                if len(x) == 1:
                    return '0'+x
                return x
            return [datetime.strptime(str(yx)+check(mx)+check(dx)+check(hx), "%Y%m%d%H") for (yx, mx, dx, hx) in zip(y, m, d, h)]

        X = pd.read_csv(os.path.join(data_dir, os.path.join('beijing_pm25', 'PRSA_data_2010.1.1-2014.12.31.csv')),
                          sep=',', parse_dates={0: [1, 2, 3, 4]}, usecols=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
                          date_parser=parser, header=None, skiprows=1)
        X[0] = X[0].astype(int) / 10**11  # convert time to continuous variable
        X[0] = X[0] - X[0][0]
        X = pd.get_dummies(X)
        X = X.values
        y = pd.read_csv(os.path.join(data_dir, os.path.join('beijing_pm25', 'PRSA_data_2010.1.1-2014.12.31.csv')),
                          sep=',', usecols=[5], header=None, skiprows=1).values
        idx = np.logical_not(np.isnan(y)).squeeze()
        X = X[idx, :]
        y = y[idx, :]
    elif name == 'bank':
        raise NotImplementedError("This appears to be a classification dataset...")
        data_frame = pd.read_csv(os.path.join(data_dir, os.path.join('bank_marketing', 'bank-full.csv')),
                                 sep=';', header=None, skiprows=1, usecols=range(0, 16))
        data_frame = pd.get_dummies(data_frame)
        X = data_frame.values
    elif name == 'protein':
        X = pd.read_csv(os.path.join(data_dir, os.path.join('protein_tertiary', 'CASP.csv')), sep=',',
                                   header=None, skiprows=1)
        y = X.values[:, :1].copy()
        X = X.values[:, 1:]
    elif name == 'tamilnadu_electricity':
        raise NotImplementedError("not a well defined regression task")
        from scipy.io import arff
        data_frame = pd.DataFrame(arff.loadarff(os.path.join(data_dir, os.path.join('tamilnadu_electricity', 'eb.arff')))[0])
        del data_frame['Sector']  # all values are the same
        data_frame = pd.get_dummies(data_frame)
        X = data_frame.values
    elif name == 'metro':
        csv = pd.read_csv(os.path.join(data_dir, os.path.join('metro', 'Metro_Interstate_Traffic_Volume.csv')),
                          sep=',', skiprows=1, header=None, parse_dates=[7], usecols=range(0, 8))
        csv[7] = csv[7].astype(int) / 10**11  # convert time to continuous variable
        csv[7] = csv[7] - csv[7][0]

        csv = pd.get_dummies(csv)
        X = csv.values[:, :7]
        y = pd.read_csv(os.path.join(data_dir, os.path.join('metro', 'Metro_Interstate_Traffic_Volume.csv')),
                          sep=',', skiprows=1, header=None, usecols=[8]).values
    elif name == 'pumadyn':
        from scipy.io import loadmat
        puma = loadmat(os.path.join(data_dir, os.path.join('pumadyn', 'pumadyn32nm.mat')))
        X = np.vstack([puma['X_tr'], puma['X_tst']])
        y = np.vstack([puma['T_tr'], puma['T_tst']])
    elif name == 'usflight':
        file = os.path.join(data_dir, os.path.join('us_flight', 'us_flight_data_year08.npz'))
        data = np.load(file)
        X = data['x_data']
        y = data['y_data']
    else:
        try:
            # the Bayesian benchmarks dataloader sets the seed which screws with our randomization
            s = np.random.randint(2**16)
            data: Dataset = get_regression_data(name, split=0, prop=1.)
            np.random.seed(s)
            #train, test = (data.X_train, data.Y_train), (data.X_test, data.Y_test)
            X = data.X_train
            y = data.Y_train
        except KeyError:
            raise ValueError('unknown dataset:', name)
    return X, y


def standardize(x_train: np.ndarray, mean=None, std=None) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    standardizes a given (N, D) array along the first dimension
    :param x_train:
        the (N, D) array
    :return:
        the standardized (N, D) array
    """
    if std is None:
        std = np.std(x_train, axis=0)

    x_train = x_train[:, std != 0.]  # remove columns with 0 variance
    std_ = std[std != 0.]

    if mean is None:
        mean = np.mean(x_train, axis=0)
    x_train = x_train - mean
    x_train = x_train / std_

    if np.isnan(x_train).any() or np.isinf(x_train).any():
        raise ValueError("standardized dataset contains NaNs!")
    return x_train, mean, std


def generate_toy_data(N=5000, seed=0):
    import torch
    from .smooth_functions import smooth_function_bias
    with torch.no_grad():
        next_seed = torch.seed()
        torch.random.manual_seed(seed)
        # Train data
        X = 5*torch.rand(N, 1, dtype=torch.float64)
        y = smooth_function_bias(X) + 1.0*torch.randn(N, 1, dtype=torch.float64)
        X_star = torch.linspace(-1.0, 6.0, 500, dtype=X.dtype).reshape(-1, 1)
        y_star = smooth_function_bias(X_star) + 1.0*torch.randn(500, 1, dtype=torch.float64)

        torch.random.manual_seed(next_seed)
        return X, y, X_star, y_star
