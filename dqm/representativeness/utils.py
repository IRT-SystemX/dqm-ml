"""
This module implements two classes, DiscretisationParams and VariableAnalysis,
providing functionality for variable counting, countplot visualization,
and discretization of variables using normal or uniform distributions.
It also includes functions for processing data for chi-square tests, calculating
expected values, and generating histograms for observed and expected values.

Authors:
    Faouzi ADJED
    Anani DJATO

Dependencies:
    numpy
    pandas
    matplotlib.pyplot
    scipy.stats
    dqm.utils.twe_logger
    seaborn

Functions : None

Classes:
    DiscretisationParams: Class for defining discretization parameters
    VariableAnalysis: Class for analyzing data distribution

Example:
from utils import VariableAnalysis, DiscretisationParams

# Example of using VariableAnalysis class
variable_analyzer = VariableAnalysis()

# Example of using the variable_counting method
my_variable = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
counts = variable_analyzer.variable_counting(my_variable)
print("Counts of unique values:")
print(counts)

# Example of using the countplot method
variable_analyzer.countplot(my_variable)
plt.show()

# Instantiate the DiscretisationParams class
discretisation_params = DiscretisationParams(
    data=my_variable,
    distribution_theory='normal',
    distribution_empirical=[-1.0, 0.0, 1.0, 2.0],
    mean=0.0,
    std=1.0
)
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from dqm.utils.twe_logger import get_logger
import seaborn as sns

logger = get_logger()


class DiscretisationParams:
    """
    Parameters for discretization.

    Args:
        data: Input data.
        distribution_params: Dictionary containing distribution parameters.
            'theory': Distribution theory ('normal' or 'uniform').
            'empirical': Empirical distribution used for discretization.
            'mean': Mean parameter for the distribution theory.
            'std': Standard deviation for the distribution theory.

    Methods:
        __init__:
            Initializes an instance of the DiscretisationParams class.

            Args:
                data: Input data.
                distribution_params: Dictionary containing distribution parameters.

            Returns:
                None

        to_dict:
            Converts the parameters to a dictionary.

            Returns:
                dict: A dictionary representation of the parameters.

            Note:
                This method is not necessary. It was created solely to have at
                least 2 methods as recommended in a class.

        get_data:
            Gets the input data.

            Returns:
                Any: The input data.
    """
    def __init__(self, data, distribution_params):
        """
        Initializes an instance of the DiscretisationParams class.

        Args:
            data (pd.Series): Input data.
            distribution_params (dict): Dictionary containing distribution parameters.
                'theory': Distribution theory ('normal' or 'uniform').
                'empirical': Empirical distribution used for discretization.
                'mean': Mean parameter for the distribution theory.
                'std': Standard deviation for the distribution theory.

        Returns:
            None
        """
        self.data = data
        self.distribution_theory = distribution_params['theory']
        self.distribution_empirical = distribution_params['empirical']
        self.mean = distribution_params['mean']
        self.std = distribution_params['std']

    def to_dict(self):
        """
        Convert the parameters to a dictionary.

        Returns:
            dict: A dictionary representation of the parameters.

        Note:
            This method is not necessary. It was created solely to have at
            least 2 methods as recommended in a class.
        """
        return {
            'data': self.data,
            'distribution_theory': self.distribution_theory,
            'distribution_empirical': self.distribution_empirical,
            'mean': self.mean,
            'std': self.std
        }

    def get_data(self):
        """
        Get the input data.

        Returns:
            Any: The input data.
        """
        return self.data


class VariableAnalysis:
    """
    This class provides functions for variable counting, countplot visualization,
    and discretization of variables using normal or uniform distributions.
    It includes functions for processing data for chi-square tests,
    calculating expected values, and generating histograms for observed and expected values.

    Args: None

    Methods:
        variable_counting
        countplot
        discretisation
        normal_discretization
        data_processing_for_chisqure_test
        uniform_discretization
        discretisation_intervals
        delete_na
        expected
        expected_hist
        observed_hist
    """

    def variable_counting(self, variable: pd.Series) -> pd.DataFrame:
        """
        Counting unique values (only int values and modalities.
        It cannot be used for float values)

        Args:
            variable (panda.Series)

        Returns:
            variable_count (DataFrame): counts of unique values
        """
        variable_count = variable.value_counts().to_frame()
        variable_count.columns = ["count"]
        variable_count.sort_index(inplace=True)
        return variable_count

    def countplot(self, variable: pd.Series) -> Optional[None]:
        """
        This function will not be used and will be deleted in the final package (to decide)
        Show the counts of observations of every category

        Args:
            variable (DataFrame)

        Returns:
            countplot (show the bar plot of counts of variable)
        """
        plt.figure(figsize=(10, 5))
        sns.countplot(x=variable)

    def discretisation(
        self,
        variable: pd.Series,
        distribution: str,
        bins: int
    ) -> List[Union[float, int]]:
        """ Discretisation of variable into bins

        Args:
            distribution (string): 'normal' ou 'uniform'
            variable (Series)
            bins (int)

        Returns:
            interval (array): discretised variable into bins
        """
        interval = []

        if distribution == 'normal':
            mean = np.mean(variable)
            std = np.std(variable)
            for i in range(1, bins):
                val = stats.norm.ppf(i / bins, mean, std)
                interval.append(val)

        elif distribution == 'uniform':
            min_value = variable.min()
            max_value = variable.max()
            for i in range(1, bins):
                val = stats.uniform.ppf(i / bins, min_value, max_value)
                interval.append(val)

        interval.insert(0, -np.inf)
        interval.append(np.inf)
        return interval

    def normal_discretization(
        self,
        bins: int,
        mean: float,
        std: float
    ) -> List[float]:
        """
        normal Discretisation of variable into bins

        Args:
            bins (int): int
            mean (float): the first parameter of the gaussian distribution
            std (float): standard

        Returns
            interval (array): discretised variable into bins
        """
        interval = []
        for i in range(1, bins):
            val = stats.norm.ppf(i / bins, mean, std)
            interval.append(val)
        interval.insert(0, -np.inf)
        interval.append(np.inf)
        return interval

    def uniform_discretization(
        self,
        bins: int,
        min_value: float,
        max_value: float,
    ) -> List[float]:
        """
        This function discretizes a variable with a uniform distribution into specified bins.
        It uses the inverse transform method with the scipy.stats.uniform.ppf function.

        Args:
            bins (int): Number of bins.
            min_value (float): Minimum value for the uniform distribution.
            max_value (float): Maximum value for the uniform distribution.

        Returns:
            interval (list): Discretized variable into bins.
              The list includes intervals with the first element representing negative infinity
              and the last element representing positive infinity.
        """
        interval = []
        for i in range(1, bins):
            val = stats.uniform.ppf(i / bins, min_value, max_value)
            interval.append(val)
        interval.insert(0, -np.inf)
        interval.append(np.inf)
        return interval

    def data_processing_for_chisqure_test(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This function is designed to preprocess the input data for chi-square tests.
        If the data type is object ('O'), it is assumed to be categorical,
        and the function converts it into value counts.
        This step is crucial for chi-square tests, which require frequency distributions.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            data (pd.DataFrame):
                Processed data suitable for chi-square tests.
        """
        if data.dtypes == 'O':
            data = data.value_counts()
        return data

    def discretisation_intervals(
        self,
        params: DiscretisationParams
    ) -> Optional[pd.DataFrame]:
        """
        This function discretizes a given set of data into intervals based on
        empirical distribution and calculates observed and expected frequencies
        for each interval. It supports both normal and uniform distribution theories.

        Args:
            params (DiscretisationParams): Parameters for discretization.

        Returns:
            intervals (Optional[DataFrame]): Intervals and counts of each interval.
              Returns None if an unsupported distribution theory is provided.

        Note:
        The function may issue a warning if there are missing values in the data.

        Example:
            interval_data = discretisation_intervals(
                DiscretisationParams(
                    data, {
                        'theory': 'normal',
                        'empirical': distribution_empirical,
                        'mean': mean, 'std': std
                    }
                )
            )
            if interval_data is not None:
                logger.info(interval_data)
        """
        alpha = 1.0
        processed_data = self.delete_na(params.data)

        if len(processed_data) != len(params.data):
            deleted_data = len(params.data) - len(processed_data)
            logger.info("the data is not complete, there are %s missed items", deleted_data)
            params.data = processed_data

        if params.distribution_theory == "normal":
            exp = self.expected(params.distribution_theory, params.data, params.mean, params.std)
        elif params.distribution_theory == "uniform":
            min_value = params.mean
            max_value = params.mean + params.std
            exp = self.expected(params.distribution_theory, params.data, min_value, max_value)

        if params.distribution_theory in ('normal', 'uniform'):
            intervals = pd.DataFrame(
                {'lower_limit': params.distribution_empirical[:-1],
                 'upper_limit': params.distribution_empirical[1:]}
            )

            observed_values = sorted(params.data)
            expected_values = sorted(exp)

            intervals['obs_freq'] = intervals.apply(
                lambda x: sum(x['lower_limit'] < i <= x['upper_limit'] for i in observed_values), axis=1) / alpha

            intervals['exp_freq'] = intervals.apply(
                lambda x: sum(x['lower_limit'] < i <= x['upper_limit'] for i in expected_values), axis=1) / alpha

            return intervals

        # Add a return value (can be None if needed)
        return None

    def delete_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove missing values (NaN) from the input data.

        Args:
            data (pd.DataFrame): The input data containing missing values.

        Returns:
            data (pd.DataFrame):
                The input data with missing values removed. If the input is a Series,
                the output will also be a Series. If the input is a DataFrame,
                the output will be a DataFrame.
        """
        data = data.dropna()
        return data

    def expected(
        self,
        distribution: str,
        data: List[float],
        *argv: float
    ) -> List[float]:
        """
        Calculate the expected values of the distribution

        Args:
            distribution (str): 'normal' or 'uniform'
            data (List[float]): Input data.
            *argv : Parameters of the distribution.

        Returns:
            n or u (List[float]): Expected values for every distribution.
        """
        if distribution == 'normal':
            mean = argv[0]
            std = argv[1]
            n = np.random.normal(mean, std, len(data))
            return n

        # Assuming distribution is 'uniform'
        min_value = argv[0]
        max_value = argv[1]
        u = np.random.uniform(min_value, max_value, len(data))
        return u

    def observed_hist(self, variable: pd.Series) -> None:
        """
        Plot the observed values of the distribution

        Args:
            variable (pd.Series): Input variable.

        Returns:
            None (plots histogram)
        """
        if variable.dtypes == 'int64':
            plt.figure(figsize=(10, 5))
            plt.hist(variable, bins=len(variable.unique()))
            plt.xlabel(variable.name)
        elif variable.dtypes == 'O':
            plt.figure(figsize=(10, 5))
            plt.bar(variable.value_counts().index, variable.value_counts())
            plt.xlabel(variable.name)
        elif variable.dtypes == 'bool':
            plt.figure(figsize=(10, 5))
            true = variable.value_counts()[True]
            false = variable.value_counts()[False]
            plt.bar(['True', 'False'], [true, false])
            plt.xlabel(variable.name)
