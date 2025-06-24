"""
This script provides functions for analyzing data distribution using chi-square tests,
goodness-of-fit tests, Kolmogorov-Smirnov tests, Shannon entropy, and confidence intervals.

Authors:
    Faouzi ADJED
    Anani DJATO

Dependencies:
    numpy
    pandas
    matplotlib
    scipy
    seaborn
    dqm.utils.twe_logger

Classes:
    DistributionAnalyzer: Class for analyzing data distribution

Functions: None

Usage: Import this script and use the provided functions for distribution analysis.

"""

from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from dqm.representativeness.utils import VariableAnalysis, DiscretisationParams
from dqm.utils.twe_logger import get_logger

logger = get_logger()
variable_analyzer = VariableAnalysis()


class DistributionAnalyzer:
    """
    Class for analyzing data distribution.

    Args:
    data (pd.DataFrame): The data to be analyzed.
    bins (int): The number of bins for analysis.
    distribution (str): The distribution type ('normal' or 'uniform').

    Methods:
        chisquare_test: Perform the chi-square test on the provided data.
            Returns p-value and confidence intervals

        kolmogorov: Calculate the Kolmogorov-Smirnov test for the chosen distribution.
            Returns the KS test p-value.

        shannon_entropy: Calculate Shannon entropy for the provided intervals.
            Returns Shannon entropy.

        grte: Calculates the Granular Relative and Theoretical Entropy (GRTE) for given data
            Returns The calculated GRTE value and the intervals discretized data
    """

    def __init__(self, data: pd.Series, bins: int, distribution: str):
        """
        Initialize DistributionAnalyzer with the provided data and parameters.

        Args:
            data (pd.Series): The data to be analyzed.
            bins (int): The number of bins for analysis.
            distribution (str): The distribution type ('normal' or 'uniform').
        """
        self.data = data
        self.bins = bins
        self.distribution = distribution
        self.logger = get_logger()
        self.variable_analyzer = VariableAnalysis()

    def chisquare_test(
        self, *par_dist: Optional[Tuple[float, float]]
    ) -> Tuple[float, pd.Series]:
        """
        Perform a chi-square test for goodness of fit.

        This method analyzes the distribution of data using a chi-square test
        for goodness of fit. It supports normal and uniform distributions.

        Args:
            *par_dist (float): Parameters for the specified distribution.

        Returns:
            p-value (float): The p-value from the chi-square test
            intervals_frequencies (pd.DataFrame): The DataFrame containing
                observed and expected frequencies.
        """
        if self.data.dtypes in ("O", "bool"):
            self.logger.error("Categorical or boolean data are not processed yet.")
            return float("nan"), pd.Series(dtype="float")

        # Create a dictionary for distribution parameters
        distribution_params = {"theory": self.distribution}

        if self.distribution == "normal":
            # if len(par_dist)>0:
            if len(par_dist) == 2:
                mean = par_dist[0]
                std = par_dist[1]
            else:
                mean = np.mean(self.data)
                std = np.std(self.data)
                # Update distribution parameters for uniform distribution
                distribution_params.update(
                    {
                        "empirical": variable_analyzer.normal_discretization(
                            self.bins, mean, std
                        ),
                        "mean": mean,
                        "std": std,
                    }
                )

            # discrete_distrib = variable_analyzer.normal_discretization(bins, mean, std)
            # Create an instance of DiscretisationParams
            discretisation_params = DiscretisationParams(self.data, distribution_params)
            intervals_frequencies = variable_analyzer.discretisation_intervals(
                discretisation_params
            )

            if sum(intervals_frequencies["exp_freq"] == 0) != 0:
                logger.error(
                    "Number of intervals is to large to get acceptable expected values"
                )

            chi = stats.chisquare(
                intervals_frequencies["obs_freq"], intervals_frequencies["exp_freq"]
            )

        elif self.distribution == "uniform":
            # if len(par_dist)>0:
            if len(par_dist) == 2:
                min_value = par_dist[0]
                max_value = par_dist[1]
            else:
                min_value = np.min(self.data)
                max_value = np.max(self.data)
                # Update distribution parameters for uniform distribution
                distribution_params.update(
                    {
                        "empirical": variable_analyzer.uniform_discretization(
                            self.bins, min_value, max_value
                        ),
                        "mean": min_value,
                        "std": max_value,
                    }
                )

            # discrete_distrib = variable_analyzer.uniform_discretization(bins, min_value, max_value)
            # Create an instance of DiscretisationParams
            discretisation_params = DiscretisationParams(self.data, distribution_params)
            intervals_frequencies = variable_analyzer.discretisation_intervals(
                discretisation_params
            )

            if sum(intervals_frequencies["exp_freq"] == 0) != 0:
                logger.error(
                    "Number of intervals is to large to get acceptable expected values"
                )

            chi = stats.chisquare(
                intervals_frequencies["obs_freq"], intervals_frequencies["exp_freq"]
            )

        if chi.pvalue < 0.05:
            logger.info(
                "pvalue = %s < 0.05: Data is not following the %s distribution",
                chi.pvalue,
                self.distribution,
            )
        else:
            logger.info(
                "pvalue = %s >= 0.05: Data is following the %s distribution",
                chi.pvalue,
                self.distribution,
            )
        return float(chi.pvalue), intervals_frequencies

    def kolmogorov(self, *par_dist: float) -> float:
        """
        Calculation of the Kolmogorov-Smirnov test for every distribution.

        Args:
            *par_dist: arbitrary positional arguments, should be numeric

        Returns:
            p-value (float): KS test p-value
        """
        # if data.dtypes in {'O', 'bool'}:
        if any(isinstance(value, (str, bool)) for value in self.data):
            logger.error("Categorical or boolean variables are not treated yet.")
            return float("nan")

        if self.distribution == "normal":
            if len(par_dist) != 2:
                logger.error("Error: Provide mean and std for normal distribution.")
                return float("nan")

            mean, std = par_dist

        elif self.distribution == "uniform":
            if len(par_dist) != 2:
                logger.error("Error: Provide min and max for uniform distribution.")
                return float("nan")

            mean, std = par_dist
        else:
            logger.error("Unsupported distribution %s ", self.distribution)
            return float("nan")

        k = (
            stats.kstest(self.data, stats.norm.cdf, args=(mean, std))
            if self.distribution == "normal"
            else stats.kstest(self.data, stats.uniform.cdf, args=(mean, mean + std))
        )

        logger.info(k)

        if k.pvalue < 0.05:
            logger.info(
                "p-value = %s < 0.05 : The data is not followingthe %s distribution",
                k.pvalue,
                self.distribution,
            )
        else:
            logger.info(
                "p-value = %s >= 0.05 : The data is not followingthe %s distribution",
                k.pvalue,
                self.distribution,
            )

        return float(k.pvalue)

    def shannon_entropy(self) -> float:
        """
        Calculation of Shannon entropy.

        Args: None

        Returns:
            Shannon entropy (float):
        """

        if self.distribution == "uniform":
            min_value, max_value = np.min(self.data), np.max(self.data)
            discrete_distrib = variable_analyzer.uniform_discretization(
                self.bins, min_value, max_value
            )
            # Create a dictionary for distribution parameters
            distribution_params = {
                "theory": self.distribution,
                "empirical": discrete_distrib,
                "mean": min_value,
                "std": max_value,
            }
            discretisation_params = DiscretisationParams(self.data, distribution_params)
            intervals = variable_analyzer.discretisation_intervals(
                discretisation_params
            )

        if self.distribution == "normal":
            mean, std = np.mean(self.data), np.std(self.data)
            discrete_distrib = variable_analyzer.normal_discretization(
                self.bins, mean, std
            )
            # Create a dictionary for distribution parameters
            distribution_params = {
                "theory": self.distribution,
                "empirical": discrete_distrib,
                "mean": mean,
                "std": std,
            }
            discretisation_params = DiscretisationParams(self.data, distribution_params)
            intervals = variable_analyzer.discretisation_intervals(
                discretisation_params
            )

        if intervals["exp_freq"].sum() == 0:
            logger.info("Leading division by zero")

        prob_exp = intervals["exp_freq"] / intervals["exp_freq"].sum()
        return float(stats.entropy(prob_exp))

    def grte(self, *args: float) -> Tuple[float, Any]:
        """
        Calculates the Granular Relative and Theoretical Entropy (GRTE) for given data.

        Args:
            *args (float): Optional arguments. For 'uniform', provide start
                and end; for 'normal', provide mean and std.

        Returns:
            grte_res (float): The calculated GRTE value.
            intervals_discretized (pd.Series): The intervals discretized data.
        """
        # Create a dictionary for distribution parameters
        distribution_params = {"theory": self.distribution}

        # Check the specified distribution type and process accordingly
        if self.distribution == "uniform":
            min_value, max_value = (
                (args[0], args[1])
                if len(args) == 2
                else (np.min(self.data), np.max(self.data))
            )
            logger.info("debut %s", min_value)
            logger.info("la fin %s", max_value)

            # Update distribution parameters for uniform distribution
            distribution_params.update(
                {
                    "empirical": variable_analyzer.uniform_discretization(
                        self.bins, min_value, max_value
                    ),
                    "mean": min_value,
                    "std": max_value,
                }
            )

        elif self.distribution == "normal":
            mean, std = (
                (args[0], args[1])
                if len(args) == 2
                else (np.mean(self.data), np.std(self.data))
            )

            # Update distribution parameters for normal distribution
            distribution_params.update(
                {
                    "empirical": variable_analyzer.normal_discretization(
                        self.bins, mean, std
                    ),
                    "mean": mean,
                    "std": std,
                }
            )

        else:
            logger.error("Expecting only uniform or normal distribution")
            return None, None

        # Create an instance of DiscretisationParams
        discretisation_params = DiscretisationParams(self.data, distribution_params)

        # Calculate the intervals for the discretized data
        intervals_discretized = variable_analyzer.discretisation_intervals(
            discretisation_params
        )

        # Compute GRTE using the entropy of expected and observed frequencies
        grte_res = np.exp(
            -2
            * abs(
                stats.entropy(intervals_discretized["exp_freq"])
                - stats.entropy(intervals_discretized["obs_freq"])
            )
        )

        # Return the GRTE result and the discretized intervals
        return float(grte_res), intervals_discretized
