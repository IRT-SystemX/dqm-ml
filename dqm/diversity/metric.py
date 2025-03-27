"""
Diversity Index Calculator

This module defines the DiversityIndexCalculator class, which
offers methods to calculate various diversity indices for categorical data.
These indices are useful in statistical analysis and data science to
understand the distribution and diversity of categorical data.

Authors:
    Faouzi ADJED
    Anani DJATO

Dependencies:
    pandas

Classes:
    DiversityIndexCalculator: Provides methods for calculating diversity indices in a dataset.

Functions: None

Usage:
    from metric import DiversityIndexCalculator
    calculator = DiversityIndexCalculator()
    dataset = pandas.Series([...])  # Replace with your data
    simpson_index = calculator.simpson(dataset)
    gini_index = calculator.gini(dataset)

These methods are useful for ecological, sociological, and
various other types of categorical data analysis.
"""

import pandas as pd


class DiversityIndexCalculator:
    """
    This class provides methods to calculate various diversity
    indices for a given dataset.

    Methods:
        num: Counts the number of each category in a dataset.
        simpson: Calculates the Simpson diversity index.
        prob: Calculates the frequencies of each category in a dataset.
        gini: Calculates the Gini-Simpson index.
    """

    def num(self, variable: pd.Series) -> pd.Series:
        """
        Calculate the number of each category of a variable.

        Args:
            variable (Series): The data series for which to count categories.

        Returns:
            n (Series): The count of each category.
        """
        n = variable.value_counts()
        return n

    def simpson(self, variable: pd.Series) -> float:
        """
        Calculate Simpson's index, which is a measure of diversity.

        Args:
            variable (Series): The data series for which to calculate the Simpson index.

        Returns:
            s (float): The Simpson diversity index.
        """
        n = self.num(variable)
        s = 1 - (sum(n * (n - 1)) / (len(variable) * (len(variable) - 1)))
        return s

    def prob(self, variable: pd.Series) -> pd.Series:
        """
        Calculate the frequencies of each category in a variable.

        Args:
            variable (Series): The data series for which to calculate frequencies.

        Returns:
            p (Series): The frequency of each category.
        """
        p = variable.value_counts() / len(variable)
        return p

    def gini(self, variable: pd.Series) -> float:
        """
        Compute the Gini-Simpson index, a metric for assessing diversity that
        takes into consideration both the quantity of distinct categories
        and the uniformity of their distribution.

        Args:
            variable (Series): The data series for which to calculate the Gini-Simpson index.

        Returns:
            g (float): The Gini-Simpson index.
        """
        p = self.prob(variable)
        g = 1 - (sum(p**2))
        return g

    def RD(self, variable: pd.Series) -> float:
        print(variable)
