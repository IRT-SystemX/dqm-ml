"""
Data Completeness Evaluation Module

This module provides tools to assess the completeness of tabular data. It is
especially useful in data preprocessing and cleaning stages of a data analysis
workflow. The module includes a class, DataCompleteness, with methods to
calculate completeness scores for dataframes and individual columns.
These methods help in identifying columns with missing data and quantifying
the extent of missingness.

Authors:
    Faouzi ADJED
    Anani DJATO

Classes:
    DataCompleteness: A class that encapsulates the methods for evaluating data completeness.

Methods:
    completeness_tabular: Calculates the average completeness score for a dataframe.
    data_completion: Calculates the completeness score for an individual data column.

Dependencies:
    numpy
    pandas
    matplotlib
    scipy
    seaborn
    warnings

Usage:
    The DataCompleteness class can be used as follows:

    from data_completeness import DataCompleteness

    # Create an instance of the class
    completeness_evaluator = DataCompleteness()

    # Load your data into a pandas DataFrame
    df = pd.read_csv('your_data_path.csv')

    # Calculate the overall completeness score for the DataFrame
    overall_score = completeness_evaluator.completeness_tabular(df)

    # Calculate the completeness score for a single column
    column_score = completeness_evaluator.data_completion(df['your_column'])

    # Print the results
    print(f'Overall Data Completeness Score: {overall_score}')
    print(f'Completeness Score for Column: {column_score}')
"""

import pandas as pd


class DataCompleteness:
    """
    This class provides methods to evaluate the completeness of tabular data.

    It includes methods to calculate completeness scores for individual columns and
    for entire dataframes by assessing the presence of non-null data.

    Methods:
        completeness_tabular: Calculate the average completeness score of a dataframe.
        data_completion: Calculate the completeness score of a single data column.
    """

    def completeness_tabular(self, data: pd.DataFrame) -> float:
        """
        Calculate the average completeness score of the entire dataframe.

        Args:
            data (pd.DataFrame): The dataframe to be evaluated for completeness.

        Returns:
            score_total(float): The average completeness score of
                all columns in the dataframe.
        """
        score_total = 0
        for column in data.columns:
            score_total += self.data_completion(data[column])
        score_total = score_total / len(data.columns)
        return score_total

    def data_completion(self, data: pd.Series) -> float:
        """
        Calculate the completeness score of a single data column.

        Args:
            data (pd.Series): The data column to be evaluated for completeness.

        Returns:
            completeness_score(float): The completeness score of the column,
                calculated as the ratio of non-null entries to total entries.
        """
        processed_data = data.dropna()
        if len(data) == len(processed_data):
            completeness_score = 1
        else:
            completeness_score = len(processed_data) / len(data)
        return completeness_score
