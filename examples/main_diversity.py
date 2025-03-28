"""
Module for Analyzing Data Diversity

This module provides a class, DataDiversityAnalyzer, for calculating and analyzing
diversity indices in datasets.
It Utilizes the DiversityCalculator and DiversityIndexCalculator classes to
compute various diversity metrics.
The module is designed to provide insights into the diversity of features
within a dataset, using statistical and analytical methods.

Author:
    Faouzi ADJED
    Anani DJATO

Classes:
    DataDiversityAnalyzer: A class for analyzing diversity in datasets.

Dependencies:
    pandas: For data manipulation and analysis.
    diversity (DiversityCalculator): For calculating diversity scores.
    metric (DiversityIndexCalculator): For calculating statistical diversity indices.
    twe_logger: For logging information during the analysis process.

Example:
    To use this module, instantiate the DataDiversityAnalyzer class and call its
    calculate_diversity_scores method with a dataset and a specific feature.
    Example usage:
    analyzer = DataDiversityAnalyzer()
    data = pd.read_csv("path_to_dataset.csv")
    feature = "example_feature"
    analyzer.calculate_diversity_scores(data, feature)
"""

from dqm.diversity.diversity import DiversityCalculator
from dqm.diversity.metric import DiversityIndexCalculator
from dqm.diversity.twe_logger import get_logger
import pandas as pd


class DataDiversityAnalyzer:
    """
    A class for analyzing the diversity of datasets.

    This class integrates with DiversityCalculator and DiversityIndexCalculator
    to compute and analyze various diversity indices within a dataset. It provides
    methods to calculate and report diversity scores for specific features, aiding
    in understanding the distribution and representation in datasets.

    Methods:
        calculate_diversity_scores(data, feature): Calculates and logs the
            diversity scores for a given feature.
        none_function(): A placeholder method that currently does nothing.
    """

    def __init__(self):
        """
        Initializes the DataDiversityAnalyzer with necessary components.

        Attributes:
            metric_calculator (DiversityIndexCalculator): Calculator for diversity indices.
            calculator (DiversityCalculator): Calculator for general diversity metrics.
            logger (Logger): Logger for logging information during the analysis.
        """
        self.metric_calculator = DiversityIndexCalculator()
        self.calculator = DiversityCalculator()
        self.logger = get_logger()

    def calculate_diversity_scores(self, data: pd.Series, feature: str):
        """
        Calculate diversity scores for a given feature in the dataset.

        Args:
            data (pandas.DataFrame): The dataset containing the feature.
            feature (str): The feature for which diversity scores are calculated.

        Returns: None
        """
        # self.logger.info("The first 5 lines of the data:")
        # self.logger.info(data.head())
        # self.logger.info("Data shape:")
        # self.logger.info(data.shape)
        # self.logger.info("Columns:")
        # self.logger.info(data.columns)
        # self.logger.info("Columns types:")
        # self.logger.info(data.dtypes)

        data = data.dropna()

        s = self.metric_calculator.simpson(data[feature])
        self.logger.info("Simpson index: %s", s)

        g = self.metric_calculator.gini(data[feature])
        self.logger.info("Gini index: %s", g)

        diversity_score = self.calculator.compute_diversity(
            data[feature], "lexical", "richness"
        )
        self.logger.info("Diversity score: %s", diversity_score)

    def none_function(self) -> None:
        """
        A placeholder function that currently performs no operation.

        Returns: None
        """
        self.logger.info("Do not something")


# Use of DataDiversityAnalyzer Class
if __name__ == "__main__":
    analyzer = DataDiversityAnalyzer()
    df = pd.read_csv(
        "datasets/features.csv", sep=","
    )
    # feature_name = "Traffic_light_green"
    # analyzer.calculate_diversity_scores(df, feature_name)

    for i in df:
        if df.dtypes[i] not in ("object", "bool"):
            print(f"For { i } feature :")
            analyzer.calculate_diversity_scores(df, i)
            print("---------------------------------------------------------")
