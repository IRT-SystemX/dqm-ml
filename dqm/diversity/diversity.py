"""
This module, DiversityCalculator, calculates various types of diversity in datasets.
It focuses on both lexical and visual diversities, employing statistical indices for
different metrics such as richness, variety, color, and shape. Useful in linguistics,
image processing, and data analysis, it helps understand the diversity of elements
in a dataset.

Authors:
    Faouzi ADJED
    Anani DJATO

Dependencies:
    numpy
    collections.Counter

Classes:
    DiversityCalculator: A class that provides methods for calculating
      different types of diversity in datasets.

Functions: None

Usage:
    To use this module, create an instance of the DiversityCalculator class and call its
    compute_diversity method with appropriate arguments.
    Example:
    calculator = DiversityCalculator()
    diversity_score = calculator.compute_diversity(data, 'lexical', 'richness')
"""

from collections import Counter
from typing import Iterable
import numpy as np

from dqm.utils.twe_logger import get_logger

logger = get_logger()


class DiversityCalculator:
    """
    A class to compute various types of diversity within data.

    This class offers methods to calculate lexical and visual diversities in datasets using
    different statistical measures. It can measure lexical diversity in terms of richness and
    variety, and visual diversity in terms of color and shape using indices like Shannon,
    Simpson, and Gini-Simpson.

    Methods:
        compute_diversity: Calculates diversity based on specified type and need.
    """

    def compute_diversity(
        self, data: Iterable, diversity_type: str, need: str
    ) -> float:
        """
        Compute diversity of given data based on type and need.

        Args:
            data (Iterable): Dataset for diversity computation.
            diversity_type (str): Type of diversity ('lexical' or 'visual').
            need (str): Specific need for calculation ('richness', 'variety', 'color', 'shape')

        Returns:
            diversity (float): Calculated diversity value.
        """
        if diversity_type == "lexical" and need == "richness":
            # Compute lexical richness using Shannon Index
            _, counts = np.unique(data, return_counts=True)
            norm_counts = counts / counts.sum()
            diversity = -(norm_counts * np.log(norm_counts)).sum()
        elif diversity_type == "lexical" and need == "variety":
            # Compute lexical variety using Simpson Index
            _, counts = np.unique(data, return_counts=True)
            norm_counts = counts / counts.sum()
            diversity = 1 - (np.square(norm_counts)).sum()
        elif diversity_type == "visual" and need == "color":
            # Compute color diversity using Richness Index
            counter = Counter(data)
            freqs = np.array(list(counter.values())) / len(data)
            diversity = len(freqs)
        elif diversity_type == "visual" and need == "shape":
            # Compute shape diversity using Gini-Simpson Index
            counter = Counter(data)
            freqs = np.array(list(counter.values())) / len(data)
            diversity = 1 - np.sum(freqs**2)
        else:
            logger.error("Invalid diversity type or need.")

        return float(diversity)  # To be homogenous on output data type

    def validate_inputs(self, diversity_type: str, need: str) -> None:
        """
        This method is added just to have at least two public methods
        in a class as required by Python coding standards.

        This method validates the inputs for compute_diversity method.

        Args:
        diversity_type (str): Type of diversity to be computed.
        need (str): Specific need for diversity calculation.

        """
        valid_diversity_types = ["lexical", "visual"]
        valid_needs = ["richness", "variety", "color", "shape"]

        if diversity_type not in valid_diversity_types:
            logger.error(
                "Invalid diversity type : %s. Must be one of %s.",
                diversity_type,
                valid_diversity_types,
            )
        if need not in valid_needs:
            logger.error("Invalid need: %s. Must be one of %s.", need, valid_needs)
