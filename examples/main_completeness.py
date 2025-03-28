"""
Main Module for Data Completeness Evaluation

This script utilizes the DataCompleteness class from the metric module to assess
the completeness of data in a DataFrame. It calculates and prints the overall
completeness score of the data as well as the completeness score for a specific
column in the DataFrame. This script is particularly useful for preliminary data
analysis in data cleaning and preprocessing stages.

Usage:
    Run this script directly after ensuring the metric.py module is available
    in the same directory or Python path. The script reads a specified CSV file
    and evaluates its data completeness.
"""

import pandas as pd
from dqm.completeness.metric import DataCompleteness


def main():
    """
    Main function to execute data completeness evaluation.

    This function creates an instance of DataCompleteness, loads a CSV file into
    a pandas DataFrame, calculates the overall completeness score for the DataFrame,
    and also calculates the completeness score for a specified column.
    The results are then printed to the console.
    """
    # Create an instance of the class
    completeness_evaluator = DataCompleteness()

    # Load your data into a pandas DataFrame
    # df = pd.read_csv('path_to_your_data')
    df = pd.read_csv(
        "datasets/challenge_welding.csv", sep=","
    )

    # Calculate the overall completeness score for the DataFrame
    overall_score = completeness_evaluator.completeness_tabular(df)

    # Calculate the completeness score for a single column
    column_score = completeness_evaluator.data_completion(df["welding-seams"])

    # Print the results
    print(f"Overall Data Completeness Score: {overall_score}")
    print(f"Completeness Score for Column: {column_score}")


if __name__ == "__main__":
    main()
