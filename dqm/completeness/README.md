# Data Completeness Evaluation Description

## Overview
This metric provides a set of tools for assessing the completeness of tabular data. 
It is particularly useful during the data preprocessing and cleaning stages of a data analysis workflow.

## Components
### metric.py (Module)
#### Description
This module includes the DataCompleteness class, which contains methods to calculate 
completeness scores for dataframes and individual columns. These methods are instrumental 
in identifying columns with missing data and quantifying the extent of missingness.

#### Authors
Faouzi ADJED

Anani DJATO

#### Class: DataCompleteness
##### Methods
- completeness_tabular(data: pd.DataFrame) -> float: 
Calculates the average completeness score for a dataframe.
- data_completion(data: pd.Series) -> float: 
Calculates the completeness score for an individual data column.

## Example

	from dqm.completeness.metric import DataCompleteness
	import pandas as pd

	def main():
    	    completeness_evaluator = DataCompleteness()
            df = pd.read_csv('your_data.csv', sep=";")
            overall_score = completeness_evaluator.completeness_tabular(df)
            column_score = completeness_evaluator.data_completion(df['your_column'])
            print(f'Overall Data Completeness Score: {overall_score}')
            print(f'Completeness Score for Column: {column_score}')

        if __name__ == "__main__":
    	    main()

