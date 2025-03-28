# Description of the use of the Representativeness metrics scripts

This project provides a set of Python scripts for analyzing data representativeness from various perspectives.
The scripts focus on variable analysis, data distribution, and the identification of Minimal Unstable Patterns (MUPs) in a dataset.

The Representativeness library primarily consists of three Python scripts and a main script that orchestrates the functionality provided by the others. 
The three scripts are named utils.py, metric.py, and mup.py. The main script, logically named main.py, coordinates the execution of the features 
offered by these scripts. Below a description of each of these scripts:

# Scripts

## Description of dqm.representativeness.utils

This module provides tools for variable analysis, visualization, and data preparation for statistical tests such as chi-square tests.
It offers a convenient interface for discretizing variables based on normal or uniform distributions and features functionality to 
generate histograms for observed and expected values.

It implements two main classes, "DiscretisationParams" and "VariableAnalysis," along with several associated functions:

### Description of DiscretisationParams class
This class is used to define discretization parameters. It takes input data (data) and distribution parameters 
("distribution_params"), including distribution theory ('normal' or 'uniform'), empirical distribution, mean (mean), and standard deviation (std). 
It provides methods to convert parameters into a dictionary ("to_dict") and retrieve input data ("get_data").

### Description of VariableAnalysis class
This class provides functionality for variable analysis, visualization, and discretization. It includes methods such as 
"variable_counting" to count unique values, "countplot" to visualize category frequencies, and "discretization" to discretize a variable. 
Methods like "data_processing_for_chisqure_test" and "delete_na" are intended to process data for statistical tests, handling missing values, 
for example. The "expected_hist" and "observed_hist" methods generate histograms for expected and observed values, respectively. 
The class also contains utility functions such as "uniform_discretization" which discretizes a variable with a uniform distribution, 
and the "discretisation_intervals" function discretizes a set of data into intervals based on empirical distribution and calculates 
observed and expected frequencies for each interval.

### Example of utilization dqm.representativeness.utils module

	# Importer les classes du script utils.py
	from utils import VariableAnalysis, DiscretisationParams

	# Exemple d'utilisation de la classe VariableAnalysis
	variable_analyzer = VariableAnalysis()

	# Exemple d'utilisation de la méthode variable_counting
	my_variable = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
	counts = variable_analyzer.variable_counting(my_variable)
	print("Counts of unique values:")
	print(counts)

	# Exemple d'utilisation de la méthode countplot (qui affiche un graphique, donc plt.show() est nécessaire)
	variable_analyzer.countplot(my_variable)
	plt.show()

	# Instancier la classe DiscretisationParams
	discretisation_params = DiscretisationParams(
    		data=my_variable,
    		distribution_params={
        		'theory': 'normal',
        		'emprical': [-1.0, 0.0, 1.0, 2.0],
        		'mean': 0.0,
        		'std': 1.0
    		}
	)

	# Exemple d'utilisation de la méthode discretisation_intervals
	interval_data = variable_analyzer.discretisation_intervals(discretisation_params)
	if interval_data is not None:
    	    print("Discretization Intervals:")
    	    print(interval_data)


## Description of the module dqm.representativeness.metric

The script is designed for analyzing the distribution of data and includes error handling for categorical or boolean variables. 
It also logs relevant information using the dqm.representativeness.twe_logger module.

The script provides a DistributionAnalyzer class with methods for analyzing data distribution using various statistical tests and measures. 
There are some key functionalities of the class like as follow :

### chisquare_test 
This method performs a chi-square test for goodness of fit on the provided data. 
It supports both normal and uniform distributions. 
The result includes the chi-square test p-value and intervals frequencies

### gof
It calculates the goodness of fit using the Kolmogorov-Smirnov test
The result is the goodness of fit (KS) p-value.

### kolmogorov
The method calculates the Kolmogorov-Smirnov test for a chosen distribution. It returns the KS test p-value

### shannon_entropy
It calculates Shannon entropy for the provided intervals

### confidence_interval
The method calculates the confidence interval for the provided data

### Example of utilization of metric.py script

	# Import the Necessary modules or class
	from metric import DistributionAnalyzer
	from twe_logger import get_logger

	analyzer = DistributionAnalyzer()
	logger = get_logger()

	# Use the Provided Methods:
	## Chi-Square Test Example
	my_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
	bins = 10
	distribution = 'normal'
	mean = 0
	std = 1

	result = analyzer.chisquare_test(my_data, bins, distribution, mean, std)
	if result is not None:
    	    p_value, intervals_frequencies = result
            logger.info("Chi-Square Test P-Value : %s", p_value)
    	    loggerinfo("Intervals Frequencies: %s", intervals_frequencies)

	# Goodness-of-Fit Test Example
	goodness_of_fit = analyzer.gof('normal', intervals_frequencies)
	if goodness_of_fit is not None:
    	    logger.info("Goodness of Fit (KS) P-Value: %s", goodness_of_fit)


	# Kolmogorov-Smirnov Test Example
	ks_p_value = analyzer.kolmogorov(my_data, 'normal', mean, std)
	if ks_p_value is not None:
    	    logger.info("Kolmogorov-Smirnov Test P-Value: %s", ks_p_value)


	# Shannon Entropy Example
	entropy = analyzer.shannon_entropy(intervals_frequencies)
	logger.info("Shannon Entropy: %s", entropy)

	# Confidence Interval Example
	confidence_interval, mean = analyzer.confidence_interval(my_data)
	if confidence_interval is not None:
    	    logger.info("Confidence Interval: %s", confidence_interval)
            logger.info("Mean: %s", mean)


## Example

You will find [here](https://github.com/IRT-SystemX/DataQualityMetrics/blob/main/examples/main_representativeness.py)
an example of script that demonstrates the usage of classes and functions from two ``dqm.representativeness `` modules.  The main() function showcases the capabilities of the modules by creating 
instances of classes and invoking their methods.

The main() function serves as a central point to showcase and test the functionality provided by these scripts. 
It creates instances of the relevant classes, performs operations on sample data, and logs the results using a logger. 


# Usage
1. Ensure you have installed dqm 

2. To perform a comprehensive analysis of variables, and distribution on a sample dataset, run the script (bash):
        ``python main_representativeness.py``
