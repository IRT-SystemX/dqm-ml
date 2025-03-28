"""
Script for Statistical Analysis of Traffic Light Data
This script performs statistical analysis on a dataset containing
information about traffic lights. It uses pandas and numpy for
data manipulation and the custom 'DistributionAnalyzer' class
from the 'metric' module for statistical tests.

The script reads a CSV dataset, calculates basic statistics
(mean, standard deviation), and performs various statistical
tests including the Chi-Square test, Kolmogorov-Smirnov test,
Shannon Entropy calculation, and the GRTE (Granular Relative and Theoretical Entropy)
analysis. These tests help in understanding the distribution
and entropy of the 'Traffic_light_unknown' variable in the dataset.

Author: 
    Faouzi Adjed
    Anani DJATO
"""
import pandas as pd
import numpy as np
from dqm.representativeness.metric import DistributionAnalyzer
#from metric import DistributionAnalyzer

# Dataset path
path = "datasets/features.csv"
data = pd.read_csv(path, sep=",")
#print("Data info:\n", data.info())
# print("The first 5 lines:\n", data.head())
var = data["contrast"]
mean = np.mean(var)
std = np.std(var)
# data = pd.Series(np.random.normal(0, 1, 1000))

# Parameters for analysis
bins = 10
distribution = 'normal'

# Instantiation of DistributionAnalyzer
analyzer = DistributionAnalyzer(var, bins, distribution)

# Using the method chisquare_test
pvalue, intervals_frequencies = analyzer.chisquare_test()
print(f"Chi-Square Test: p-value = {pvalue}")

# Using the method kolmogorov
ks_pvalue = analyzer.kolmogorov(mean, std)
print(f"Kolmogorov-Smirnov Test: p-value = {ks_pvalue}")

# Using the method shannon_entropy
entropy = analyzer.shannon_entropy()
print(f"Shannon Entropy: {entropy}")

# Using the method grte
grte_result, intervals_discretized = analyzer.grte()
print(f"GRTE: {grte_result}")
