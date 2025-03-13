# Using the Diversity Analysis Python Scripts
## Description:
This set of Python scripts is designed to calculate and analyze various diversity indices in datasets. 
It includes modules for diversity calculation (diversity.py), metric calculation (metric.py), and a main script (main.py) that demonstrates their usage.


Diversity is a collection of Python scripts designed to calculate and analyze various diversity indices in datasets. 
This collection consists of three main components: diversity.py, metric.py, and main.py. 
These scripts provide a comprehensive approach to understanding the diversity within both lexical and visual datasets.

## Components
### Diversity Calculator (diversity.py)
Provides a class DiversityCalculator for calculating different types of diversity (lexical and visual) in datasets.

### Metric Calculator (metric.py)
Offers additional metrics, likely including statistical indices like Simpson Index and Gini-Simpson Index for deeper data analysis.

### Main Script (main.py)
Demonstrates the usage of the DiversityCalculator and DiversityIndexCalculator classes for real-world data application.

The script demonstrates the usage of classes and from diversity and metric scripts: 
The script showcases the capabilities of the modules by creating instances of classes and invoking their methods.

The main script serves as a central point to showcase and test the functionality provided by these scripts. 
It creates instances of classes, performs operations on sample data, and logs the results using a logger. 

## Getting Started
To utilize scripts, follow these steps:

### Setup
Install any required dependencies, such as numpy and pandas.

### Using Diversity Calculator
Import the DiversityCalculator from diversity.py.
Initialize the calculator and use the compute_diversity method with your data.
Specify the type of diversity and the aspect you are interested in ('lexical' and 'richness').

### Using Metric Calculator
This step depends on the functionality provided in metric.py. 
Generally, import the relevant class and use its methods for additional metrics.

### Executing the Main Script
Run main.py to see other (diversity.py and metric.py) scripts in action.
This script will utilize the aforementioned classes to calculate diversity scores for provided sample datasets.
