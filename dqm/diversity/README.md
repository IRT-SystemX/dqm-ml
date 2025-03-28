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

## Getting Started

### Using Diversity Calculator
Import the DiversityCalculator : ```from dqm.diversity.diversity import DiversityCalculator```
Initialize the calculator and use the compute_diversity method with your data.
Specify the type of diversity and the aspect you are interested in ('lexical' and 'richness').

### Using Metric Calculator
This step depends on the functionality provided in from ```dqm.diversity.metric```
Generally, import the relevant class and use its methods for additional metrics.

### Example 
See the example [here](
https://github.com/IRT-SystemX/DataQualityMetrics/blob/develop/examples/main_diversity.py) to see other (diversity.py and metric.py) scripts in action available .
This script will utilize the aforementioned classes to calculate diversity scores for provided sample datasets.
