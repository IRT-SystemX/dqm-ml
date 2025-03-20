# Data_Quality_Metrics

Data Quality Metrics called **DQM** is a python library wich computes Three data inherent metrics and two data-model dependent metrics.

The data inherent metrics are (defintions from Confiance.ai program): 
- Diverity : Computes the presence in the dataset of all required information defined in the specification (requirements, Operational Design Domain (ODD) . . . ).
- Representativeness : is defined as the conformity of the distribution of the key characteristics of the dataset according to a specification (requirements, ODD.. . )
- Completeness : is defiend by the degree to which subject data associated with an entity has values for all expected attributes and related entity instances in a specific context of use.

The data-model dependent metrics are (definition from Confiance. ai program): 
- Domain Gap : In the context of a computer vision task, the Domain Gap (DG) refers to the difference in semantic, textures and shapes between two distributions of images and it can lead to poor performances when a model is trained on a given distribution and then is applied to another one.
- Coverage : The coverage of a couple "Dataset + ML Model" is the ability of the execution of the ML Model on this dataset to generate elements that match the expected space.

For Each metric, several approaches are developped to handle the maximum of data types. For more technical and scientific details, you can refer back to this [delivrable](https://irtsystemx.sharepoint.com/:b:/r/sites/IAdeConfiance833/Documents%20partages/General/Livrables%20Documentaires/Final_Structure/04%20-%20Scientific_Contributions/Scientific%20Contribution%20to%20Counterfactual-based%20Metrics%20for%20the%20evaluation%20of%20image%20classifier.pdf?csf=1&web=1&e=PlePjs)

## Project description
Several approches are developped as decscribed in the [figure below](docs/www/library_view.PNG)

<img src="docs/www/library_view.PNG" width="1024"/>

The approaches developped are: 
- Representativeness : 
  - $\chi^2$ goodness of fit test for uniform and normal distibutions. 
  - Kolmogrov Smirnov test for uniform and normal distributions. 
  - Granular and Relative Theorithecal Entropy GRTE proposed and developed in Confiance.ai program. 
- Diverity : 
  - Relative Diverity developed and implemented in Confiance.ai program
  - Gini-Simpson and Simposon indices. 
- Completeness : 
  - Ratio of filled inofrmation
- Domain Gap : 
  - MMD 
  - CMD 
  - Wasserstein 
  - H-Divergence
  - FID
  - Kullback-Leiblur MultiVariate Normal distribution
- Coverage : 
  - Approches developed in Neural Coverage (NCL) given [here](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage). 

## Installation 
The installation of the library can be done fom Gitlab repository and complile the package or form Harbor by uploading the complied library. 

### package building from Gitlab 
- Clone the repositeory : 
  - <code>git clone https://git.irt-systemx.fr/confianceai/ec_5/ec5_as23/data_quality_metrics </code>
- Move to the right folder : 
  - <code>cd data_quality_metrics</code>
- Local building 
  - <code> python -m pip install -e ./ </code>
  - after execution the commande line above, a python package will be built under *dqm* name. Then , an import of the package will be possible. 
- dist building 
  - to be completed

## Usage
All validated and verified functions are detailed in the files **call_main.py**. 

### implented tests:
- All tests are implented by using VDP use case in the file *final-tests.ipynb* 
- For Completeness : 
  - import the metric <code> from dqm.completeness.metric import DataCompleteness </code>
  - call completeenss metric <code> completeness_evaluator = DataCompleteness() </code>
  - For the whole dataset (dataframe : df): 
    - <code> completeness_evaluator.completeness_tabular(df) </code>
  - for a specific column : 
    - <code> completeness_evaluator.data_completion(df['Car']) </code>
- For Representativness : 
  - import the metric <code> from dqm.representativeness.metric import DistributionAnalyzer </code>
  - For normal distribution : 
    - <code>analyzer = DistributionAnalyzer(df['Car'], 20, 'normal') </code>
    - $\chi^2$ test : 
      - <code> pvalue, intervals_frequencies = analyzer.chisquare_test() </code>, where the *pvalue* designe the probability value to accepte the null hypothesis and intervals_frenquecies details the theoretical and observed frenquecies in each bin. 
    - KS test : 
      - <code>ks_pvalue = analyzer.kolmogorov()</code>, where *ks_pvalue* is the pvlaue of the test. 
    - GRTE method : 
      - <code> grte_result, intervals_discretized = analyzer.grte() </code>
  
  These methods can include required normal paramters (mean and standard deviation)
  - The same code can be used for **uniform** distibution.  
- For diversity : 
  - import metric : <code>from dqm.diversity.metric import DiversityIndexCalculator </code>
  - call the function : <code>metric_calculator = DiversityIndexCalculator() </code>
  - Diversity scores : 
    - Simpson index : <code> metric_calculator.simpson(data['Car']) </code>
    - Gini-Simpson index : <code>metric_calculator.gini(data['Car']) </code>
  - Relative diversity : 
    - <code> ... </code>

## Domain Gap Metrics
Domain gap metrics are used to quantify the differences between datasets that originate from different domains.  
These metrics play a crucial role in various applications where understanding and managing domain shifts are essential.  
 - Domain Adaptation: aims to transfer knowledge learned from a source domain to a target domain
 - Transfer Learning: aims to evaluate the knowledge shift between 2 domains 
 - Dataset Selection: aims to extract a subsamples of a dataset with more representative data
 - Dataset Augmentation: aims to improve a dataset representativity by adding new coherent datas
 - Bias analysis: aims to detect bias in datasets

 ### Practice

 Each metrics must compute with a configuration file, this configuration file contains informations about data location and pre-processing, model used for feature extraction, and metric name. A collection of preset config file are available in cfg folder. You will find bellow steps to perform the computing of metrics using terminal command and notebook utilization.

 #### Terminal Computation (example with Proxy as Distance)
 - Move to the right folder : <code>cd dqm/domain_gap</code>
 - Setup a config file : reference config file are available for each metric in <code> ./cfg/proxy </code>
 - Run main : <code> python main.py --cfg path/to/config.json </code>

#### Notebook Computation (example with Proxy as Distance)
 - Import metric : <code> from dqm.domain_gap.utils import ProxyAsDistance </code>
 - Instanciate metric : <code> pad = ProxyAsDistance() </code>
 - Compute metric : <code> pad.compute_image_distance </code>

## Functions
*add description for each function developed* 

- Wasserstein (https://arxiv.org/abs/2201.02824) : distance defined in optimal transport theory. The basic idea is to find the most efficient way to "move" one distribution to match another, where efficiency is measured in terms of both the amount of "mass" moved and the distance over which it is moved. In our implementation, we consider 2 approachs :
    - 1-dimension : we compute the Wasserstein distance for each feature and averages the distances over all features
    - 2-dimension : we compute the 2D Wasserstein distance between features extracted from two sets of images, using a method that involves computing a covariance matrix and projecting the features onto its eigenvectors

- Proxy A Distance (http://arxiv.org/pdf/1412.4446) : theoretical concept that measures the ability of a hypothesis class to distinguish between two domains, PAD is an approximation of the H-divergence that uses the performance of a classifier to estimate the divergence.
