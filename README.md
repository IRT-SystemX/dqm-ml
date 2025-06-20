<div align="center">
    <img src="_static/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">dqm-ml</h1>
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.10-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.11-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.12-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MPL-2">
	</a>
    <a href="_static/pylint/pylint.txt">
        <img src="_static/pylint/pylint.svg" alt="Pylint Score">
    </a>
    <a href="_static/flake8/index.html">
        <img src="_static/flake8/flake8.svg" alt="Flake8 Report">
    </a>
	<a href="_static/coverage/index.html">
        <img src="_static/coverage/coverage.svg" alt="Coverage report">
    </a>

</div>

# Data Quality Metrics

The current version of the Data Quality Metrics (called **dqm-ml**) computes three data inherent metrics and one data-model dependent metric.

The data inherent metrics are 
- **Diversity** : Computes the presence in the dataset of all required information defined in the specification (requirements, Operational Design Domain (ODD) . . . ).
- **Representativeness** : is defined as the conformity of the distribution of the key characteristics of the dataset according to a specification (requirements, ODD.. . )
- **Completeness** : is defined by the degree to which subject data associated with an entity has values for all expected attributes and related entity instances in a specific context of use.

The data-model dependent metrics are: 
- **Domain Gap** : In the context of a computer vision task, the Domain Gap (DG) refers to the difference in semantic, textures and shapes between two distributions of images and it can lead to poor performances when a model is trained on a given distribution and then is applied to another one.

(Definitions from [Confiance.ai program](https://www.confiance.ai/))

[//]: # (- Coverage : The coverage of a couple "Dataset + ML Model" is the ability of the execution of the ML Model on this dataset to generate elements that match the expected space.)

For each metric, several approaches are developped to handle the maximum of data types. For more technical and scientific details, please refer to this [deliverable](https://catalog.confiance.ai/records/p46p6-1wt83/files/Scientific_Contribution_For_Data_quality_assessment_metrics_for_Machine_learning_process-v2.pdf?download=1)

## Project description
Several approches are developped as described in the figure below.

<img src="_static/library_view.png" width="1024"/>

In the current version, the available metrics are: 
- Representativeness:
  - $\chi^2$ Goodness of fit test for Uniform and Normal Distributions 
  - Kolmogorov Smirnov test for Uniform and Normal Distributions
  - Granular and Relative Theorithecal Entropy GRTE proposed and developed in the Confiance.ai Research Program
- Diversity: 
  - Relative Diversity developed and implemented in Confiance.ai Research Program
  - Gini-Simpson and Simposon indices 
- Completeness: 
  - Ratio of filled information
- Domain Gap: 
  - MMD 
  - CMD 
  - Wasserstein 
  - H-Divergence
  - FID
  - Kullback-Leiblur MultiVariate Normal Distribution

[//]: # (- Coverage : )

[//]: # (  - Approches developed in Neural Coverage &#40;NCL&#41; given [here]&#40;https://github.com/Yuanyuan-Yuan/NeuraL-Coverage&#41;. )

# Getting started 

## Set up a clean virtual environnement

Linux setting:

```
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
```

Windows setting:

```
pip install virtual env 
virtualenv myenv 
.\myenv\Scripts\activate
```

## Install the library
You can install it by directly downloading from PyPi using the command:

````
pip install dqm-ml
````

Or you can installing it from the source code by launching the following command:

```
pip install .
```

## Usage

[//]: # (All validated and verified functions are detailed in the files **call_main.py**. )

Each metric is used by importing the corresponding modules and class into your code.
For more information about each metric, refer to the specific README.md in ```dqm/<metric_name>``` subfolders

## Available examples

Many examples of DQM-ML applications are avalaible in the folder ```/examples```

You will find :

2 jupyter_notebooks:

- **multiple_metrics_tests.ipynb** : A notebook applying completeness, diversity and representativeness metrics on an example dataset.
- **domain_gap.ipynb** : A notebook demonstrating an example of applying domain_gap metrics to a generated synthetic dataset.

4 python scripts:

Those scripts named **main_X.py** gives an example of computation of approaches implemented for metrics <X> on samples.

The ```main_domain_gap.py``` script must be called with a config file passed as an argument using ```--cfg```.

For example:

``` python examples/main_domain_gap.py --cfg examples/domain_gap_cfg/cmd/cmd.json``` 

We provide in the folder ```/examples/domain_gap_cfg``` a set of config files for each domain_gap approaches`:

For some domain_gap examples, the **200_bird_dataset** will be required. It can be downloaded from this [link](http://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models/200-birds-species.zip). The zip archive will be extracted into the ```examples/datasets/``` folder.

## References



```
@inproceedings{chaouche2024dqm,
  title={DQM: Data Quality Metrics for AI components in the industry},
  author={Chaouche, Sabrina and Randon, Yoann and Adjed, Faouzi and Boudjani, Nadira and Khedher, Mohamed Ibn},
  booktitle={Proceedings of the AAAI Symposium Series},
  volume={4},
  number={1},
  pages={24--31},
  year={2024}
}
```

[HAL link](https://hal.science/hal-04719346v1)
