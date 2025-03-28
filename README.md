<div align="center">
    <img src="_static/Logo_ConfianceAI.png" width="20%" alt="ConfianceAI Logo" />
    <h1 style="font-size: large; font-weight: bold;">DQM</h1>
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9-efefef">
    </a>
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

[![Code style: Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev)
[![Code style: flake8](https://img.shields.io/badge/code%20style-flake8-1c4a6c.svg)](https://flake8.pycqa.org/en/latest/)

</div>

# Data_Quality_Metrics

Data Quality Metrics called **DQM** is a python library which computes three data inherent metrics and one data-model dependent metrics.

The data inherent metrics are (defintions from Confiance.ai program): 
- **Diversity** : Computes the presence in the dataset of all required information defined in the specification (requirements, Operational Design Domain (ODD) . . . ).
- **Representativeness** : is defined as the conformity of the distribution of the key characteristics of the dataset according to a specification (requirements, ODD.. . )
- **Completeness** : is defiend by the degree to which subject data associated with an entity has values for all expected attributes and related entity instances in a specific context of use.

The data-model dependent metrics are (definition from Confiance. ai program): 
- **Domain Gap** : In the context of a computer vision task, the Domain Gap (DG) refers to the difference in semantic, textures and shapes between two distributions of images and it can lead to poor performances when a model is trained on a given distribution and then is applied to another one.

[//]: # (- Coverage : The coverage of a couple "Dataset + ML Model" is the ability of the execution of the ML Model on this dataset to generate elements that match the expected space.)

For each metric, several approaches are developped to handle the maximum of data types. For more technical and scientific details, you can refer back to this [delivrable](https://catalog.confiance.ai/records/p46p6-1wt83/files/Scientific_Contribution_For_Data_quality_assessment_metrics_for_Machine_learning_process-v2.pdf?download=1)

## Project description
Several approches are developped as decscribed in the figure below.

<img src="_static/library_view.png" width="1024"/>

In the current version, the available metrics are: 
- Representativeness : 
  - $\chi^2$ goodness of fit test for uniform and normal distibutions. 
  - Kolmogorov Smirnov test for uniform and normal distributions. 
  - Granular and Relative Theorithecal Entropy GRTE proposed and developed in Confiance.ai program. 
- Diversity : 
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

[//]: # (- Coverage : )

[//]: # (  - Approches developed in Neural Coverage &#40;NCL&#41; given [here]&#40;https://github.com/Yuanyuan-Yuan/NeuraL-Coverage&#41;. )

# Getting started 

## Set up a clean virtual environnement
Validated python version : [3.9.18, 3.12.9]

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

## Install the library (restricted access)

You can install it by a direct downloading from PyPi using the command 

````
pip install dqm
````

You can installing it from it github sources by launching the following command
````
pip install git+https://github.com/IRT-SystemX/DataQualityMetrics
````
If you got the source code as a zip file, you can install the library from the root directory of the archive by typing : 
```
pip install .
```

## Usage

[//]: # (All validated and verified functions are detailed in the files **call_main.py**. )

Each metric is used by importing the corresponding modules and class into your code
For more information about each metrics see the specific README.md in ```dqm/metrics_name``` subfolders

## Available examples

Many examples of DQM application are present in folder ```/examples```

You will find :

2 jupyter_notebooks:

- **multiple_metrics_tests.ipynb** : A notebook applying completeness, diversity and representativeness metrics on an example dataset.
- **domain_gap.ipynb** : A notebook showing example of applying domain_gap metrics on a generated synthetic dataset

4 python scripts:

Those scripts named **main_X.py** gives an example of computation of metrics implemented for domain X on an example dataset.

The ```main_domain_gap.py``` script requires to be called with a config file passed as an 
argument named ```--cfg``` .

Thus, it is called for example with such command:

``` python examples/main_domain_gap.py --cfg examples/domain_gap_cfg/cmd/cmd.json``` 

We propose in folder ```/examples/domain_gap_cfg ```a set of config files for each domain_gap metrics`:

For some domain_gap examples configuration files the **200_bird_dataset** will be required. It can be downloaded at this [link](http://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models/200-birds-species.zip).
The zip archive shall be extracted in ```examples/datasets/``` folder.

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