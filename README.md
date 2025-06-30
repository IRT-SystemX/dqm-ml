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

<br>
<br>

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

There is two ways to use the dqm library :
  - Import dqm package and call the dqm functions within your python code
  - In standalone mode using direct command line from a terminal, or run the DQm-ML container

### Standalone mode

You can use the dqm-ml directly to evaluate your dataset, by using the "dqm-ml" command from your terminal.

The command line has the following form :

```dqm-ml --pipeline_config_path path_to_your_pipeline_file --result_file_path path_to_your_result_file```

This mode requires two user parameters:
  - pipeline_config_path : A path to a yaml file that will define the pipeline of evaluation you want to apply on your datasests
  - result_file_path : A yaml file containing the set of computed scores for each defined metric in your pipeline

For example, if your pipeline file is located at path :  ```examples/pipeline_example.yaml ``` and you want your result file to be stored at  ```"examples/results_pipeline_example.yaml```, you will type in your terminal : 

  ```dqm-ml --pipeline_config_path "examples/pipeline_example.yaml" --result_file_path "examples/results_pipeline_example.yaml"```

### Pipeline definition

A dqm-ml pipeline is a yaml file that contains the list of dataset you want to evaluate, and the list of metrics you want to compute on each ones.
This file has a primary key **pipeline_definition** containing  a list of items where each item has the following required fields:
  - dataset : The path to the dataset you want to evaluate .
  - domain : The category of metric you want to apply
  - metrics : The list of metrics to compute on the dataset . (For completeness only this field is not used)
  
For representativeness domain only, the following additional parameters fields are required:
  - bins : 
  - distribution :

  You can use an optionnal field :
  - columns : The list of columns from your dataset on which you want to restrict the computations of metrics. If this field is missing, by default the metrics are applied on all columns of the given dataset

The field ```datasets ```, can be a path to a single file  or a path to a folder. If the path points on a single file,  the file content is directly loaded and considered as the final dataset to evaluate. Supported extension for files are "csv, txt, xls,xlsx, pq and parquet". In case of csv or txt file,  you can set a ```separator ``` field to indicate the separator to be used to parse the file.

If the defined path is a folder, all files within the folder will be automatically concatened along the rows axis to build the final dataset that will be considered for the evaluation. For folders, you can use an additional ```extension ``` field to concatenate only the files with the specified extension in the target folder. By default, all present files are tried to be concatenated.

For example:

```
 - domain : "representativeness"
    extension: "txt"
    metrics: ["chi-square","GRTE"]
    bins : 10
    distribution : "normal"
    dataset: "tdata/my_data_folder"
    columns_names : ["col_1", "col_5","col_9"]
```


For domain gap, because the metrics apply only on images datasets, the definition is quite different, the item has the following field
  - ```domain```: defining the name of the domain thus here "domain_gap"
  - ```metrics``` :   The list of metrics you want to compute, and for each item you have two fields
      - metrics_name : The name of metric to compute
      - method_config : The user configuration of the metric. In this part you define the source and target datasets, the chosen models, and other user parameters

An example of pipeline file defining the computations of many metrics from the four domains is given below:
```
pipeline_definition:
  - domain : "completeness"
    dataset : "tests/sample_data/completeness_sample_data.csv"
    columns_names : ["column_1","column_3","column_6","column_9"]

  - domain : "representativeness"
    metrics: ["chi-square","GRTE"]
    bins : 10
    distribution : normal
    dataset: "tests/sample_data/SMD_test_ds_sample.csv"
    columns_names : ["column_2","column_4", "column_6"]
    
  - domain : "diversity"
    metrics: ["simpson","gini"]
    dataset: "tests/sample_data/SMD_test_ds_sample.csv"
    columns_names : ["column_2","column_4", "column_6"]

  - domain: "domain_gap"
    metrics:
      - metric_name: wasserstein
        method_config:
            DATA: 
                batch_size: 32
                height: 299
                width: 299
                norm_mean: [0.485,0.456,0.406]
                norm_std: [0.229,0.224,0.225]
                source: "tests/sample_data/image_test_ds/c20"
                target: "tests/sample_data/image_test_ds/c33"  
            MODEL:
                arch: "resnet18"
                device: "cpu"
                n_layer_feature: -2
            METHOD: 
                name: "fid"
```

The result file produced at the end of this pipeline is a yaml file containing the  pipeline configuration file content augmented with a "scores" field in each item, containing the metrics computed scores.

Example of result_score:

```
pipeline_definition:
- domain: completeness
  dataset: tests/sample_data/completeness_sample_data.csv
  columns_names:
  - column_1
  - column_3
  - column_6
  - column_9
  scores:
    overall_score: 0.61825
    column_1: 1
    column_3: 0.782
    column_6: 0.48
    column_9: 0.211
- domain: representativeness
  metrics:
  - chi-square
  - GRTE
  bins: 10
  distribution: normal
  dataset: tests/sample_data/SMD_test_ds_sample.csv
  columns_names:
  - column_2
  - column_4
  - column_6
  scores:
    chi-square:
      column_2: 1.8740034461104008e-34
      column_4: 2.7573644464553625e-86
      column_6: 3.469236770038776e-64
    GRTE:
      column_2: 0.8421470393366073
      column_4: 0.7615162001699769
      column_6: 0.6955152215780268
```

To create your own pipeline definition, it is adviced to start from one existing model of pipeline present in the ```examples/ ``` folder.

###  Use the dockerized version

To build locally the docker image, from the root folder of the repository use the command:

```docker build . -f dockerfile -t your_image_name:tag```

The command line to run the dqm container has the following form :

```docker run -e PIPELINE_CONFIG_PATH="path_to_your_pipeline_file" -e RESULT_FILE_PATH="path_to_the_result_file" irtsystemx/dqm-ml:1.1.1```

You need to mount the ```PIPELINE_CONFIG_PATH``` path to ```/tmp/in/$PIPELIN_CONFIG_PATH``` and the ```$RESULT_FILE_PATH``` to  ```/tmp/out/$RESULT_FILE_PATH```
Moreover, all datasets directories referenced in your pipeline file shall be mounted in the docker

For example if your pipeline file is stored at ```examples/pipeline_example_docker.yaml``` and you want your result file to be stored at ```results_docker/result_file.yaml```
and all your datasets used in your pipeline are stored locally into ```/tests``` folder and defined on ```data_storage/..``` in your pipeline file 

The command would be :

```docker run -e PIPELINE_CONFIG_PATH="pipeline_example_docker.yaml" -e RESULT_FILE_PATH="result_file.yaml" -v ${PWD}/examples:/tmp/in -v ${PWD}/tests/:/data_storage/ -v ${PWD}/results_docker:/tmp/out irtsystemx/dqm-ml:1.1.1```

### User with proxy server

The computation of domain gap metrics requires the use of pretrained models that are automatically downloaded by pytorch in a local cache directory during the first call of those metrics.

For users behind a proxy server, this download could fail. To overcome this issue, you can manually get those pretrained models by downloading the zip archive from this  [link](http://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models//dqm-ml_pretrained_models.zip) and extract it in the following folder : ``` your_user_directory/.cache/torch/hub/checkpoints/```

### Use the library within your python code

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

For some domain_gap examples, the **200_bird_dataset** will be required. It can be downloaded from this [link](http://minio-storage.apps.confianceai-public.irtsysx.fr/ml-models/200-birds-species.zip). The zip archive shall be extracted into the ```examples/datasets/``` folder.

1 pipeline example that instanciates every metrics implemented in dqm-ml named ```pipeline_example.yaml``` and its corresponding results ```results_pipeline_example.yaml```.

1 pipeline example similar to the previous one, but with different datasets path, as shown in the example of how using the containerized version.

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
