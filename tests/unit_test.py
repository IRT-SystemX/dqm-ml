# This file implements unit test for DQM-ml library

import sys
import os
sys.path.append('..')

# Import external dependencies

import pandas as pd
import pytest
from pathlib import Path
import numpy as np
import argparse
from PIL import Image
import torch
import yaml

# Import internal dependencies

from dqm.completeness.metric import DataCompleteness
from dqm.diversity.diversity import DiversityCalculator
from dqm.diversity.metric import DiversityIndexCalculator
from dqm.representativeness.metric import DistributionAnalyzer
from dqm.domain_gap.metrics import CMD, MMD, Wasserstein, ProxyADistance, FID, KLMVN
from dqm.domain_gap.utils import load_config, display_resume

# To force this test file as reference path

ROOT_PATH = str(Path(__file__).parent.resolve()) + os.sep # To point on test directory

# Load global unit tests configuration 

with open(ROOT_PATH+"/tests_config/unit_tests_config.yaml", 'r') as stream:
    tests_config = yaml.safe_load(stream)


def test_completeness():

    # load test configuration
    
    expected_scores=tests_config["completeness"]["expected_scores"]
    epsilon=tests_config["completeness"]["params"]["tolerance"]
    col_names=tests_config["completeness"]["params"]["columns_names"]
    dataset_path=tests_config["completeness"]["params"]["dataset"]
    
    # Load test dataset
    
    df=pd.read_csv(ROOT_PATH+dataset_path) 

    # Init evaluator and calculate the completeness scores for each chosen columns
    
    completeness_evaluator = DataCompleteness()

    # Test completeness by columns
    
    for col in col_names:
        computed_score = completeness_evaluator.data_completion(df[col])
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score,abs=epsilon), \
        f"For column : {col}, the distance between computed value : {computed_score} and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}"

    # Test overall completeness

    computed_score = completeness_evaluator.completeness_tabular(df) 
    expected_score = expected_scores["overall_score"]
    assert computed_score == pytest.approx(expected_score,abs=epsilon), \
    f"For overall_score, the distance between computed value : {computed_score} and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}"

    
@pytest.mark.parametrize("metric", ["simpson","gini"])
def test_diversity_metrics(metric : str):

    # load test configuration
    expected_scores=tests_config["diversity"]["expected_scores"][metric]
    epsilon=tests_config["diversity"]["params"]["tolerance"]
    col_names=tests_config["diversity"]["params"]["columns_names"]
    dataset_path=tests_config["diversity"]["params"]["dataset"]
    
    # Load test datasets
    df=pd.read_csv(ROOT_PATH+dataset_path) #columns 1,3,6,9 with 

    metric_calculator= DiversityIndexCalculator()

    # Compute diversity metrics and compare with expected values
    for col in col_names:

        match metric:
            case "simpson":
                computed_score=metric_calculator.simpson(df[col])
            case "gini":
                computed_score=metric_calculator.gini(df[col])
            case _:
               raise Exception("The given metric", metric, "is not implemented") 
        
        expected_score = expected_scores[col]
        assert computed_score == pytest.approx(expected_score,abs=epsilon), \
        f"For column : {col}, the distance between computed value : {computed_score} and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}"

@pytest.mark.parametrize("metric", ["chi-square","kolmogorov-smirnov","shannon-entropy","GRTE"])
def test_representativeness(metric):

    # load test configuration
    
    expected_scores=tests_config["representativeness"]["expected_scores"][metric]
    epsilon=tests_config["representativeness"]["params"]["tolerance"]
    col_names=tests_config["representativeness"]["params"]["columns_names"]
    dataset_path=tests_config["representativeness"]["params"]["dataset"]
    bins = tests_config["representativeness"]["params"]["bins"]
    distribution = tests_config["representativeness"]["params"]["distribution"]

    # Load test datasets
    
    df=pd.read_csv(ROOT_PATH+dataset_path) #columns 1,3,6,9 with 
  
    # Compute representativeness metrics and compare with expected values

    for col in col_names:
        
        var= df[col]
        mean = np.mean(var)
        std = np.std(var)
        
        analyzer = DistributionAnalyzer(var, bins, distribution)
       
        match metric:

            case "chi-square":
                pvalue, intervals_frequencies = analyzer.chisquare_test()
                computed_score=pvalue

            case "kolmogorov-smirnov":
                computed_score = analyzer.kolmogorov(mean, std)

            case "shannon-entropy":
                computed_score = analyzer.shannon_entropy()

            case "GRTE":    
                grte_result, intervals_discretized = analyzer.grte()
                computed_score=grte_result

            case _:
                raise Exception("The given metric", metric, "is not implemented")

        expected_score= expected_scores[col]
        assert computed_score == pytest.approx(expected_score,abs=epsilon), \
        f"For column : {col}, the distance between computed value : {computed_score} and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}"

@pytest.mark.parametrize("metric", ["wasserstein","FID","KLMVN","PAD","MMD","CMD"])
def test_domain_gaps(metric): 

    # load test configuration
    
    expected_score=tests_config["domain_gap"][metric]["expected_score"]
    epsilon=tests_config["domain_gap"][metric]["params"]["tolerance"]
    config_method=tests_config["domain_gap"][metric]["params"]["method_config"]

    # Overload dataset path with absolute path

    tests_config["domain_gap"][metric]["params"]["method_config"]["DATA"]["source"]=ROOT_PATH+tests_config["domain_gap"][metric]["params"]["method_config"]["DATA"]["source"]
    tests_config["domain_gap"][metric]["params"]["method_config"]["DATA"]["target"]=ROOT_PATH+tests_config["domain_gap"][metric]["params"]["method_config"]["DATA"]["target"]

    # Compute domain_gap metrics and compare with expected values
    
    match metric : 

        case "wasserstein" :
            wass = Wasserstein()
            computed_score = wass.compute_1D_distance(config_method)

        case "FID" :
            fid = FID()
            computed_score = fid.compute_image_distance(config_method)

        case "KLMVN" :
            klmvn = KLMVN()
            computed_score = klmvn.compute_image_distance(config_method)

        case "PAD":
            pad = ProxyADistance()
            computed_score = pad.compute_image_distance(config_method)

        case "MMD":
            mmd = MMD()
            computed_score = mmd.compute(config_method)

        case"CMD":
            cmd = CMD()
            computed_score=cmd.compute(config_method)

        case _:

            raise Exception("The given metric", metric, "is not implemented")

    assert computed_score == pytest.approx(expected_score,abs=epsilon), \
    f"For metric the distance between computed value : {computed_score} and expected one ---> {expected_score} is greater than the accepted tolerance {epsilon}"

