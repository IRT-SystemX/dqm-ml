# This files contain functionnal tests to test metrics right computations

import sys
sys.path.append('..')

import pandas as pd
import pytest
from pathlib import Path
import numpy as np
import argparse
from PIL import Image
import torch

from dqm.completeness.metric import DataCompleteness
from dqm.diversity.diversity import DiversityCalculator
from dqm.diversity.metric import DiversityIndexCalculator
from dqm.representativeness.metric import DistributionAnalyzer
from dqm.domain_gap.metrics import CMD, MMD, Wasserstein, ProxyADistance, FID, KLMVN
from dqm.domain_gap.utils import load_config, display_resume


ROOT_PATH = str(Path(__file__).parent.resolve()) # To point on test directory

# Test of completeness

def test_completeness():

    # Test parameters

    # Those expected scores have been computed manually
    expected_scores={"column_score_1" :1,
                     "column_score_3" :0.78,
                     "column_score_6" :0.48,
                     "column_score_9" :0.211,
                     "overall_score" :0.618
                    } 
    # Accepted Tolerence threshold for comparing computed values and expected values 
    epsilon=0.01
    
    # Load test dataset
    df=pd.read_csv(ROOT_PATH+"/sample_data/completeness_sample_data.csv") # columns 1,3,6,9 
    # print(df)
    
    # Init evaluator
    
    completeness_evaluator = DataCompleteness()
    
    # Calculate the completeness scores for each column of the dataset

    computed_scores={
    "column_score_1" : completeness_evaluator.data_completion(df["column_1"]),
    "column_score_3" : completeness_evaluator.data_completion(df["column_3"]),
    "column_score_6" : completeness_evaluator.data_completion(df["column_6"]),
    "column_score_9" : completeness_evaluator.data_completion(df["column_9"]),
    "overall_score": completeness_evaluator.completeness_tabular(df)   
    }

    # Display results for debug
   
    print("computed scores",computed_scores)
    print("expected_scores",expected_scores)

    # Test approx equality between computed scores and expected ones
    
    for col_name in computed_scores.keys():
        assert computed_scores[col_name] == pytest.approx(expected_scores[col_name], abs=epsilon), \
            f"Value {computed_scores[col_name]} is not close to the expected one ---> {expected_scores[col_name]}"

def test_diversity():
    
    # Test parameters : set from methods themselves

    expected_scores={
        "column_2":{
            "simpson":0.993,"gini":0.992#"lexical_richness":0,"lexical_variety":0,"visual_color":0,"visual_shape":0
        },
        "column_4":{
            "simpson":0.990,"gini":0.989#"lexical_richness":0,"lexical_variety":0,"visual_color":0,"visual_shape":0
        },
        "column_6":{
            "simpson":0.944,"gini":0.9439#"lexical_richness":0,"lexical_variety":0,"visual_color":0,"visual_shape":0
        }
    } 
                    
    # Accepted Tolerence threshold for comparing computed values and expected values 
    epsilon=0.001
    
    # Load test datasets
    df=pd.read_csv(ROOT_PATH+"/sample_data/SMD_test_ds_sample.csv") #columns 1,3,6,9 with 
    # print(df)

   # We choose only 3 columns from the dataset for the tests
    features=["column_2","column_4","column_6"]
    computed_scores={}

    # Compute diversity metrics -> gini and simpson
    
    metric_calculator= DiversityIndexCalculator()
    diversity_calculator=DiversityCalculator()
    
    for feat in features:
        
        computed_scores[feat]={
            "simpson":metric_calculator.simpson(df[feat]),
            "gini":metric_calculator.gini(df[feat]),
            # "lexical_richness":diversity_calculator.compute_diversity(df[feat], "lexical", "richness"),
            # "lexical_variety":diversity_calculator.compute_diversity(df[feat], "lexical", "variety"),
            # "visual_color":diversity_calculator.compute_diversity(df[feat], "visual", "color"),
            # "visual_shape":diversity_calculator.compute_diversity(df[feat], "visual", "shape")
        }

    print("computed" , computed_scores)
    print("expected" , expected_scores)
    
    # Test approx equality between computed scores and expected ones

    for col_name in computed_scores.keys():
        for metric_name in ["simpson","gini"]: #,"lexical_richness","lexical_variety","visual_color","visual_shape"]:
            assert computed_scores[col_name][metric_name] == pytest.approx(expected_scores[col_name][metric_name], abs=epsilon), \
            f"Value {computed_scores[col_name][metric_name]} is not close to the expected one --->{expected_scores[col_name][metric_name]}"
    
def test_representativeness():
    expected_scores={
        "column_2":{
            "chi-square":0,"kolmogorov-smirnov":0,"shannon_entropy":2.3,"GRTE":0.7
        },
        "column_4":{
            "chi-square":0,"kolmogorov-smirnov":0,"shannon_entropy":2.3,"GRTE":0.69
        },
        "column_6":{
            "chi-square":0,"kolmogorov-smirnov":0,"shannon_entropy":2.29,"GRTE":0.69
        }
    }
    
    # Load test datasets
    df=pd.read_csv(ROOT_PATH+"/sample_data/SMD_test_ds_sample.csv") #columns 1,3,6,9 with 
    print(df)
   # We choose only 3 columns from the dataset for the tests
    features=["column_2","column_4","column_6"]
    computed_scores=dict(zip(features,len(features)*[{}]))
    
    # Parameters for analysis
    bins = 10
    distribution = 'normal'         
    # Accepted Tolerence threshold for comparing computed values and expected values 
    epsilon=0.1

    for feat in features:
        print("feature", feat)
        var= df[feat]
        mean = np.mean(var)
        std = np.std(var)
        print("meanvar",mean,std)
        analyzer = DistributionAnalyzer(var, bins, distribution)
        # Using the method chisquare_test
        pvalue, intervals_frequencies = analyzer.chisquare_test()
        computed_scores[feat]["chi-square"]=pvalue
        #computed_scores[feat]["chi-square-interval_freq"]=intervals_frequencies
        # print(f"Chi-Square Test: p-value = {pvalue}")
        
        # Using the method kolmogorov
        ks_pvalue = analyzer.kolmogorov(mean, std)
        # print(f"Kolmogorov-Smirnov Test: p-value = {ks_pvalue}")
        computed_scores[feat]["kolmogorov-smirnov"]=ks_pvalue
        
        # Using the method shannon_entropy
        entropy = analyzer.shannon_entropy()
        # print(f"Shannon Entropy: {entropy}")
        computed_scores[feat]["shannon_entropy"]=entropy
        
        # Using the method grte
        grte_result, intervals_discretized = analyzer.grte()
        print("grte_results",grte_result)
        # print(f"GRTE: {grte_result}")
        computed_scores[feat]["GRTE"]=grte_result
        #computed_scores[feat]["GRTE-intervals_discr"]=intervals_discretized

       
    # Test approx equality between computed scores and expected ones

    print("computed scores : ", computed_scores)
    print("expected scores : ", expected_scores)
    for col_name in computed_scores.keys():
        print("validation metrique", col_name)
        for metric_name in ["chi-square","kolmogorov-smirnov","shannon_entropy","GRTE"]:
            assert computed_scores[col_name][metric_name] == pytest.approx(expected_scores[col_name][metric_name], abs=epsilon), \
            f"Value {computed_scores[col_name][metric_name]} is not close to the expected one --->{expected_scores[col_name][metric_name]}"
        
 
def domain_gap_wassertein():

    expected_wasserstein=5.16
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"

    # Wasserstein

    epsilon=0.1
    expected_score=0.34
    
    wass_config_json = {
    	"DATA": {
    		"batch_size": 10,
    		"height": 299,
    		"width": 299,
    		"norm_mean": [
    				0.485,
    				0.456,
    				0.406
    			],
    		"norm_std": [
    				0.229,
    				0.224,
    				0.225
    			],
    		"source": source_folder, 
    		"target": target_folder  
    	},
    	"MODEL": {
            "arch": "resnet18",
    		"device": "cpu",
    		"n_layer_feature": -2
        	},
    	"METHOD": {
    		"name": "wasserstein",
    		"dimension": "1D"
    	}
    }
    wass = Wasserstein()
    wasserstein_score = wass.compute_1D_distance(wass_config_json)

    print(f"wasserstein score: {wasserstein_score.item()}")
    assert wasserstein_score == pytest.approx(expected_score, abs=epsilon)
    
def domain_gap_FID():

    epsilon=1
    expected_score=444.10

    # Load data
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"

    fid = FID()
    
    # Define your own config file, you can find examples in dqm/domain_gap/cfg/{metric_name}
    fid_config_json = {
    	"DATA": {
    		"batch_size": 32,                      # Features will be compute on {batch_size} images at the same time
    		"height": 299,                         # Resize images height to {height} value
    		"width": 299,                          # Resize images width to {width} value
    		"norm_mean": [                         # Normalize images mean with {norm_mean} values for RGB channels
    				0.485,
    				0.456,
    				0.406
    			],
    		"norm_std": [                          # Normalize images std with {norm_std} values for RGB cahnnels
    				0.229,
    				0.224,
    				0.225
    			],
    		"source": source_folder,      # source images are retrieved from {source} path
    		"target": target_folder       # target images are retrieved from {target} path
    	},
    	"MODEL": {
    		"device": "cpu",                       # Metric will be computed in {device}
    		"n_layer_feature": -2                  # the layer extractor feature will be the:
        	},                                     # i-th if int       |  {n_layer_feature} if str
    	"METHOD": {
    		"name": "fid"                          # Metric name, used only with CLI
    	}
    }

    # Compute the metric
    FID_score = fid.compute_image_distance(fid_config_json)
    print(f"FIDn score: {FID_score.item()}")   
    assert FID_score == pytest.approx(expected_score, abs=epsilon)

def domain_gap_KLMVN():

    epsilon=1
    expected_score=14576664

     # Data path
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"
    
    klmvn = KLMVN()
    # Define your own config file, you can find examples in dqm/domain_gap/cfg/{metric_name}
    klmvn_config_json = {
	"DATA": {
		"batch_size": 10,
		"height": 28,
		"width": 28,
		"norm_mean": [
				0.485,
				0.456,
				0.406
			],
		"norm_std": [
				0.229,
				0.224,
				0.225
			],
		"source": source_folder, 
		"target": target_folder 
	},
	"MODEL": {
        "arch": "resnet18",
		"device": "cpu",
		"n_layer_feature": -2
    	},
	"METHOD": {
		"name": "klmvn"
	}
}

    # Compute the metric
    KLMVN_score = klmvn.compute_image_distance(klmvn_config_json)
    print(f"KLMVN score: {KLMVN_score.item()}")   
    assert  KLMVN_score == pytest.approx(expected_score, abs=epsilon)

def domain_gap_PAD():

    epsilon=0.1
    expected_score=1.95

     # Data path
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"
    
    pad = ProxyADistance()
    # Define your own config file, you can find examples in dqm/domain_gap/cfg/{metric_name}
    pad_config_json = {
	"DATA": {
		"height": 224,
		"width": 224,
		"batch_size": 10,
		"norm_mean": [
			0.485,
			0.456,
			0.406
		],
		"norm_std": [
			0.229,
			0.224,
			0.225
		],
		"source": source_folder, 
		"target": target_folder 
	},
	"MODEL": {
		"arch": ["efficientnet_b0","vgg16"],
		"device": "cpu",
		"n_layer_feature": -2
	},
	"METHOD": {
		"name": "proxy",
        "evaluator": "mse"
	}
    }

    # Compute the metric
    pad_score = pad.compute_image_distance(pad_config_json)
    print(f"PAD score: {pad_score.item()}")   
    assert  pad_score == pytest.approx(expected_score, abs=epsilon)

def domain_gap_MMD():

    epsilon=0.1
    expected_score=355.8

     # Data path
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"
    
    mmd = MMD()
    # Define your own config file, you can find examples in dqm/domain_gap/cfg/{metric_name}
    mmd_config_json = {
	"DATA": {
		"height": 224,
		"width": 224,
		"batch_size": 10,
		"norm_mean": [
			0.485,
			0.456,
			0.406
		],
		"norm_std": [
			0.229,
			0.224,
			0.225
		],
		"source": source_folder, 
		"target": target_folder 
	},
	"MODEL": {
        "arch": "resnet18",
		"device": "cpu",
		"n_layer_feature": -2
    	},
	"METHOD": {
		"name": "mmd",
		"kernel": "linear",
		"kernel_params": {
			"gamma": 1.0,
			"degree": 3.0,
			"coefficient0": 1.0 
		}
	}
}

    # Compute the metric
    mmd_score = mmd.compute(mmd_config_json)
    print(f"MMD score: {mmd_score}")   
    assert  mmd_score == pytest.approx(expected_score, abs=epsilon)


def domain_gap_CMD():

    epsilon=0.1
    expected_score=0.13

     # Data path
    source_folder = ROOT_PATH+"/sample_data/image_test_ds/c20"
    target_folder = ROOT_PATH+"/sample_data/image_test_ds/c33"
    
    cmd = CMD()
    # Define your own config file, you can find examples in dqm/domain_gap/cfg/{metric_name}
    cmd_config_json = {
	"DATA": {
		"height": 224,
		"width": 224,
		"batch_size": 10,
		"norm_mean": [
			0.485,
			0.456,
			0.406
		],
		"norm_std": [
			0.229,
			0.224,
			0.225
		],
		"source": source_folder,
		"target": target_folder
	},
	"MODEL": {
		"arch": "resnet18",
        "n_layer_feature" : [
            "maxpool",
            "layer1.1.relu_1",
            "layer2.1.relu_1", 
            "layer3.1.relu_1", 
            "layer4.1.relu_1"],
        "feature_extractors_layers_weights" : [1, 1, 1, 1, 1],
        "device": "cpu"
	},
	"METHOD": {
		"name": "cmd",
        "k": 5
	}
}

    # Compute the metric
    cmd_score=cmd.compute(cmd_config_json)
    print(f"CMD score: {cmd_score}")   
    assert  cmd_score == pytest.approx(expected_score, abs=epsilon)

# Run to test

# test_completeness()
# test_diversity()
# test_representativeness()  
# domain_gap_wassertein()
# domain_gap_FID()
# domain_gap_KLMVN()
# domain_gap_PAD()
# domain_gap_MMD()
# domain_gap_CMD()