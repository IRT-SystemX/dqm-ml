# This scripts is the entry for using DQM with command line and docker

# test commnad : python main.py --pipeline_config_path "../examples/pipeline_example.yaml" --result_file_path "results/res.yaml"

import argparse
import yaml
from pathlib import Path
import os
import pandas as pd
from dqm.completeness.metric import DataCompleteness
from dqm.diversity.metric import DiversityIndexCalculator
from dqm.representativeness.metric import DistributionAnalyzer
import numpy as np
from dqm.domain_gap.metrics import CMD, MMD, Wasserstein, ProxyADistance, FID, KLMVN

ROOT_PATH = str(Path(__file__).parent.resolve()) + os.sep # To point on test directory

def load_raw_data(file,separator):
    extension=file.split(".")[-1]
    if extension not in ["csv", "xslx","parquet","pq","txt"]:
        raise Exception ("The file named", file, "has an extension that is not supported :--> .", extension)

    match extension:
        case "csv" | "txt":
            df=pd.read_csv(file,sep=separator)
        case "xslx" |"xls":
            df=pd.read_excel(file)
        case "parquet" | "pq":
            df= pd.read_parquet(file)
    
    return df

def load_dataframe(config_dict):
    """
    This function load a pandas datframe from config dict passed as input. This config dict comes from a pipeline configuration 
    Args:
        input_path : dict - Input path to scan to import dataframe
    """
    
    # Scan extension and seprator option for csv file
    extension=""
    separator=","
    dataset_path= config_dict["dataset"]
   
    if "extension" in config_dict.keys():
        extension=config_dict["extension"]
    if "separator" in config_dict.keys():
        separator=config_dict["separator"]

    df= pd.DataFrame()

    if not os.path.exists():
        raise Exception("the dataset", dataset_path, " does not exists")

    # In case of a directory , iterate on file and concatenate raw data
    if os.path.isdir(dataset_path):
         
        search_path=Path(dataset_path)
        file_list = [str(x) for x in list(search_path.rglob("*."+extension))] # Search all files in folder and subfolder with sepcified extension
        print("list des fichiers a concatener :", file_list)
        
        for file_path in file_list:
            tmp_df=load_raw_data(file_path,separator)
            df=pd.concat([df,tmp_df])

    else: # otherwise direct load file content as dataframe
        df= load_raw_data(dataset_path,separator)

    return df
   

def main():
    """
    Main script of DQM component:

    Args:
        pipeline_config_path : str - Path to the pipeline definition where you specify each metric you want to compute and its params
        result_file_path : str - Path the output YAML file where all computed metrics scores are stored 
    """

    parser = argparse.ArgumentParser(description="Main script of DQM")

    parser.add_argument(
        "--pipeline_config_path",
    required=True,
    type=str,
    help="Path to the pipeline definition where you specify each metric you want to compute and its params"
    )

    parser.add_argument(
        "--result_file_path",
    required=True,
    type=str,
    help="Path the output YAML file where all computed metrics scores are stored"
    )

    args = parser.parse_args()
    # print("args", args)

    # Read the pipeline configuration file

    with open(args.pipeline_config_path, 'r') as stream:
        pipeline_config = yaml.safe_load(stream)

    # Crate output diretory if it does not exist
    # print("creation directory ", args.result_file_path.split(os.sep)[:-1])
    Path((os.sep).join(args.result_file_path.split(os.sep)[:-1])).mkdir(parents=True, exist_ok=True)

    # Init output results dict, we keep parameters from config , we will just complete this config with scores fields

    res_dict=pipeline_config.copy()
    # Loop on metrics to compute

    for idx in range(0,len(pipeline_config["pipeline_definition"])):
        
        item=pipeline_config["pipeline_definition"][idx]
        # print("item_en_cours :", item)
        
        # For metrics working on tabular
        if item["domain"] != "domain_gap":
            
            # Load dataframe for specified domain
            print("procesing dataset :", item["dataset"], "for domain : ",item["domain"] )
            main_df=load_dataframe(item)

            # init col variable
            working_columns=list(main_df.columns) # By default

            # Prepare score field that will be added to result dict
            res_dict["pipeline_definition"][idx]["scores"]={}

            # Work only of specified column in keyword exists
            if "columns_names" in item.keys():
                working_columns=item["columns_names"]
            
            match item["domain"]:

                case "completeness" :

                    # Compute completness scores
                    completeness_evaluator = DataCompleteness()
                    res_dict["pipeline_definition"][idx]["scores"]["overall_score"]=completeness_evaluator.completeness_tabular(main_df) 
                    
                    for col in working_columns :
                        res_dict["pipeline_definition"][idx]["scores"][col]= completeness_evaluator.data_completion(main_df[col])
                    
                case "diversity" :

                    # Compute diversity scores
                    metric_calculator= DiversityIndexCalculator()

                    for metric in item["metrics"]:
                        res_dict["pipeline_definition"][idx]["scores"][metric]={}
                        for col in working_columns :
                            match metric:
                                case "simpson":
                                    computed_score=metric_calculator.simpson(main_df[col])
                                case "gini":
                                    computed_score=metric_calculator.gini(main_df[col])
                                case _:
                                   raise Exception("The given metric", metric, "is not implemented") 
            
                            res_dict["pipeline_definition"][idx]["scores"][metric][col]=computed_score     


                case "representativeness" :

                    #Prepare output fields in result dict
                    for metric in item["metrics"]:
                        res_dict["pipeline_definition"][idx]["scores"][metric]={}

                    # init analyzer
                    bins=item["bins"]
                    distribution=item["distribution"]

                    # Compute representativeness
                    for col in working_columns :

                        var= main_df[col]
                        mean = np.mean(var)
                        std = np.std(var)
                        analyzer = DistributionAnalyzer(var, bins, distribution)

                        for metric in item["metrics"]:
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
                            
                            res_dict["pipeline_definition"][idx]["scores"][metric][col]=computed_score    
                                        
        # Specificely for domain gap metrics . . 
        else:
             # Init score output file
             res_dict["pipeline_definition"][idx]["scores"]={}

            # iterate of metrics 

             for metric_dict in item["metrics"]:
                
                config_method=metric_dict["method_config"]
                metric=metric_dict["metric_name"]
                print("procesing domain gap for metric : ", metric, "for datasets : ",config_method["DATA"]["source"],config_method["DATA"]["target"] )
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

                # Add computed metric to results

                res_dict["pipeline_definition"][idx]["scores"][metric]=float(computed_score)

                    
    # Export final results to yaml file
    # print("final results")
    # print(res_dict)
    with open(args.result_file_path, 'w+') as ff:
        yaml.dump(res_dict, ff,default_flow_style=False,sort_keys=False)





if __name__ == "__main__":
    main()