# This scripts is the entry for using DQM with command line and docker

# test commnad : python main.py --pipeline_config_path "../examples/pipeline_example.yaml" --result_file_path "results/res.yaml"

import argparse
import yaml
from pathlib import Path
import os
import pandas as pd
import glob

ROOT_PATH = str(Path(__file__).parent.resolve()) + os.sep # To point on test directory

def load_raw_data(file,separator):
    extension=file.split(".")[-1]
    if not extension in ["csv", "xslx","parquet","pq","txt"]:
        raise Exception ("The file named", file, "has an extension that is not supported :--> .", extension)

    match extension:
        case "csv" | "txt":
            df=pd.read_csv(file,sep=separator)
        case "xslx" |"xls":
            df=pd.read_excel(file)
        case "parquet" | "pq":
            df= pd_read_parquet(file)
    
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
    print("args", args)

    # Read the pipeline configuration file

    with open(args.pipeline_config_path, 'r') as stream:
        pipeline_config = yaml.safe_load(stream)

    print(pipeline_config)

    # Crate output file if it does not exist

    if not os.path.isfile(args.result_file_path):
         Path(args.result_file_path).mkdir(parents=True, exist_ok=True)

    # prepare computation

    res_dict={}

    # Loop on metrics to compute

    for item in pipeline_config["pipeline_definition"]:
        print("domain traite :", item["domain"])
        

        if item["domain"] != "domain_gap":
            print("dataset :", item["dataset"])
            main_df=load_dataframe(item)

        else:
            print("domain_gap auto loading  not implemented yet")

        print(main_df)











if __name__ == "__main__":
    main()