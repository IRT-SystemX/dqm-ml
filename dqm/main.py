"""
This script is the entry point for using DQM with command line and docker
"""

from pathlib import Path
import os
import argparse
import numpy as np
import yaml
import pandas as pd

from dqm.completeness.metric import DataCompleteness
from dqm.diversity.metric import DiversityIndexCalculator
from dqm.representativeness.metric import DistributionAnalyzer
from dqm.domain_gap.metrics import CMD, MMD, Wasserstein, ProxyADistance, FID, KLMVN
from dqm.utils.twe_logger import get_logger

# Init the logger
logger = get_logger()


def load_dataframe(config_dict):
    """
    This function loads a pandas dataframe from the config dict passed as input.
    This config dict comes from a pipeline configuration: An example of such pipeline is present in examples/ folder

    Args:
       config_dict (dict): Dict containing a metric configuration
    """

    extension_filter = ""
    separator = ","
    dataset_path = config_dict["dataset"]

    # Scan existing extension and separator fields if they exists
    if "extension" in config_dict.keys():
        extension_filter = config_dict["extension"]
    if "separator" in config_dict.keys():
        separator = config_dict["separator"]

    df = pd.DataFrame()

    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        logger.exception(
            "FileNotFoundError -> The dataset %s does not exists", dataset_path
        )
        raise FileNotFoundError

    # In case of a path to a directory , iterate on files and concatenate raw data
    if os.path.isdir(dataset_path):
        search_path = Path(dataset_path)

        # Search all files in folder and subfolder with specified extension
        file_list = [str(x) for x in list(search_path.rglob("*." + extension_filter))]
        logger.debug("Number of files found in target folder : %s", str(len(file_list)))

        if not file_list:
            logger.exception(
                "Error, no data files have been found in the dataset folder"
            )
            raise ValueError

        # Concatenate raw data of found datafiles
        for file_path in file_list:
            tmp_df = load_raw_data(file_path, separator)
            df = pd.concat([df, tmp_df])

    else:  # otherwise direct load file content as dataframe
        df = load_raw_data(dataset_path, separator)

    return df


def load_raw_data(file, separator):
    """
    This function load a raw data file content as a pandas dataframe

    Args:
       file (str): Path of the file to load
       separator (str): Separator to use when processing csv and txt format file

    Returns:
       df (pandas.DataFrame): Output dataframe
    """

    # Check if the passed extension of input path is supported
    extension = file.split(".")[-1]
    if extension not in ["csv", "txt", "xslx", "xls", "parquet", "pq"]:
        logger.exception(
            "The file named %s has an extension that is not supported :--> %s",
            file,
            extension,
        )
        raise ValueError

    # Call the appropriate read function
    match extension:
        case "csv" | "txt":
            df = pd.read_csv(file, sep=separator)
        case "xslx" | "xls":
            df = pd.read_excel(file)
        case "parquet" | "pq":
            df = pd.read_parquet(file)

    return df


def main():
    """
    Main script of DQM component:

    Args:
        pipeline_config_path (str): Path to the pipeline definition you want to apply
        result_file_path : (str): Path the output YAML file where all computed metrics scores are stored
    """

    parser = argparse.ArgumentParser(description="Main script of DQM")

    parser.add_argument(
        "--pipeline_config_path",
        required=True,
        type=str,
        help="Path to the pipeline definition where you specify each metric you want to compute and its params",
    )

    parser.add_argument(
        "--result_file_path",
        required=True,
        type=str,
        help="Path the output YAML file where all computed metrics scores are stored",
    )

    args = parser.parse_args()

    logger.info("Starting DQM-ML . .")

    # Read the pipeline configuration file

    with open(args.pipeline_config_path, "r", encoding="utf-8") as stream:
        pipeline_config = yaml.safe_load(stream)

    # Create output diretory if it does not exist

    Path((os.sep).join(args.result_file_path.split(os.sep)[:-1])).mkdir(
        parents=True, exist_ok=True
    )

    logger.debug("creation directory : %s ", args.result_file_path.split(os.sep)[:-1])

    # Init output results dict, we start from the input config dict, we will just complete this dict with scores fields

    res_dict = pipeline_config.copy()

    # Loop on metrics to compute

    for idx in range(0, len(pipeline_config["pipeline_definition"])):
        item = pipeline_config["pipeline_definition"][idx]

        # For metrics working only on tabular (all metrics excepted domain gap category)
        if item["domain"] != "domain_gap":
            logger.info(
                "procesing dataset : %s for domain : %s ",
                item["dataset"],
                item["domain"],
            )

            # Load dataset
            main_df = load_dataframe(item)

            # Init list of columns on which metrics shall be computed
            working_columns = list(main_df.columns)  # By default, consider all columns

            # Overload with column_names fied, if it does exist in configuration
            if "columns_names" in item.keys():
                working_columns = item["columns_names"]

            # Init score field that will be filled
            res_dict["pipeline_definition"][idx]["scores"] = {}

            # Call the corresponding metric computation functions
            match item["domain"]:
                case "completeness":
                    # Compute overall completness scores
                    completeness_evaluator = DataCompleteness()
                    res_dict["pipeline_definition"][idx]["scores"]["overall_score"] = (
                        completeness_evaluator.completeness_tabular(main_df)
                    )

                    # Compute column specific completeness
                    for col in working_columns:
                        res_dict["pipeline_definition"][idx]["scores"][col] = (
                            completeness_evaluator.data_completion(main_df[col])
                        )

                case "diversity":
                    # Compute diversity scores
                    metric_calculator = DiversityIndexCalculator()

                    for metric in item["metrics"]:
                        res_dict["pipeline_definition"][idx]["scores"][metric] = {}
                        for col in working_columns:
                            match metric:
                                case "simpson":
                                    computed_score = metric_calculator.simpson(
                                        main_df[col]
                                    )
                                case "gini":
                                    computed_score = metric_calculator.gini(
                                        main_df[col]
                                    )
                                case _:
                                    raise ValueError(
                                        "The given metric", metric, "is not implemented"
                                    )

                            res_dict["pipeline_definition"][idx]["scores"][metric][
                                col
                            ] = computed_score

                case "representativeness":
                    # Prepare output fields in result dict
                    for metric in item["metrics"]:
                        res_dict["pipeline_definition"][idx]["scores"][metric] = {}

                    # Init analyzer
                    bins = item["bins"]
                    distribution = item["distribution"]

                    # Compute representativeness
                    for col in working_columns:
                        var = main_df[col]
                        mean = np.mean(var)
                        std = np.std(var)
                        analyzer = DistributionAnalyzer(var, bins, distribution)

                        for metric in item["metrics"]:
                            match metric:
                                case "chi-square":
                                    pvalue, _ = analyzer.chisquare_test()
                                    computed_score = pvalue

                                case "kolmogorov-smirnov":
                                    computed_score = analyzer.kolmogorov(mean, std)

                                case "shannon-entropy":
                                    computed_score = analyzer.shannon_entropy()

                                case "GRTE":
                                    grte_result, _ = analyzer.grte()
                                    computed_score = grte_result

                                case _:
                                    raise ValueError(
                                        "The given metric", metric, "is not implemented"
                                    )

                            res_dict["pipeline_definition"][idx]["scores"][metric][
                                col
                            ] = computed_score

        # Specificely for domain gap metrics . .
        else:
            # Init score output file
            res_dict["pipeline_definition"][idx]["scores"] = {}

            # Iterate on metrics items

            for metric_dict in item["metrics"]:
                config_method = metric_dict["method_config"]
                metric = metric_dict["metric_name"]

                logger.info(
                    "procesing domain gap for metric : %s for source dataset :  %s and target dataset : %s",
                    metric,
                    config_method["DATA"]["source"],
                    config_method["DATA"]["target"],
                )

                match metric:
                    case "wasserstein":
                        wass = Wasserstein()
                        computed_score = wass.compute_1D_distance(config_method)

                    case "FID":
                        fid = FID()
                        computed_score = fid.compute_image_distance(config_method)

                    case "KLMVN":
                        klmvn = KLMVN()
                        computed_score = klmvn.compute_image_distance(config_method)

                    case "PAD":
                        pad = ProxyADistance()
                        computed_score = pad.compute_image_distance(config_method)

                    case "MMD":
                        mmd = MMD()
                        computed_score = mmd.compute(config_method)

                    case "CMD":
                        cmd = CMD()
                        computed_score = cmd.compute(config_method)

                    case _:
                        logger.exception(
                            "The given metric %s is not implemented", metric
                        )
                        raise ValueError

                # Add computed metric to results

                res_dict["pipeline_definition"][idx]["scores"][metric] = float(
                    computed_score
                )

    # Export final results to yaml file

    with open(args.result_file_path, "w+", encoding="utf-8") as ff:
        yaml.dump(res_dict, ff, default_flow_style=False, sort_keys=False)

    logger.info("pipeline final results exported to file : %s", args.result_file_path)


if __name__ == "__main__":
    main()
