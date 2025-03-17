"""
Domain Gap Metric Calculation Script

This script defines a PyTorch module, DomainMeter, for domain adaptation using
Central Moment Discrepancy (CMD) and Kullback-Leibler (KL) divergence.
The script also includes custom dataset classes (RMSELoss and PandasDatasets)
for loading images from a Pandas DataFrame and implementing a custom root mean 
square error (RMSE) loss.

Authors:
    Sabrina CHAOUCHE
    Yoann RANDON
    Faouzi ADJED

Classes:
    ModelConfiguration
    RMSELoss
    PandasDatasets
    DomainMeter


Dependencies:
    torch
    torchvision
    mlflow
    cmd (DomainMeter class from cmd module)
    twe_logger (Custom logging module)

Usage:
Run the script with optional arguments '--cfg'
for the JSON config file path and '--tsk'
for the JSON task config file path.
"""

from typing import Tuple, List
import json
import os
import torch
from dqm.domain_gap import custom_datasets
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor


def load_config(config_file):
    """Load configuration from a JSON file."""
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file '{config_file}' does not exist.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in '{config_file}'.")
        exit(1)


def display_resume(cfg, dist, time_lapse):
    # Display a summary of the computation
    print("-" * 80)
    print("Summary")
    print(f"source: {cfg['DATA']['source']}")
    print(f"target: {cfg['DATA']['target']}")
    if "batch_size" in cfg["DATA"]:
        print(f"batch size: {cfg['DATA']['batch_size']}")
    if "device" in cfg["MODEL"]:
        print(f"device: {cfg['MODEL']['device']}")
    if "arch" in cfg["MODEL"]:
        print(f"model: {cfg['MODEL']['arch']}")
    if "archs" in cfg["MODEL"]:
        print(f"models: {cfg['MODEL']['archs']}")
    # Check if 'dist' is a tensor and convert to float if necessary
    # distance = dist.item() if isinstance(dist, torch.Tensor) else dist
    # ========================================================================
    if dist is not None:
        distance = dist.item() if isinstance(dist, torch.Tensor) else dist
    else:
        distance = None
    # ========================================================================
    print(f"distance: {distance}")
    print(f"method : {cfg['METHOD']['name']}")
    if "evaluator" in cfg["METHOD"]:
        print(f"evaluator : {cfg['METHOD']['evaluator']}")
    print(f"compute time: {round(time_lapse, 2)} seconds")
    print("-" * 80)


# Function to generate transform
def generate_transform(
    img_size: Tuple[int, int], norm_mean: List[float], norm_std: List[float]
):
    """
    Generate transform to change data input into compatible model inputs

    Args:
        image_size (Tuple[int, int]): value to resize image
        norm_mean (List[float]): normalization mean
        norm_std (List[float]): normalization standard deviation

    Returns:
        transform: a function which apply multiple changes to data
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),  # Resize image
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=norm_mean, std=norm_std),  # Normalize
        ]
    )
    return transform


def extract_nth_layer_feature(model, n):
    # Get the model's named layers
    layer_names = list(dict(model.named_modules()).keys())

    if isinstance(n, list):
        # Create a feature extractor with the nth layer
        feature_extractor = create_feature_extractor(model, return_nodes=n)

        return feature_extractor

    # Handle integer input (layer index)
    if isinstance(n, int):
        # Convert negative index to positive (e.g., -1 means last layer)
        if n < 0:
            n = len(layer_names) + n

        # Ensure the layer index is valid
        if n >= len(layer_names) or n < 0:
            raise ValueError(
                f"Layer index {n} is out of range for the model with {len(layer_names)} layers."
            )

        # Extract the n-th layer's name
        nth_layer_name = layer_names[n]

    # Handle string input (layer name)
    elif isinstance(n, str):
        if n not in layer_names:
            raise ValueError(
                f"Layer name '{n}' not found in the model. Available layers are: {layer_names}"
            )
        nth_layer_name = n

    else:
        raise TypeError(
            "The argument 'n' must be either an integer (layer index) or a string (layer name) or a list of string (layer names)."
        )

    # Create a feature extractor with the nth layer
    feature_extractor = create_feature_extractor(
        model, return_nodes={nth_layer_name: "features"}
    )

    return feature_extractor


def load_model(model_str: str, device: str) -> torch.nn.Module:
    """
    Loads a model based on the input string.

    If the string contains '.pt' or '.pth', tries to load a saved PyTorch model from a file.
    If the string matches a known torchvision model (e.g., 'resnet18'), it loads the corresponding model.

    Parameters:
    model_str (str): The model string or file path.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    """

    # Check if the string is a path to a saved model file
    if model_str.endswith((".pt", ".pth")):
        # Verify the file exists
        if os.path.exists(model_str):
            # Attempt to load the model directly
            try:
                model = torch.load(model_str)
                print(f"Loaded model from {model_str}")
                return model
            except Exception as e:
                raise ValueError(f"Error loading model from file: {e}")
        else:
            raise FileNotFoundError(f"Model file '{model_str}' not found.")

    else:
        model = torchvision.models.get_model(model_str, pretrained=True).to(device)
    return model


def compute_features(dataloader, model, device):
    """
    Compute features from a model for images in the DataLoader batch by batch.

    Args:
        dataloader (DataLoader): DataLoader object to load images in batches.
        model (torch.nn.Module): Pre-trained model to extract features.
        device (torch.device): Device to run the model (e.g., CPU or GPU).

    Returns:
        torch.Tensor: A concatenated tensor of features for all images.
    """
    model.eval()  # Set the model to evaluation mode
    all_features = []

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            batch = batch.to(device)  # Move the batch to the target device (GPU/CPU)
            features = model(batch)["features"].squeeze()  # Extract features
            all_features.append(features)

    return torch.cat(all_features)  # Concatenate features from all batches


def construct_dataloader(folder_path: str, transform, batch_size: int):
    """
    Loads images from a folder and returns a DataLoader for batch-wise processing.

    Args:
        folder_path (str): Path to the folder containing images.
        transform (transform): Transform object to fine-tune data for model input.
        batch_size (int): Number of images per batch.

    Returns:
        DataLoader: A DataLoader object that yields batches of transformed images.
    """
    dataset = custom_datasets.ImagesFromFolderDataset(folder_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
