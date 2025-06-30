"""
This module defines a GapMetric class responsible for calculating
the Domain Gap distance between source and target data using various methods and models.
It utilizes various methods and models for this purpose.

Authors:
    Yoann RANDON
    Sabrina CHAOUCHE
    Faouzi ADJED

Dependencies:
    time
    json
    argparse
    typing
    mlflow
    torchvision.models (resnet50, ResNet50_Weights, resnet18, ResNet18_Weights)
    utils (DomainMeter)
    twe_logger

Classes:
    DomainGapMetrics: Class for calculating Central Moment Discrepancy (CMD)
        distance between source and target data.

Functions: None

Usage:
1. Create an instance of GapMetric.
2. Parse the configuration using parse_config().
3. Load the CNN model using load_model(cfg).
4. Compute the CMD distance using compute_distance(cfg).
5. Log MLflow parameters using set_mlflow_params(cfg).
6. Process multiple tasks using process_tasks(cfg, tsk).
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dqm.domain_gap.utils import (
    extract_nth_layer_feature,
    generate_transform,
    load_model,
    compute_features,
    construct_dataloader,
)

from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm
from scipy.linalg import eigh


import ot
import numpy as np

from sklearn import svm


class Metric:
    """Base class for defining a metric."""

    def __init__(self) -> None:
        """Initialize the Metric instance."""
        pass

    def compute(self) -> float:
        """Compute the value of the metric."""
        pass


# ==========================================================================#
#                      MMD - Maximum Mean Discrepancy                       #
# ==========================================================================#
class MMD(Metric):
    """Maximum Mean Discrepancy metric class defintion"""

    def __init__(self) -> None:
        super().__init__()

    def __rbf_kernel(self, x, y, gamma: float) -> float:
        """
        Computes the Radial Basis Function (RBF) kernel between two sets of vectors.

        Args:
            x (torch.Tensor): Tensor of shape (N, D), where N is the number of samples.
            y (torch.Tensor): Tensor of shape (M, D), where M is the number of samples.
            gamma (float): Kernel coefficient, typically 1 / (2 * sigma^2).

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M) with RBF similarities.
        """
        k = torch.cdist(x, y, p=2.0)
        k = -gamma * k
        return torch.exp(k)

    def __polynomial_kernel(
        self, x, y, degree: float, gamma: float, coefficient0: float
    ) -> torch.Tensor:
        """
        Computes the Polynomial Kernel between two tensors.

        The polynomial kernel is defined as:
            K(x, y) = (γ * ⟨x, y⟩ + c) ^ d

        where:
            - ⟨x, y⟩ is the dot product of `x` and `y`
            - γ (gamma) is a scaling factor
            - c (coefficient0) is a bias term
            - d (degree) is the polynomial degree

        Args:
            x (torch.Tensor): A tensor of shape (N, D), where N is the number of samples.
            y (torch.Tensor): A tensor of shape (M, D), where M is the number of samples.
            degree (float): The degree of the polynomial.
            gamma (float): The scaling factor for the dot product.
            coefficient0 (float): The bias term.

        Returns:
            torch.Tensor: A kernel matrix of shape (N, M) containing polynomial similarities.
        """
        k = torch.matmul(x, y) * gamma + coefficient0
        return torch.pow(k, degree)

    @torch.no_grad()
    def compute(self, cfg) -> float:
        """
        Computes a domain gap metric between two datasets using a specified kernel method.

        This function extracts features from source and target datasets using a deep learning model,
        applies a specified kernel function (linear, RBF, or polynomial), and computes a similarity
        measure between the datasets.

        Args:
            cfg (dict): Configuration dictionary containing:
                - `DATA`:
                    - `source` (str): Path to the source dataset.
                    - `target` (str): Path to the target dataset.
                    - `batch_size` (int): Batch size for dataloaders.
                    - `width` (int): Width of input images.
                    - `height` (int): Height of input images.
                    - `norm_mean` (tuple): Mean for normalization.
                    - `norm_std` (tuple): Standard deviation for normalization.
                - `MODEL`:
                    - `arch` (str): Model architecture.
                    - `n_layer_feature` (int): Layer from which features are extracted.
                    - `device` (str): Device to run computations ('cpu' or 'cuda').
                - `METHOD`:
                    - `kernel` (str): Kernel type ('linear', 'rbf', 'poly').
                    - `kernel_params` (dict): Parameters for the chosen kernel.

        Returns:
            float: Computed domain gap value based on the selected kernel.

        Raises:
            AssertionError: If source and target datasets have different sizes.
        """
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]
        batch_size = cfg["DATA"]["batch_size"]
        image_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        model = cfg["MODEL"]["arch"]
        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        device = cfg["MODEL"]["device"]
        kernel = cfg["METHOD"]["kernel"]
        kernel_params = cfg["METHOD"]["kernel_params"]
        device = device

        transform = generate_transform(image_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        loaded_model = load_model(model, device)
        feature_extractor = extract_nth_layer_feature(loaded_model, n_layer_feature)

        source_features_t = compute_features(source_loader, feature_extractor, device)
        target_features_t = compute_features(target_loader, feature_extractor, device)

        # flatten features to compute on matricial features
        source_features = source_features_t.view(source_features_t.size(0), -1)
        target_features = target_features_t.view(target_features_t.size(0), -1)

        # Both datasets (source and target) have to have the same size
        assert len(source_features) == len(target_features)

        feature_extractor.eval()

        # Get the features of the source and target datasets using the model
        if kernel == "linear":
            xx = torch.matmul(source_features, source_features.t())
            yy = torch.matmul(target_features, target_features.t())
            xy = torch.matmul(source_features, target_features.t())

            return torch.mean(xx + yy - 2.0 * xy).item()

        if kernel == "rbf":
            gamma = kernel_params.get("gamma", 1.0)
            if source_features.dim() == 1:
                source_features = torch.unsqueeze(source_features, 0)
            if target_features.dim() == 1:
                target_features = torch.unsqueeze(target_features, 0)
            xx = self.__rbf_kernel(source_features, source_features, gamma)
            yy = self.__rbf_kernel(target_features, target_features, gamma)
            xy = self.__rbf_kernel(source_features, target_features, gamma)

            return torch.mean(xx + yy - 2.0 * xy).item()

        if kernel == "poly":
            degree = kernel_params.get("degree", 3.0)
            gamma = kernel_params.get("gamma", 1.0)
            coefficient0 = kernel_params.get("coefficient0", 1.0)
            xx = self.__polynomial_kernel(
                source_features, source_features.t(), degree, gamma, coefficient0
            )
            yy = self.__polynomial_kernel(
                target_features, target_features.t(), degree, gamma, coefficient0
            )
            xy = self.__polynomial_kernel(
                source_features, target_features.t(), degree, gamma, coefficient0
            )

            return torch.mean(xx + yy - 2.0 * xy).item()


# ==========================================================================#
#                     CMD - Central Moments Discrepancy v2                 #
# ==========================================================================#


class RMSELoss(nn.Module):
    """
    Compute the Root Mean Squared Error (RMSE) loss between the predicted values and the target values.

    This class provides a PyTorch module for calculating the RMSE loss, which is a common metric for
    evaluating the accuracy of regression models. The RMSE is the square root of the average of squared
    differences between predicted values and target values.

    Attributes:
        mse (nn.MSELoss): Mean Squared Error loss module with reduction set to "sum".
        eps (float): A small value added to the loss to prevent division by zero and ensure numerical stability.

    Methods:
        forward(yhat, y): Compute the RMSE loss between the predicted values `yhat` and the target values `y`.
    """

    def __init__(self, eps=0):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class CMD(Metric):

    def __init__(self) -> None:
        super().__init__()

    def __get_unbiased(self, n: int, k: int) -> int:
        """
        Computes an unbiased normalization factor for higher-order statistical moments.

        This function calculates the product of `(n-1) * (n-2) * ... * (n-k+1)`,
        which is used to adjust higher-order moment estimations to be unbiased.

        Args:
            n (int): Total number of samples.
            k (int): Order of the moment being computed.

        Returns:
            int: The unbiased normalization factor.

        Raises:
            AssertionError: If `n <= 0`, `k <= 0`, or `n <= k`.
        """
        assert n > 0 and k > 0 and n > k
        output = 1
        for i in range(n - 1, n - k, -1):
            output *= i
        return output

    def __compute_moments(
        self,
        dataloader,
        feature_extractor,
        k,
        device,
        shapes: dict,
        axis_config: dict[str, tuple] = None,
        apply_sigmoid: bool = True,
        unbiased: bool = False,
    ) -> dict:
        """
        Computes the first `k` statistical moments of feature maps extracted from a dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of input data.
            feature_extractor (callable): Function or model that extracts features from input data.
            k (int): Number of moments to compute (e.g., mean, variance, skewness, etc.).
            device (torch.device): Device on which to perform computations (e.g., "cuda" or "cpu").
            shapes (dict): Dictionary mapping layer names to their corresponding tensor shapes.
            axis_config (dict[str, tuple], optional): Dictionary specifying summation and viewing axes.
                Defaults to `{"sum_axis": (0, 2, 3), "view_axis": (1, -1, 1, 1)}`.
            apply_sigmoid (bool, optional): Whether to apply a sigmoid function to extracted features.
                Defaults to True.
            unbiased (bool, optional): Whether to apply unbiased estimation for higher-order moments.
                Defaults to False.

        Returns:
            dict: A dictionary containing computed moments for each layer. The structure is:
                {
                    "layer_name": {
                        0: mean tensor,
                        1: second moment tensor,
                        ...
                        k-1: kth moment tensor
                    },
                    ...
                }
        """
        # Initialize axis_config if None
        if axis_config is None:
            axis_config = {"sum_axis": (0, 2, 3), "view_axis": (1, -1, 1, 1)}

        # Initialize statistics dictionary
        moments = {layer_name: dict() for layer_name in shapes.keys()}
        for layer_name, shape in shapes.items():
            channels = shape[1]
            for j in range(k):
                moments[layer_name][j] = torch.zeros(channels).to(device)

        # Initialize normalization factors for each layer
        nb_samples = {layer_name: 0 for layer_name in shapes.keys()}  # TOTOTOTO

        # Iterate through the DataLoader
        for batch in dataloader:
            batch = batch.to(device)
            batch_size = batch.size(0)

            # Update the sample count for normalization
            for layer_name, shape in shapes.items():
                nb_samples[layer_name] += batch_size * shape[2] * shape[3]

            # Compute features for the current batch
            features = feature_extractor(batch)

            # Compute mean (1st moment)
            for layer_name, feature in features.items():
                if apply_sigmoid:
                    feature = torch.sigmoid(feature)
                moments[layer_name][0] += feature.sum(axis_config.get("sum_axis"))

            # Normalize the first moment (mean)
            for layer_name, n in nb_samples.items():
                moments[layer_name][0] /= n

        # Compute higher-order moments (k >= 2)
        for batch in dataloader:
            batch = batch.to(device)
            features = feature_extractor(batch)

            for layer_name, feature in features.items():
                if apply_sigmoid:
                    feature = torch.sigmoid(feature)

                # Calculate differences from the mean
                difference = feature - moments[layer_name][0].view(
                    axis_config.get("view_axis")
                )

                # Accumulate moments for k >= 2
                for j in range(1, k):
                    moments[layer_name][j] += (difference ** (j + 1)).sum(
                        axis_config.get("sum_axis")
                    )

        # Normalize higher-order moments
        for layer_name, n in nb_samples.items():
            for j in range(1, k):
                moments[layer_name][j] /= n
                if unbiased:
                    nb_samples_unbiased = self.__get_unbiased(n, j)
                    moments[layer_name][j] *= n**j / nb_samples_unbiased

        return moments

    @torch.no_grad()
    def compute(self, cfg) -> float:
        """
        Compute the Central Moment Discrepancy (CMD) loss between source and target datasets using a pre-trained model.

        This method calculates the CMD loss, which measures the discrepancy between the distributions of features
        extracted from source and target datasets. The features are extracted from specified layers of the model,
        and the loss is computed as a weighted sum of the differences in moments of the feature distributions.

        Args:
            cfg (Dict): A configuration dictionary containing the following keys:
                - "DATA": Dictionary with data-related configurations:
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.
                    - "batch_size" (int): The batch size for data loading.
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                - "MODEL": Dictionary with model-related configurations:
                    - "arch" (str): The architecture of the model to use.
                    - "n_layer_feature" (list of int): List of layer numbers from which to extract features.
                    - "feature_extractors_layers_weights" (list of float): Weights for each feature layer.
                    - "device" (str): The device to run the model on (e.g., "cpu" or "cuda").
                - "METHOD": Dictionary with method-related configurations:
                    - "k" (int): The number of moments to consider in the CMD calculation.

        Returns:
            float: The computed CMD loss between the source and target datasets.

        The method performs the following steps:
        1. Constructs data loaders for the source and target datasets with specified transformations.
        2. Loads the model and sets it up on the specified device.
        3. Extracts features from the specified layers of the model for both datasets.
        4. Computes the moments of the feature distributions for both datasets.
        5. Calculates the CMD loss as a weighted sum of the differences in moments.
        6. Returns the total CMD loss.

        Raises:
            AssertionError: If the source and target datasets do not have the same number of samples.
            AssertionError: If the keys of the feature weights dictionary do not match the specified feature layers.
        """
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]
        batch_size = cfg["DATA"]["batch_size"]
        image_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        model = cfg["MODEL"]["arch"]
        feature_extractors_layers = cfg["MODEL"]["n_layer_feature"]
        k = cfg["METHOD"]["k"]
        feature_extractors_layers_weights = cfg["MODEL"][
            "feature_extractors_layers_weights"
        ]
        device = cfg["MODEL"]["device"]

        transform = generate_transform(image_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        # Both datasets (source and target) have to have the same dimension (number of samples)
        assert (
            source_loader.dataset[0].size() == target_loader.dataset[0].size()
        ), "dataset must have the same size"

        loaded_model = load_model(model, device)
        loaded_model.eval()
        feature_extractor = extract_nth_layer_feature(
            loaded_model, feature_extractors_layers
        )

        # Initialize RMSE Loss
        rmse = RMSELoss()

        # Initialize feature weights dictionary => TO DO:
        # Add the features wights dict (layers wights dict) as an input of the function
        feature_weights = {
            node: weight
            for node in feature_extractors_layers
            for weight in feature_extractors_layers_weights
        }
        assert set(feature_weights.keys()) == set(feature_extractors_layers)
        # The keys of the feature weights dict
        # have to be the same as the return nodes specified in the cfg file

        # Get channel info for each layer
        sample = torch.randn(1, 3, image_size[1], image_size[0])  # (N,C,H,W)
        with torch.no_grad():
            output = feature_extractor(sample.to(device))
            shapes = {k: v.size() for k, v in output.items()}

        # Compute source moments
        source_moments = self.__compute_moments(
            source_loader, feature_extractor, k, device, shapes
        )
        target_moments = self.__compute_moments(
            target_loader, feature_extractor, k, device, shapes
        )

        # Compute CMD Loss
        total_loss = 0.0
        for layer_name, weight in feature_weights.items():
            layer_loss = 0.0
            for statistic_order, statistic_weight in enumerate(
                feature_extractors_layers_weights
            ):
                source_moment = source_moments[layer_name][statistic_order]
                taregt_moment = target_moments[layer_name][statistic_order]
                layer_loss += statistic_weight * rmse(source_moment, taregt_moment) / k
            total_loss += weight * layer_loss / len(feature_weights)

        return total_loss.item()


# ========================================================================== #
#                            PROXY-A-DISTANCE                                #
# ========================================================================== #


class ProxyADistance(Metric):
    def __init__(self):
        super().__init__()

    def adapt_format_like_pred(self, y, pred):
        """
        Convert a list of class indices into a one-hot encoded tensor matching the format of the predictions.

        This method takes a list of class indices and converts it into a one-hot encoded tensor that matches the
        shape and format of the provided predictions tensor. This is useful for comparing ground truth labels
        with model predictions in a consistent format.

        Args:
            y (torch.Tensor or list): A 1D tensor or list containing class indices. Each element should be an
                                     integer representing the class index.
            pred (torch.Tensor): A 2D tensor containing predicted probabilities or scores for each class.
                                The shape should be (N, C), where N is the number of samples and C is the
                                number of classes.

        Returns:
            torch.Tensor: A one-hot encoded tensor of the same shape as `pred`, where each row has a 1 at the
                          index of the true class and 0 elsewhere.

        The method performs the following steps:
        1. Initializes a zero tensor with the same shape as `pred`.
        2. Iterates over each class index in `y` and sets the corresponding position in the new tensor to 1.
        """
        # iterate over pred
        new_y_test = torch.zeros_like(pred)
        for i in range(len(y)):
            new_y_test[i][int(y[i])] = 1
        return new_y_test

    def function_pad(self, x, y, error_metric) -> float:
        """
        Computes the PAD (Presentation Attack Detection) value using SVM classifier.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            x_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            dict: A dictionary containing PAD either using MSE or MAE metric.
        """
        c = 1
        kernel = "linear"
        pad_model = svm.SVC(C=c, kernel=kernel, probability=True, verbose=0)
        pad_model.fit(x, y)
        pred = torch.from_numpy(pad_model.predict_proba(x))
        adapt_y_test = self.adapt_format_like_pred(y, pred)

        # Calculate the MSE
        if error_metric == "mse":
            error = F.mse_loss(adapt_y_test, pred)

        # Calculate the MAE
        if error_metric == "mae":
            error = torch.mean(torch.abs(adapt_y_test - pred))
        pad_value = 2.0 * (1 - 2.0 * error)

        return pad_value

    def compute_image_distance(self, cfg: Dict) -> float:
        """
        Compute the average image distance between source and target datasets using multiple models.

        This method calculates the average image distance between features extracted from source and target
        image datasets using multiple pre-trained models. The distance is computed using a specified evaluation
        function for each model, and the average distance across all models is returned.

        Args:
            cfg (Dict): A configuration dictionary containing the following keys:
                - "DATA": Dictionary with data-related configurations:
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.
                    - "batch_size" (int): The batch size for data loading.
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                - "MODEL": Dictionary with model-related configurations:
                    - "arch" (list of str): List of model architectures to use.
                    - "n_layer_feature" (int): The layer number from which to extract features.
                    - "device" (str): The device to run the models on (e.g., "cpu" or "cuda").
                - "METHOD": Dictionary with method-related configurations:
                    - "evaluator" (str): The evaluation function to use for computing the distance.

        Returns:
            float: The computed average image distance between the source and target datasets across all models.

        The method performs the following steps:
        1. Constructs data loaders for the source and target datasets with specified transformations.
        2. Iterates over each model specified in the configuration.
        3. Loads each model and sets it up on the specified device.
        4. Extracts features from the specified layer of the model for both datasets.
        5. Computes the combined features and labels for the source and target datasets.
        6. Calculates the distance using the specified evaluation function.
        7. Returns the average distance across all models.
        """
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]
        batch_size = cfg["DATA"]["batch_size"]
        image_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        models = cfg["MODEL"]["arch"]
        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        device = cfg["MODEL"]["device"]
        evaluator = cfg["METHOD"]["evaluator"]

        transform = generate_transform(image_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        sum_pad = 0
        for model in models:
            loaded_model = load_model(model, device)

            feature_extractor = extract_nth_layer_feature(loaded_model, n_layer_feature)

            source_features = compute_features(source_loader, feature_extractor, device)
            target_features = compute_features(target_loader, feature_extractor, device)

            combined_features = torch.cat((source_features, target_features), dim=0)
            combined_labels = torch.cat(
                (
                    torch.zeros(source_features.size(0)),
                    torch.ones(target_features.size(0)),
                ),
                dim=0,
            )

            # Compute pad
            pad_value = self.function_pad(combined_features, combined_labels, evaluator)

            sum_pad += pad_value

        return sum_pad / len(models)


# ========================================================================== #
#                            Wasserstein_Distance                            #
# ========================================================================== #


class Wasserstein:
    def __init__(self):
        super().__init__()

    def compute_cov_matrix(self, tensor):
        """
        Compute the covariance matrix of a given tensor.

        This method calculates the covariance matrix for a given tensor, which represents a set of feature vectors.
        The covariance matrix provides a measure of how much the dimensions of the feature vectors vary from the mean
        with respect to each other.

        Args:
            tensor (torch.Tensor): A 2D tensor where each row represents a feature vector.
                                  The tensor should have shape (N, D), where N is the number of samples
                                  and D is the dimensionality of the features.

        Returns:
            torch.Tensor: The computed covariance matrix of the feature vectors, with shape (D, D).

        The method performs the following steps:
        1. Computes the mean vector of the feature vectors.
        2. Centers the feature vectors by subtracting the mean vector.
        3. Computes the covariance matrix using the centered feature vectors.
        """
        mean = torch.mean(tensor, dim=0)
        centered_tensor = tensor - mean
        return torch.mm(centered_tensor.t(), centered_tensor) / (tensor.shape[0] - 1)

    def compute_1D_distance(self, cfg):
        """
        Compute the average 1D Wasserstein Distance between corresponding features from source and target datasets.

        This method calculates the average 1D Wasserstein Distance between features extracted from source and target
        image datasets using a pre-trained model. The features are extracted from a specified layer of the model,
        and the distance is computed for each corresponding feature dimension.

        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - "MODEL": Dictionary with model-related configurations:
                    - "arch" (str): The architecture of the model to use.
                    - "device" (str): The device to run the model on (e.g., "cpu" or "cuda").
                    - "n_layer_feature" (int): The layer number from which to extract features.
                - "DATA": Dictionary with data-related configurations:
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                    - "batch_size" (int): The batch size for data loading.
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.

        Returns:
            float: The computed average 1D Wasserstein Distance between the source and target image features.

        The method performs the following steps:
        1. Loads the model and sets it up on the specified device.
        2. Constructs data loaders for the source and target datasets with specified transformations.
        3. Extracts features from the specified layer of the model for both datasets.
        4. Computes the 1D Wasserstein Distance for each corresponding feature dimension.
        5. Returns the average distance across all feature dimensions.
        """
        model = cfg["MODEL"]["arch"]
        device = cfg["MODEL"]["device"]
        loaded_model = load_model(model, device)
        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        image_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        device = cfg["MODEL"]["device"]
        batch_size = cfg["DATA"]["batch_size"]
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]

        transform = generate_transform(image_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        loaded_model = load_model(model, device)
        feature_extractor = extract_nth_layer_feature(loaded_model, n_layer_feature)

        source_features = compute_features(source_loader, feature_extractor, device)
        target_features = compute_features(target_loader, feature_extractor, device)

        sum_wass_distance = 0
        for n in range(min(len(source_features), len(target_features))):
            source_feature_n = source_features[:, n]
            target_feature_n = target_features[:, n]
            sum_wass_distance += wasserstein_distance(
                source_feature_n, target_feature_n
            )
        return sum_wass_distance / len(source_features)

    def compute_slice_wasserstein_distance(self, cfg):
        """
        Compute the Sliced Wasserstein Distance between two sets of image features.

        This method calculates the Sliced Wasserstein Distance between features extracted from source and target
        image datasets using a pre-trained model. The features are projected onto a lower-dimensional space using
        the eigenvectors corresponding to the largest eigenvalues of the covariance matrix. The distance is then
        computed between these projections.

        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - "MODEL": Dictionary with model-related configurations:
                    - "arch" (str): The architecture of the model to use.
                    - "device" (str): The device to run the model on (e.g., "cpu" or "cuda").
                    - "n_layer_feature" (int): The layer number from which to extract features.
                - "DATA": Dictionary with data-related configurations:
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                    - "batch_size" (int): The batch size for data loading.
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.

        Returns:
            float: The computed Sliced Wasserstein Distance between the source and target image features.

        The method performs the following steps:
        1. Loads the model and sets it up on the specified device.
        2. Constructs data loaders for the source and target datasets with specified transformations.
        3. Extracts features from the specified layer of the model for both datasets.
        4. Concatenates the features and computes the covariance matrix.
        5. Computes the eigenvalues and eigenvectors of the covariance matrix.
        6. Projects the features onto a lower-dimensional space using the eigenvectors.
        7. Computes the Sliced Wasserstein Distance between the projected features.
        """
        model = cfg["MODEL"]["arch"]
        device = cfg["MODEL"]["device"]

        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        image_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        batch_size = cfg["DATA"]["batch_size"]
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]

        transform = generate_transform(image_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        loaded_model = load_model(model, device)
        feature_extractor = extract_nth_layer_feature(loaded_model, n_layer_feature)

        source_features = compute_features(source_loader, feature_extractor, device)
        target_features = compute_features(target_loader, feature_extractor, device)

        all_features = torch.concat((source_features, target_features))
        labels = torch.concat(
            (torch.zeros(len(source_features)), torch.ones(len(target_features)))
        )
        cov_matrix = self.compute_cov_matrix(all_features)

        values, vectors = eigh(cov_matrix.detach().numpy())

        # Select the last two eigenvalues and corresponding eigenvectors
        values = values[-2:]  # Get the last two eigenvalues
        vectors = vectors[:, -2:]  # Get the last two eigenvectors
        values, vectors = torch.from_numpy(values), torch.from_numpy(vectors)
        vectors = vectors.T

        new_coordinates = torch.mm(vectors, all_features.T).T
        mask_source = labels == 0
        mask_target = labels == 1

        x0 = new_coordinates[mask_source]
        x1 = new_coordinates[mask_target]

        return ot.sliced_wasserstein_distance(x0, x1)


# ========================================================================== #
#                            Frechet Inception Distance                      #
# ========================================================================== #


class FID(Metric):
    def __init__(self):
        super().__init__()
        self.model = "inception_v3"

    def calculate_statistics(self, features: torch.Tensor):
        """
        Calculate the mean and covariance matrix of a set of features.

        This method computes the mean vector and the covariance matrix for a given set of features.
        It converts the features from a PyTorch tensor to a NumPy array for easier manipulation and
        statistical calculations.

        Args:
            features (torch.Tensor): A 2D tensor where each row represents a feature vector.
                                     The tensor should have shape (N, D), where N is the number of
                                     samples and D is the dimensionality of the features.

        Returns:
            tuple: A tuple containing:
                - mu (numpy.ndarray): The mean vector of the features, with shape (D,).
                - sigma (numpy.ndarray): The covariance matrix of the features, with shape (D, D).

        The function performs the following steps:
        1. Converts the features tensor to a NumPy array for easier manipulation.
        2. Computes the mean vector of the features.
        3. Computes the covariance matrix of the features.
        """
        # Convert features to numpy for easier manipulation
        features_np = features.detach().numpy()

        # Compute the mean and covariance
        mu = np.mean(features_np, axis=0)
        sigma = np.cov(features_np, rowvar=False)

        return mu, sigma

    def compute_image_distance(self, cfg: dict):
        """
        Compute the Frechet Inception Distance (FID) between two sets of images.

        This method calculates the FID between images from a source and target dataset using a pre-trained
        InceptionV3 model to extract features. The FID is a measure of the similarity between two distributions
        of images, commonly used to evaluate the quality of generated images.

        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - "MODEL": Dictionary with model-related configurations:
                    - "device" (str): The device to run the model on (e.g., "cpu" or "cuda").
                    - "n_layer_feature" (int): The layer number from which to extract features.
                - "DATA": Dictionary with data-related configurations:
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                    - "batch_size" (int): The batch size for data loading.
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.

        Returns:
            torch.Tensor: The computed FID score, representing the distance between the source and target image 
            distributions.

        The method performs the following steps:
        1. Loads the InceptionV3 model and sets it up on the specified device.
        2. Constructs data loaders for the source and target datasets with specified transformations.
        3. Extracts features from the specified layer of the model for both datasets.
        4. Calculates the mean and covariance of the features for both datasets.
        5. Computes the FID score using the means and covariances of the features.
        6. Ensures the FID score is positive by taking the absolute value.
        """
        device = cfg["MODEL"]["device"]
        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        img_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        batch_size = cfg["DATA"]["batch_size"]
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]

        transform = generate_transform(img_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        inception_v3 = load_model(self.model, device)
        feature_extractor = extract_nth_layer_feature(inception_v3, n_layer_feature)

        # compute features as tensor
        source_features = compute_features(source_loader, feature_extractor, device)
        target_features = compute_features(target_loader, feature_extractor, device)

        # Calculate statistics for source features
        mu1, sigma1 = self.calculate_statistics(source_features)

        # Calculate statistics for target features
        mu2, sigma2 = self.calculate_statistics(target_features)

        diff = mu1 - mu2

        # Compute the square root of the product of the covariance matrices
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        )

        positive_fid = torch.abs(torch.tensor(fid))
        return positive_fid


# ========================================================================== #
#      Kullback-Leibler divergence for MultiVariate Normal distribution      #
# ========================================================================== #


class KLMVN(Metric):
    """Instanciate KLMVN class to compute KLMVN metrics"""

    def __init__(self):
        super().__init__()

    def calculate_statistics(self, features: torch.Tensor):
        """
        Calculate the mean and covariance matrix of a set of features.

        This function computes the mean vector and the covariance matrix for a given set of features.
        It ensures that the feature matrix has full rank, which is necessary for certain statistical
        operations.

        Args:
            features (torch.Tensor): A 2D tensor where each row represents a feature vector.
                                     The tensor should have shape (N, D), where N is the number of
                                     samples and D is the dimensionality of the features.

        Returns:
            tuple: A tuple containing:
                - mu (torch.Tensor): The mean vector of the features, with shape (D,).
                - sigma (torch.Tensor): The covariance matrix of the features, with shape (D, D).

        Raises:
            AssertionError: If the feature matrix does not have full rank.

        The function performs the following steps:
        1. Computes the mean vector of the features.
        2. Centers the features by subtracting the mean vector.
        3. Computes the covariance matrix using the centered features.
        4. Checks the rank of the feature matrix to ensure it has full rank.
        """
        # Compute the mean of the features
        mu = torch.mean(features, dim=0)

        # Center the features by subtracting the mean
        centered_features = features - mu

        # Compute the covariance matrix (similar to np.cov with rowvar=False)
        # (N - 1) is used for unbiased estimation
        sigma = torch.mm(centered_features.T, centered_features) / (
            features.size(0) - 1
        )

        # Compute the rank of the feature matrix
        rank_feature = torch.linalg.matrix_rank(features)

        # Ensure the feature matrix has full rank
        assert rank_feature == features.size(0), "The feature matrix is not full rank."

        return mu, sigma

    def regularize_covariance(self, cov_matrix, epsilon=1e-6):
        """
        Regularize a covariance matrix by adding a small value to its diagonal elements.

        This function enhances the numerical stability of a covariance matrix by adding a small constant
        to its diagonal. This is particularly useful when the covariance matrix is nearly singular or
        when performing operations that require the matrix to be positive definite.

        Args:
            cov_matrix (numpy.ndarray): The covariance matrix to be regularized. It should be a square matrix.
            epsilon (float, optional): A small value to add to the diagonal elements of the covariance matrix.
                                       Default is 1e-6.

        Returns:
            numpy.ndarray: The regularized covariance matrix with the small value added to its diagonal.

        The function performs the following steps:
        1. Adds the specified `epsilon` value to the diagonal elements of the input covariance matrix.
        2. Returns the modified covariance matrix.
        """
        # Add a small value to the diagonal for numerical stability
        return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

    def klmvn(self, mu1, cov1, mu2, cov2, device):
        """
        Compute the Kullback-Leibler (KL) divergence between two multivariate normal distributions.

        This method calculates the KL divergence between two multivariate normal distributions defined by
        their mean vectors and covariance matrices. It assumes that the covariance matrices are diagonal.

        Args:
            mu1 (torch.Tensor): Mean vector of the first multivariate normal distribution.
            cov1 (torch.Tensor): Diagonal elements of the covariance matrix of the first distribution.
            mu2 (torch.Tensor): Mean vector of the second multivariate normal distribution.
            cov2 (torch.Tensor): Diagonal elements of the covariance matrix of the second distribution.
            device (torch.device): The device (CPU or GPU) on which to perform the computation.

        Returns:
            torch.Tensor: The computed KL divergence between the two distributions.

        The method performs the following steps:
        1. Constructs diagonal covariance matrices from the provided diagonal elements.
        2. Creates multivariate normal distributions using the mean vectors and covariance matrices.
        3. Computes the KL divergence between the two distributions.
        """
        # assume diagonal matrix
        p_cov = torch.eye(len(cov1), device=device) * cov1
        q_cov = torch.eye(len(cov2), device=device) * cov2

        # build pdf
        p = torch.distributions.multivariate_normal.MultivariateNormal(mu1, p_cov)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mu2, q_cov)

        # compute KL Divergence
        kld = torch.distributions.kl_divergence(p, q)
        return kld

    def compute_image_distance(self, cfg: dict) -> float:
        """
        Compute the distance between image features from source and target datasets using a pre-trained model.

        This method calculates the distance between the statistical representations of image features extracted
        from two datasets. It uses a pre-trained model to extract features from specified layers and computes
        the Kullback-Leibler divergence between the distributions of these features.

        Args:
            cfg (dict): A configuration dictionary containing the following keys:
                - "MODEL": Dictionary with model-related configurations:
                    - "device" (str): The device to run the model on (e.g., "cpu" or "cuda").
                    - "arch" (str): The architecture of the model to use.
                    - "n_layer_feature" (int): The layer number from which to extract features.
                - "DATA": Dictionary with data-related configurations:
                    - "width" (int): The width of the images.
                    - "height" (int): The height of the images.
                    - "norm_mean" (list of float): Mean values for image normalization.
                    - "norm_std" (list of float): Standard deviation values for image normalization.
                    - "batch_size" (int): The batch size for data loading.
                    - "source" (str): Path to the source folder containing images.
                    - "target" (str): Path to the target folder containing images.

        Returns:
            float: The computed distance between the source and target image features.

        The method performs the following steps:
        1. Loads the model and sets it up on the specified device.
        2. Constructs data loaders for the source and target datasets with specified transformations.
        3. Extracts features from the specified layer of the model for both datasets.
        4. Calculates the mean and covariance of the features for both datasets.
        5. Regularizes the covariance matrices to ensure numerical stability.
        6. Computes the Kullback-Leibler divergence between the feature distributions.
        """

        device = cfg["MODEL"]["device"]
        model = cfg["MODEL"]["arch"]
        n_layer_feature = cfg["MODEL"]["n_layer_feature"]
        img_size = (cfg["DATA"]["width"], cfg["DATA"]["height"])
        norm_mean = cfg["DATA"]["norm_mean"]
        norm_std = cfg["DATA"]["norm_std"]
        batch_size = cfg["DATA"]["batch_size"]
        source_folder_path = cfg["DATA"]["source"]
        target_folder_path = cfg["DATA"]["target"]

        transform = generate_transform(img_size, norm_mean, norm_std)
        source_loader = construct_dataloader(source_folder_path, transform, batch_size)
        target_loader = construct_dataloader(target_folder_path, transform, batch_size)

        loaded_model = load_model(model, device)
        feature_extractor = extract_nth_layer_feature(loaded_model, n_layer_feature)

        # compute features as tensor
        source_features = compute_features(source_loader, feature_extractor, device)
        target_features = compute_features(target_loader, feature_extractor, device)

        # Calculate statistics for source features
        mu1, cov1 = self.calculate_statistics(source_features)
        cov1 = self.regularize_covariance(cov1)

        # Calculate statistics for target features
        mu2, cov2 = self.calculate_statistics(target_features)
        cov2 = self.regularize_covariance(cov2)

        dist = self.klmvn(mu1, cov1, mu2, cov2, device)
        return dist
