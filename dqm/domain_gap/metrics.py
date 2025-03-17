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
from torchvision.models.feature_extraction import create_feature_extractor

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
from scipy.linalg import pinv, det


import ot
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dqm.domain_gap.feature_extractor import FeatureExtractor


class Metric:
    def __init__(self) -> None:
        pass

    def compute(self) -> float:
        pass


# ==========================================================================#
#                      MMD - Maximum Mean Discrepancy                       #
# ==========================================================================#
class MMD(Metric):

    def __init__(self) -> None:
        super().__init__()

    def __rbf_kernel(self, x, y, gamma: float) -> float:
        k = torch.cdist(x, y, p=2.0)
        k = -gamma * k
        return torch.exp(k)

    def __polynomial_kernel(
        self, x, y, degree: float, gamma: float, coefficient0: float
    ) -> float:
        k = torch.matmul(x, y) * gamma + coefficient0
        return torch.pow(k, degree)

    @torch.no_grad()
    def compute(self, cfg) -> float:
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
        # iterate over pred
        new_y_test = torch.zeros_like(pred)
        for i in range(len(y)):
            new_y_test[i][int(y[i])] = 1
        return new_y_test

    def function_pad(self, x, y, eval) -> float:
        """
        Computes the PAD (Presentation Attack Detection) value using SVM classifier.

        Args:
            x (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            x_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.

        Returns:
            dict: A dictionary containing PAD, MSE, and MAE values.
        """
        c = 1
        kernel = "linear"
        pad_model = svm.SVC(C=c, kernel=kernel, probability=True, verbose=0)
        pad_model.fit(x, y)
        pred = torch.from_numpy(pad_model.predict_proba(x))
        adapt_y_test = self.adapt_format_like_pred(y, pred)

        # Calculate the MSE
        if eval == "mse":
            error = F.mse_loss(adapt_y_test, pred)
            # pred :  [[0.76 0.24],[0.35 0.65]]
            # y_test : [0,1]
            # y_test :[[1.00 0.00],[0.00 1.00]]

        if eval == "mae":
            error = torch.mean(torch.abs(adapt_y_test - pred))
        pad_value = 2.0 * (1 - 2.0 * error)

        return pad_value

    def compute_image_distance(self, cfg: Dict):
        """
        method which compute Proxy as Distance metrics

        Returns:
            distance : the Proxy as Distance metrics value
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
        eval = cfg["METHOD"]["evaluator"]

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
            pad_value = self.function_pad(combined_features, combined_labels, eval)

            sum_pad += pad_value

        return sum_pad / len(models)


# ========================================================================== #
#                            Wasserstein_Distance                            #
# ========================================================================== #


class Wasserstein:
    def __init__(self):
        super().__init__()

    def compute_cov_matrix(self, tensor):
        mean = torch.mean(tensor, dim=0)
        centered_tensor = tensor - mean
        return torch.mm(centered_tensor.t(), centered_tensor) / (tensor.shape[0] - 1)

    def compute_1D_distance(self, cfg):

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
        # Convert features to numpy for easier manipulation
        features_np = features.detach().numpy()

        # Compute the mean and covariance
        mu = np.mean(features_np, axis=0)
        sigma = np.cov(features_np, rowvar=False)

        return mu, sigma

    def compute_image_distance(self, cfg: dict):
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
    def __init__(self):
        super().__init__()

    def calculate_statistics(self, features: torch.Tensor):
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
        # Add a small value to the diagonal for numerical stability
        return cov_matrix + epsilon * np.eye(cov_matrix.shape[0])

    def klmvn(self, mu1, cov1, mu2, cov2, device):
        # assume diagonal matrix
        p_cov = torch.eye(len(cov1), device=device) * cov1
        q_cov = torch.eye(len(cov2), device=device) * cov2

        # build pdf
        p = torch.distributions.multivariate_normal.MultivariateNormal(mu1, p_cov)
        q = torch.distributions.multivariate_normal.MultivariateNormal(mu2, q_cov)

        # compute KL Divergence
        kld = torch.distributions.kl_divergence(p, q)
        return kld

    def compute_image_distance(self, cfg: dict):
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
