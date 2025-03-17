# Model-Based Metrics for Domain Gap Estimation in Image Datasets

## Introduction
In the context of machine learning, the domain gap refers to the discrepancy between the source and target domains, especially in visual recognition tasks where models trained on one dataset may not generalize well to another. This gap can arise due to differences in data distributions, such as variations in lighting conditions, image resolution, camera angles, object occlusion, and background noise.

## Quick start

### source and target domain inputs
Source and target datasets have to be provided as two csv files in the Json configuration file. Image paths must be specified in the `path` column of the csv files.

### How to use our library
We proposed 2 approaches to use our implemented metrics. The first appraoch consists of importing the desired metrics class and use it a notebook or script in your python code. The second approach is to use a metric through CLI, with the main.py script : `python main.py --cfg cfg.json`. Note that a collection of config files is available in "cfg" folder.

<!--- 
### More than two datasets to measure (Will be refactored/ work only on CMD)
You can provide with the `--tsk` command line argument a task list that contain a list of data, a list of features and a list of metric to compute. See the `cfg/task_list.json` example 
--->


### Model

Model backbone and (optionally) associated feature extraction can be provided.
Currently every models available on torch can be used as feature extraction, their default weights are based on Imagenet Dataset.  
!! WARNING !! Each metric need specific input to compute, we recommand to use default configuration file to avoid conflict between feature extraction model and metric.


### Metrics
Our current domain gap metric collection: 
* Central Mean Discrepancy of order k
* Kullback Leiblur - divergence with features modelized as a MultiVariate Normal distribution with diagonal covariance.
* Maximum Mean Discrepancy
* Wasserstein Distance 1D 
* Sliced Wasserstein distance
* Proxy A Distance
* Frechet Inception Distance

### Quick Overview:
#### CMD:
Metric which captures high-order central moments statistics beyond the mean (1st order) and variance (2nd order) like skewness (3rd order) or kurtosis (4th order) and measure the discrepancies considering 2 distributions.
#### KLMVN
Measure the shape and entropy between two multivariate normal distributions, estimating how a distribution diverge from another.

#### MMD (Maximum Mean Discrepancy) Class

##### Overview
The MMD (Maximum Mean Discrepancy) Python module is designed for statistical analysis, 
particularly in computing the discrepancy between two data distributions. This module 
provides a flexible and efficient way to compare distributions using PyTorch tensors, 
making it ideal for applications in machine learning and data science.

The module offers support for various kernel functions including linear, 
Radial Basis Function (RBF), and polynomial kernels. 
This versatility allows users to choose the most appropriate kernel for their specific data analysis task.

##### Features
Kernel Flexibility: Supports linear, RBF, and polynomial kernels.
Device Compatibility: Works with computations on both CPU and GPU devices.
PyTorch Integration: Seamlessly integrates with PyTorch tensors for efficient computation.

##### Dependencies
torch: Required for tensor operations and computations.
twe_logger: A custom logger for logging events and errors 
(must be available in the PYTHONPATH or in the same directory).

##### Usage
- Initialization
First, create an instance of the MMD class by specifying the desired kernel and computing device.

	from mmd import MMD

	mmd_computer = MMD(kernel='rbf', device='cpu')
	
-  Computing MMD
To compute the MMD between two samples, use the compute method with your data tensors.

	import torch

	# Sample tensors
	x = torch.rand(10, 5)  # Sample tensor x
	y = torch.rand(10, 5)  # Sample tensor y

# MMD computation
	mmd_value = mmd_computer.compute(x, y)
	print("MMD Value:", mmd_value)

##### Class Documentation
- class MMD
A class for computing the Maximum Mean Discrepancy (MMD) between two samples. 
MMD is a measure of the difference between two probability distributions based on samples from these distributions.
  -  Attributes
     - kernel (str): The type of kernel used for computation (options: 'linear', 'rbf', 'poly').
     - device (str): The computing device ('cpu' or 'cuda').

  - Methods
    - __init__(self, kernel, device='cpu'): Initializes the MMD object.
    - kernel(self): Getter for the kernel attribute.
    - kernel(self, value): Setter for the kernel attribute.
    - device(self): Getter for the device attribute.
    - device(self, value): Setter for the device attribute.
    - __rbf_kernel(self, x, y, gamma): Private method to compute the RBF kernel.
    - __polynomial_kernel(self, x, y, params): Private method to compute the polynomial kernel.
    - compute(self, x, y, **kwargs): Computes the MMD value between two samples.
<!---
## License
This MMD module is open-sourced and freely available for modification and distribution, 
keeping in mind the credits to the original authors
--->
#### Wasserstein1D
Estimate a dsitance using Earth Mover distance algorithm: how much 2 normal distribution are differents considering distributions means.
#### SWD
Compute a distance between two distribution considering only the two most different embeddings features components which have highest eigen values.
#### PAD
Evaluate how well a proxy classifier can distinguish between 2 distributions embeddings by returning the mean of the proxy classifer performance while it  discriminate between two distribution features generated with multiple features extractor model.
#### FID 
Measurement using mean and covariance between 2 distributions Inceptionv3 embeddings. 
