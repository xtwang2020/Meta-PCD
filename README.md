# Meta-PCD
This is our implementation of Meta-PCD which is a meta-network-based method for point cloud denoising.
![Meta-PCD](https://user-images.githubusercontent.com/95417188/229996803-080b2fcd-760b-4402-8b3d-de949cb2b8c4.jpg)

# Prerequisites
CUDA and CuDNN (changing the code to run on CPU should require few changes)

Python 2.7

PyTorch 1.0


# Setup
Install required python packages, if they are not already installed (tensorboardX is only required for training):

pip install numpy

pip install scipy

pip install tensorboardX

# Clone this repository:

git clone https://github.com/xtwang2020/Meta-PCD.git


# Denoising
To denoise point clouds using default settings and calculate the metric:

cd Meta-PCD
python test.py
python metric_cal.py

# Acknowledgements
Part of this implementation is based on PointCleanNet and PCPNet.
