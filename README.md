# Generative Adversarial Network in YUV Color Space for Thin Cloud Removal
This repository is an implementation of "Generative Adversarial Learning in YUV Color Space for Thin Cloud Removal on Satellite Imagery", in Remote Sensing 2021.
# Requirements
- Python3 (tested with 3.6) 

- Tensorflow (tested with 1.9)

- cuda (tested with 9.0)

- cudnn (tested with 7.5)

- OpenCV

- tflearn

- matplotlib

# Preparing the data
- The complete RICE_DATASET for Cloud Removal can be downloaded from: https://github.com/BUPTLdy/RICE_DATASET

- Sentinel-2A images can be downloaded on the SENTINEL Hub website：
https://apps.sentinel-hub.com/sentinel-playground/

- An open-source noise generation tool FastNoise Lite (https://github.com/Auburn/FastNoiseLite) is uesd for the simulation of cloud layers.

# Training examples
At first, you should configure the associated hyperparameters and file paths in config.py.Then, if you set the model name as "WGAN_", execute the following code：

python cloud_generation.py --checkpoint_gan=model/WGAN_

# Testing examples
python test.py  --model_path=model/CloudRemovalGAN_WGAN_

# License
Academic use only.
