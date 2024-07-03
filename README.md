# Diffusion Model PyTorch

Implements a Diffusion Model in PyTorch on the Oxford 102 Flower dataset (https://www.kaggle.com/datasets/nunenuh/pytorch-challange-flower-dataset). 

### Code:
The main code is located in the following files:
* main.py - main entry file for training the network
* train.py - training code
* diffusion_model.py - implements the diffusion model
* diffusion_schedules.py - implements an offset cosine diffusion schedule for training and generating
* model_building_blocks.py - residual block, Up Block, Down block and sinusoidal embedding to use in the network
* unet_model.py - implements the UNet network for use in the diffusion model
* generate_images.py - Function to generate and plot images whilst training
* display.py - helper function to plot images
* lint.sh - runs linters on the code
