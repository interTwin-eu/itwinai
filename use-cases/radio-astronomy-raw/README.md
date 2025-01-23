# PulsarRFI_NN

This repository holds different implementations for recognition of pulsars. The first concept is semantic segmentation using a UNet, 
source code and examples can be found in the subdirectory [unet_semantic_segmentation](unet_semantic_segmentation). 
Another concept uses a Convolutional Neural Network which can be inspected in the subdirectory [cnn](cnn).

## Setup to run it locally
Following these steps will enable you to install all necessary dependencies as well as try the components of PulsarRFI_NN in an virtual environment. The project has been *tried and tested on Ubuntu 22.04 (x86_64)*, if you are using any other operating system inform yourself how to install the necessary dependencies. 

For Ubuntu systems, please install the following dependencies:

```
sudo apt update
sudo apt install git python3.10-venv build-essential python3-dev
```

The package can be cloned from the gitlab repository as:

```
git clone https://gitlab.com/ml-ppa/pulsarrfi_nn.git
cd pulsarrfi_nn
```

From this point you can navigate into the subdirectories [unet_semantic_segmentation](unet_semantic_segmentation) and [cnn](cnn) and follow the README files you find there for further instructions.