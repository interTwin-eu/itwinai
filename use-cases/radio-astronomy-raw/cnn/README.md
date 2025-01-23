# PulsarRFI_NN

## Overview
This repository is a meticulously organized collection of advanced Python Jupyter notebooks, utility scripts, machine learning models, and associated assets, meticulously designed to perform comprehensive analysis and evaluation on various datasets, primarily focusing on signal and radio-frequency interference (RFI) data. It encapsulates a broad spectrum of functionalities including data creation, classification, model training, and performance evaluation, presented through a coherent and user-friendly interface.

## Setup to run it locally
We recommend using virtual environment for running this package. This will help organize dependencies and avoid conflics with packages you have installed globally on your computer. You can create a virtual environment with the following command:
```
python3 -m venv <venv_name>
``` 
followed by activating the virtual environment as:
```
source <venv_name>/bin/activate
```
If you have activated the virtual environment, you should see the name <venv_name> at the beginning of the command prompt in your terminal. The required packages or dependencies then can be installed in the virtual environment as:
```
pip install -r requirements.txt
```
Please step through the subdirectories data_generation, training and models_inspection to find more instructions there. You will be led through the generation of data which is needed to train a neural network, train the actual network and at the end evaluate its accuracy regarding the detection of pulsars. Below you find more details of the individual components, please read them before proceeding.

## Components:
1. Data Creator Class:
A robust Python class, DataCreator, is designed to create intricate datasets consisting of varied signals and RFIs. It encompasses various methods to manipulate and transform data, plot instances, and incorporate realistic elements like noise, allowing the creation of highly customizable and versatile datasets for exhaustive analysis.

2. Model Definitions:
In the models.py file, several Convolutional Neural Network (CNN) models are defined, each with unique architectures and layer configurations, allowing users to explore and compare different model structures.

3. TensorFlow Training Script:
tf_training.py script orchestrates the training process of the models, incorporating advanced techniques like custom callbacks for checkpointing, comprehensive metrics evaluation, and detailed logging. It also plots the learning curves, giving visual insights into the model's learning process.

4. Jupyter Notebooks:

> real_data_classification.ipynb:
> This notebook loads pre-trained models to classify new datasets and logs each model's predictions, serving as a tool for evaluating model generalization on unseen data.

> test_labeling_accuracy.ipynb:
> This notebook compares model predictions to actual labels, calculates classification metrics, and visually represents the results, providing insight into model performance and reliability.

> Pulse_analysis.ipynb:
> This notebook evaluates models' accuracy on spectral data with varying pulse parameters like Signal-to-Noise Ratio (SNR), Dispersion Measure (DM), and pulse width, outlining the models' adaptability and limitations in diverse conditions.

> NBRFI_analysis.ipynb:
> This focuses on the evaluation of models related to Narrowband RFI (NBRFI), exploring their performance under different conditions of intensity, number of RFIs, and RFI width, enriching understanding of model capabilities and constraints with different RFI parameters.

> BBRFI_analysis.ipynb:
> This notebook concentrates on Broadband RFI (BBRFI) analysis, assessing model responses to varying intensity levels, numbers of RFIs, and RFI width, shedding light on model adaptability and proficiency in varied interference scenarios.

5. Configuration Files and Utility Scripts:
Essential path configurations, constants, and utility functions are encapsulated in files like config.py for seamless access and modifications across the repository.

6. Directory Structure:
The repository is structured with dedicated directories for checkpoints, images, and datasets, ensuring organized storage and easy retrieval of models, visualizations, and data files respectively.

## Functionality:
The synergistic integration of the components allows for a seamless flow from data creation to model evaluation:

## Data Creation and Pre-processing:
Leveraging the DataCreator class for generating datasets imbued with realistic attributes and subsequent data transformations and normalizations.

## Model Training and Optimization:
Employing diverse model architectures for training on the created datasets, optimizing their performances, and saving the best models.

## Model Evaluation and Analysis:
Using the models to make predictions on new datasets and rigorously analyzing the predictions through various metrics, confusion matrices, and visual representations using the provided Jupyter notebooks.

## Result Visualization:
Rendering detailed plots and visual insights into the models’ performance and learning, aiding in intuitive understanding and further analysis.

## Usability:
The repository is designed with user-centricity, offering extensive configurability, detailed documentation, and intuitive interfaces, making it suitable for both novice and experienced users in fields like signal processing, machine learning, and data analysis. Whether it is generating datasets with specific characteristics, exploring diverse model architectures, or conducting in-depth performance evaluations, this repository provides the necessary tools and frameworks.

## Conclusion:
This comprehensive repository serves as an extensive resource for researchers, data scientists, and enthusiasts, offering advanced tools and methods to explore, analyze, and evaluate models in the domain of signal and radio-frequency interference data. Its modular and cohesive structure, coupled with detailed documentation, makes it an invaluable asset for learning, experimentation, and advancement in the field.