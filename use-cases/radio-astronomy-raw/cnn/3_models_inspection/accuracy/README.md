# Models accurasy analysis

## Description

These two notebooks synergistically collaborate to fulfill the pipeline of classifying and evaluating signal and RFI data, applying pre-trained models to unseen datasets and subsequently conducting exhaustive analyses of the classifications made against the true labels.

+ real_data_classification.ipynb:
This notebook is instrumental in applying various pre-trained models to a new dataset. It methodically loads each model and classifies the dataset, creating comprehensive tables of the model's predicted labels. It meticulously iterates over the dataset, leveraging TensorFlow and Panda libraries to manipulate and store the data efficiently, preparing a systematic ground for further analysis.

+ test_labeling_accuracy.ipynb:
Building upon the classifications made, this notebook delves deep into analytical exploration, computing extensive metrics and creating visual representations to assess the models’ performance. It utilizes advanced statistical and visualization libraries to compare predicted labels against actual ones, drawing confusion matrices and plotting detailed graphs to offer insightful perspectives into the models’ accuracy, precision, recall, and F1 score for each class, enabling a nuanced understanding of model efficacy and areas of improvement.
This repository contains two main Jupyter notebooks:

## real_data_classification.ipynb
In this notebook:

A set of models are loaded from a specified path.
Each model is used to make predictions on a test dataset.
The predictions made by each model are saved to a CSV file.


## test_labeling_accuracy.ipynb
This notebook is used to:

Compare models' predictions against actual labels.
Calculate various classification metrics, like precision, recall, and F1-score, both on a per-class basis and overall.
Visualize the metrics and confusion matrices for every model.

# Results
After running both notebooks, the model's predictions on the test dataset are evaluated, and various metrics and visualizations are generated to understand the model's performance better.