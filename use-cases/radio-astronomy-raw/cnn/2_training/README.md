# Training

## Structure:
+ `checkpoints/`

> This directory is used to store the checkpoint files generated during model training. These files can be used to resume training or for model evaluation.
+ `datasets/`

> This directory is used to store datasets. The datasets are used for training, validating, and testing the model.
+ `images/`

> This directory is used to store images, likely related to the evaluation of the models such as plots and graphs illustrating the model performance.
+ `config.py`

> This file is used to store configurations such as paths to other files and folders.
+ `models.py`

> This file defines various TensorFlow models to be used for training.
+ `tf_training.py`

> This file contains the primary code to execute model training using TensorFlow tools on the datasets located in the datasets/ directory, with models defined in models.py.

+ `tf&horovod_training.py`

> This file contains the primary code to execute model training using TensorFlow tools and Horovod (for distributed GPU training) on the datasets located in the datasets/ directory, with models defined in models.py.

### Usage

Prepare your dataset and place it in the datasets/ directory, or adjust the `PATH2FILES` variable in `config.py` to point to your dataset location.

Choose the appropriate model in models.py or define a new one if needed.

Run the `tf_training.py` file, specifying the resolution and the model name as command-line arguments:

```bash
python tf_training.py [RESOLUTION] [MODEL_NAME]
```

Replace `[RESOLUTION]` with the resolution you want to train your model with and `[MODEL_NAME]` with the name of the model you want to use, which should be one of the keys in the models_htable dictionary defined in `models.py`.

### Example
```bash
python tf_training.py 128 PROT_LE_1_230908_0
```

### TensorFlow and Horovod Usage
All of the above is also true for running `tf&horovod_training.py` 

## Models
The `models.py` file contains multiple model architectures (e.g., `model_PROT_LE_1_230908_0`, `model_PROT_LE_1_230908_1`, etc.) designed for various use cases. Choose the one that best fits your needs or define a new one following the existing pattern.

## Results and Evaluation
After running the training script, evaluation plots will be saved in the images/ directory. Check the generated plots for insights into model performance, such as loss and accuracy over epochs, and adjust your model or training parameters as needed.
