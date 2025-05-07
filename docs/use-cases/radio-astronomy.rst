Pulsar Segmentation and Analysis for Radio-Astronomy (HTW Berlin)
===============================================================================================
The code is adapted from 
`this repository <https://gitlab.com/ml-ppa/pulsarrfi_nn/-/tree/version_0.2/unet_semantic_segmentation?ref_type=heads>`_.
Please visit the original repository for more technical information on the code. 
This use case features a sophisticated pipeline composed of few neural networks.

Integration Author: Oleksandr Krochak, FZJ

Environment Management
-----------------------------------------------------------------------------------------------
It is recommended to use the UV environment for running this pipeline. 
The overview of itwinai-wide module dependencies can be found in `intertwin/pyproject.toml`.
By running `uv sync --extra devel --extra torch --extra radio-astronomy`, the uv lockfile will 
be generated/updated that ensures that correct dependencies are installed. If you want to 
change some use-case specific dependencies, please do so in pyproject.toml in the radio-astronomy
section. Afterwards, re-run `uv sync` with the same flags.

Alternatively, you can install the required dependencies from the use-case directory:
`pip install requirements.txt`

Running from a configuration file
-----------------------------------------------------------------------------------------------
You can run the full pipeline sequence by executing the following commands locally. 
itwinai will read these commands from the `config.yaml` file in the root of the repository.
1. Generate the synthetic data            - `itwinai exec-pipeline +pipe_key=syndata_pipeline`
2. Initialize and train a UNet model      - `itwinai exec-pipeline +pipe_key=unet_pipeline`
3. Initialize and train a FilterCNN model - `itwinai exec-pipeline +pipe_key=fcnn_pipeline`
4. Initialize and train a CNN1D model     - `itwinai exec-pipeline +pipe_key=cnn1d_pipeline`
5. Compile a full pipeline and test it    - `itwinai exec-pipeline +pipe_key=evaluate_pipeline`

When running on HPC, you can use the `batch.sh` SLURM script to run these commands.

Logging with MLflow
-----------------------------------------------------------------------------------------------
By default, the `config.yaml` ensures that the MLflow logging is enabled during the training.
During or after the run, you can launch an MLflow server by executing
`mlflow server --backend-store-uri mllogs/mlflow` and connecting to `http://127.0.0.1:5000/` 
in your browser.

Test suite
-----------------------------------------------------------------------------------------------
The test suite is located in the `tests/use-cases/radio-astronomy` folder. 

Before running the test suite, you should make sure that the pytorch fixture in:
`tests/use-cases/radio-astronomy/test_radio-astronomy.py`:torch_env()  
is correctly defined and corresponds to the virtual environment where itwinai is installed on 
your system. 

It contains integration tests for each of the pipelines 1-5 mentioned above. The configuration
and execution of the test suite is defined in: 
`tests/use-cases/radio-astronomy/test_radio-astronomy.py` 
and in the configuration file in the use-case repository:
`use-cases/radio-astronomy/.config-test.yaml`. 
If you are updating the test suite, make sure you update both of these files. 

Feel free to change the pytest markers as needed, but be careful with pushing these changes. 
Tests should be able to run in an isolated environment. 
