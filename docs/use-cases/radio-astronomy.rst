Pulsar Segmentation and Analysis for Radio-Astronomy (HTW Berlin)
===============================================================
The code is adapted from
`this repository <https://gitlab.com/ml-ppa/pulsarrfi_nn/-/tree/version_0.2/unet_semantic_segmentation?ref_type=heads>`_.
Please visit the original repository for more technical information on the code. 
This use-case features a sophisticated pipeline composed of few neural networks.

Environment Management
-----------------------------------------------------------------
It is recommended to use UV environment for running this pipeline. 
The overview of itwinai-wide module dependencies can be found in `intertwin/pyproject.toml`.
By running `uv sync --extra devel --extra torch --extra radio-astronomy`, uv lockfile will 
be generated/updated that ensures that correct dependencies are installed. If want to 
change some use-case specific dependencies, please do so in pyproject.toml in the radio-astronomy
section. Afterwards, re-run `uv sync` with the same flags.

Running from a configuration file
----------------------------------
You can run the full pipeline sequency by executing the following commands locally. 
Itwinai will read these commands from the `config.yaml` file in the root of the repository.
1. Generate the synthetic data            - `itwinai exec-pipeline +pipe_key=syndata_pipeline`
2. Initialize and train a UNet model      - `itwinai exec-pipeline +pipe_key=unet_pipeline`
3. Initialize and train a FilterCNN model - `itwinai exec-pipeline +pipe_key=fcnn_pipeline`
4. Initialize and train a CNN1D model     - `itwinai exec-pipeline +pipe_key=cnn1d_pipeline`
5. Compile a full pipeline and test it    - `itwinai exec-pipeline +pipe_key=evaluate_pipeline`

When running on HPC, you can use the `batch.sh` SLURM script to run these commands.

Logging with MLflow
----------------------------------
By default, the `config.yaml` ensures that the MLflow logging is enabled during the training.
During or after the run, you can launch an MLflow server by executing
`mlflow server --backend-store-uri mllogs/mlflow` and connecting to `http://127.0.0.1:5000/` 
in your browser.