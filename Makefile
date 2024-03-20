# Install PyTorch env (GPU support)
torch-gpu: env-files/torch/pytorch-env-gpu.yml
	micromamba env create -p ./.venv-pytorch --file env-files/torch/pytorch-env-gpu.yml -y
	micromamba run -p ./.venv-pytorch python -m pip install -e .

# Install PyTorch env (CPU only)
torch-cpu: env-files/torch/pytorch-env-cpu.yml
	micromamba env create -p ./.venv-pytorch --file env-files/torch/pytorch-env-cpu.yml -y
	micromamba run -p ./.venv-pytorch python -m pip install -e .


# Install TensorFlow 2.13. Creates ./.venv-tf folder.
# Ref: https://www.tensorflow.org/install/pip#step-by-step_instructions
tf-2.13: env-files/tensorflow/tensorflow-2.13.yml
	echo "Installing TensorFlow 2.13 env"
	micromamba env create -p ./.venv-tf --file env-files/tensorflow/tensorflow-2.13.yml -y
	micromamba run -p ./.venv-tf pip install nvidia-cudnn-cu11==8.6.0.163
	
	mkdir -p ./.venv-tf/etc/conda/activate.d
	echo 'CUDNN_PATH=$$(dirname $$(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	echo 'export LD_LIBRARY_PATH=$$CUDNN_PATH/lib:$$CONDA_PREFIX/lib/:$$LD_LIBRARY_PATH' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$$CONDA_PREFIX/lib/\n' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	
	micromamba run -p ./.venv-tf pip install --upgrade pip
	micromamba run -p ./.venv-tf pip install tensorflow==2.13.*
	micromamba run -p ./.venv-tf pip install -e .

	mkdir -p ./.venv-tf/lib/nvvm/libdevice
	cp ./.venv-tf/lib/libdevice.10.bc ./.venv-tf/lib/nvvm/libdevice/

# Install TensorFlow 2.11. Creates ./.venv-tf folder.
# Ref: https://skeptric.com/tensorflow-conda/
tf-2.11: env-files/tensorflow/tensorflow-2.11.yml
	micromamba env create -p ./.venv-tf --file env-files/tensorflow/tensorflow-2.11.yml -y
	@# Env variables
	mkdir -p ./.venv-tf/etc/conda/activate.d
	micromamba run -p ./.venv-tf echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$CONDA_PREFIX/lib/' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	micromamba run -p ./.venv-tf echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$$CONDA_PREFIX/lib/' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	@# Add library
	mkdir -p ./.venv-tf/lib/nvvm/libdevice/
	cp ./.venv-tf/lib/libdevice.10.bc ./.venv-tf/lib/nvvm/libdevice/
	micromamba run -p ./.venv-tf pip install --upgrade pip
	micromamba run -p ./.venv-tf pip install tensorflow==2.11.0
	micromamba run -p ./.venv-tf pip install -e .

# Install TensorFlow 2.10. Creates ./.venv-tf folder.
# Ref: https://phoenixnap.com/kb/how-to-install-tensorflow-ubuntu
tf-2.10: env-files/tensorflow/tensorflow-2.10.yml
	micromamba env create -p ./.venv-tf --file env-files/tensorflow/tensorflow-2.10.yml -y
	mkdir -p ./.venv-tf/etc/conda/activate.d
	micromamba run -p ./.venv-tf echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:$$CONDA_PREFIX/lib/' >> ./.venv-tf/etc/conda/activate.d/env_vars.sh
	micromamba run -p ./.venv-tf pip install --upgrade pip
	micromamba run -p ./.venv-tf pip install tensorflow==2.10
	micromamba run -p ./.venv-tf pip install -e .


tf-2.13-cpu: env-files/tensorflow/tensorflow-2.13-cpu.yml
	echo "Installing TensorFlow 2.13 env"
	micromamba env create -p ./.venv-tf --file env-files/tensorflow/tensorflow-2.13-cpu.yml -y
	micromamba run -p ./.venv-tf pip install --upgrade pip
	micromamba run -p ./.venv-tf pip install tensorflow==2.13.*
	micromamba run -p ./.venv-tf pip install -e .

	mkdir -p ./.venv-tf/lib/nvvm/libdevice
	cp ./.venv-tf/lib/libdevice.10.bc ./.venv-tf/lib/nvvm/libdevice/