# Install PyTorch env (GPU support)
torch-env: env-files/torch/generic_torch.sh
	env ENV_NAME=.venv-pytorch \
		bash -c 'bash env-files/torch/generic_torch.sh'
	.venv-pytorch/bin/horovodrun --check-build 

# Install PyTorch env (without GPU support: Horovod has not NCCL support)
torch-env-cpu: env-files/torch/generic_torch.sh
	env ENV_NAME=.venv-pytorch \
		NO_CUDA=1 \
		bash -c 'bash env-files/torch/generic_torch.sh'
	.venv-pytorch/bin/horovodrun --check-build 

# Install TensorFlow env (GPU support)
tensorflow-env: env-files/tensorflow/generic_tf.sh
	env ENV_NAME=.venv-tf \
		bash -c 'bash env-files/tensorflow/generic_tf.sh'
	@#.venv-tf/bin/horovodrun --check-build

# Install TensorFlow env (without GPU support: Horovod has not NCCL support)
tensorflow-env-cpu: env-files/tensorflow/generic_tf.sh
	env ENV_NAME=.venv-tf \
		NO_CUDA=1 \
		bash -c 'bash env-files/tensorflow/generic_tf.sh'
	@#.venv-tf/bin/horovodrun --check-build

test:
	.venv-pytorch/bin/pytest -v tests/ -m "not slurm"

test-jsc: tests/run_on_jsc.sh
	bash tests/run_on_jsc.sh

torch-gpu-mamba: env-files/torch/pytorch-env-gpu.yml
	micromamba env create -p ./.venv-pytorch --file env-files/torch/pytorch-env-gpu.yml -y
	micromamba run -p ./.venv-pytorch python -m pip install -e .[dev]

# Install PyTorch env (GPU support) on Juelich Super Computer (tested on HDFML system)
torch-gpu-jsc: env-files/torch/createEnvJSC.sh env-files/torch/generic_torch.sh
	sh env-files/torch/createEnvJSC.sh

# Install Tensorflow env (GPU support) on Juelich Super Computer (tested on HDFML system)
tf-gpu-jsc: env-files/tensorflow/createEnvJSCTF.sh env-files/tensorflow/generic_tf.sh
	sh env-files/tensorflow/createEnvJSCTF.sh

# Install PyTorch env (CPU only)
torch-cpu-mamba: env-files/torch/pytorch-env-cpu.yml
	micromamba env create -p ./.venv-pytorch --file env-files/torch/pytorch-env-cpu.yml -y
	micromamba run -p ./.venv-pytorch python -m pip install -e .[dev]

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

# Install TensorFlow 2.13 for CPU only systems. Creates ./.venv-tf folder.
tf-2.13-cpu: env-files/tensorflow/tensorflow-2.13-cpu.yml
	echo "Installing TensorFlow 2.13 env"
	micromamba env create -p ./.venv-tf --file env-files/tensorflow/tensorflow-2.13-cpu.yml -y
	micromamba run -p ./.venv-tf pip install --upgrade pip
	micromamba run -p ./.venv-tf pip install tensorflow==2.13.*
	micromamba run -p ./.venv-tf pip install -e .

# # Install PyTorch env (GPU support)
# torch-env2:
# 	python3 -m venv .venv-pytorch
# 	.venv-pytorch/bin/pip install -e .[dev,torch]
# 	@# Install horovod AFTER torch
# 	@# https://github.com/horovod/horovod/pull/3998
# 	env HOROVOD_CPU_OPERATIONS=MPI \
# 		HOROVOD_GPU_ALLREDUCE=NCCL \
# 		HOROVOD_NCCL_LINK=SHARED \
# 		HOROVOD_NCCL_HOME=$$EBROOTNCCL \
# 		HOROVOD_WITH_PYTORCH=1 \
# 		HOROVOD_WITHOUT_TENSORFLOW=1 \
# 		HOROVOD_WITHOUT_MXNET=1 \
# 		bash -c '.venv-pytorch/bin/pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17'
# 	.venv-pytorch/bin/horovodrun --check-build 

# # Install PyTorch env (Horovod has not NCCL support)
# torch-env-cpu:
# 	python3 -m venv .venv-pytorch
# 	.venv-pytorch/bin/pip install -e .[dev,torch-cpu]
# 	@# Install horovod AFTER torch
# 	@# https://github.com/horovod/horovod/pull/3998
# 	env HOROVOD_WITH_PYTORCH=1 \
# 		HOROVOD_WITHOUT_TENSORFLOW=1 \
# 		HOROVOD_WITHOUT_MXNET=1 \
# 		bash -c '.venv-pytorch/bin/pip install --no-cache-dir git+https://github.com/thomas-bouvier/horovod.git@compile-cpp17'
# 	.venv-pytorch/bin/horovodrun --check-build


# # Install PyTorch env (without GPU support: Horovod has not NCCL support)
# torch-env-cpu: 
# 	env ENV_NAME=.venv-pytorch \
# 		NO_CUDA=1 \
# 		bash -c 'bash env-files/torch/generic_torch.sh'
# 	.venv-pytorch/bin/horovodrun --check-build 

# # Install TensorFlow env (GPU support)
# tensorflow-env:
# 	python3 -m venv .venv-tf
# 	.venv-tf/bin/pip install -e .[dev,tensorflow]
# 	env HOROVOD_GPU=CUDA \
#   		HOROVOD_GPU_OPERATIONS=NCCL \
#   		HOROVOD_WITH_TENSORFLOW=1 \
# 		bash -c '.venv-tf/bin/pip install --no-cache-dir horovod[tensorflow,keras]'