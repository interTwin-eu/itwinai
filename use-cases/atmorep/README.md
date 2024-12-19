#Installing instructions

step-by-step guide for the installation
1. git clone --recurse-submodules https://github.com/interTwin-eu/itwinai.git
2. cd itwinai & git checkout --track origin/iluise-atmorep
3. python -m venv pyenv & source pyenv/bin/activate
4. cd use-cases/atmorep
5. load modules
````
ml --force purge
ml Stages/2024 GCC OpenMPI CUDA/12 cuDNN MPI-settings/CUDA
ml Python CMake HDF5 PnetCDF libaio mpi4py
````
5. install packages
````
pip install -e ".[torch,dev,nvidia,tf]" \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
````