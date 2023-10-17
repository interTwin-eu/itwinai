nname='torch_env'

# create env inside container
# python3 -m venv $nname --system-site-packages
# source ${nname}/bin/activate

# install wheels -- from this point on, feel free to add anything
pip3 install -r ../../env-files/torch/pytorch-env-gpu-container.txt
pip3 install -e ../../