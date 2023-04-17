import sys
import subprocess

try:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", 'ppppppppp'])
except subprocess.CalledProcessError as err:
    print('something went wrong..')
    print(err)
print("asds")


# import os
# from mamba.api import install

# env_name = os.path.join(os.getcwd(), './.venv')
# install(env_name,('matplotlib=3', 'ipympl'), ('conda-forge', ))
