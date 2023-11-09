FROM python:3.9.12

WORKDIR /usr/src/app

RUN pip install --upgrade pip

# Install pytorch (cpuonly)
# Ref:https://pytorch.org/get-started/previous-versions/#linux-and-windows-5
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir lightning

# Add 3DGAN custom requirements
COPY use-cases/3dgan/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install itwinai and dependencies
COPY pyproject.toml ./
COPY src ./
RUN  pip install --no-cache-dir .

# Add 3DGAN use case files
COPY use-cases/3dgan/* ./

# Run inference
CMD [ "python", "train.py", "-p", "inference-pipeline.yaml"]