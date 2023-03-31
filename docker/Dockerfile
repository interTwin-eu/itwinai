# A simple container to test the containarization of the app
# Drawback: very large image 


# FROM continuumio/miniconda:latest
FROM condaforge/mambaforge:latest

COPY environment.yml .
RUN mamba env create -n itwin-ai --file environment.yml

# Install app from source code
COPY . /app/
RUN conda run -n itwin-ai python -m pip install --no-deps /app

CMD [ \
    "mamba", "run", "-n", "itwin-ai", \
    "python", "--version" \
    ]