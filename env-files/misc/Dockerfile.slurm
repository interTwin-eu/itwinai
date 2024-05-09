# FROM nvcr.io/nvidia/pytorch:24.04-py3 
FROM nvcr.io/nvidia/pytorch:23.09-py3 

# 23.09-py3: torch==2.1.0
# 24.04-py3: torch==2.3.0

# Install SLURM
RUN apt-get -y update && DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential \
    && cd /root \
    && wget https://download.schedmd.com/slurm/slurm-23.02.7.tar.bz2 \
    && tar -xaf slurm*tar.bz2 && rm slurm*tar.bz2 && cd slurm-23.02.7  \
    && ./configure --prefix /usr/local && make install \
    && ldconfig -n /usr/local/lib && ldconfig -n /usr/local/lib/slurm && ldconfig -n /usr/local/include/slurm
