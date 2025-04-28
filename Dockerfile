
FROM ubuntu:20.04

COPY . /app/

# prepare system packages, libgomp1 is required by elastix
# adapted from Superelastix/elastix Dockerfile

RUN apt-get update && apt-get -qq install libgomp1 -y && apt-get install -y wget && apt-get install curl -y

# elastix

RUN mkdir /opt/elastix && \
	curl -SsL https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2 | tar -C /opt/elastix -xvjf -

ENV PATH=/opt/elastix/elastix-5.0.1-linux/bin:$PATH

ENV LD_LIBRARY_PATH=/opt/elastix/elastix-5.0.1-linux/lib

# ilastik
# Dockerfile adapted from labsyspharm/mcmicro-ilastik

ARG ILASTIK_BINARY=ilastik-1.4.0b27-Linux.tar.bz2
RUN mkdir /opt/ilastik && \
	curl -SsL "https://files.ilastik.org/${ILASTIK_BINARY}" | tar -C /opt/ilastik -xvjf - --strip-components=1

# miniconda

ENV CONDA_DIR=/opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	/bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

# Set working directory for the project
WORKDIR /app
 
# Create Conda environment from the YAML file
COPY environment.yml .
RUN conda env create -f environment.yml
 
# Override default shell and use bash
SHELL ["conda", "run", "-n", "miaaim-dev", "/bin/bash", "-c"]

# install development version of miaaim
RUN python -m pip install .

# install jupyter
# RUN conda install jupyter -y
RUN python -m pip install jupyterlab

