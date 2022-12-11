#!/bin/bash
mkdir -p miniconda3/miaaim-dev
conda create -p miniconda3/miaaim-dev
source activate miniconda3/miaaim-dev
conda install -y numpy
conda insatll -y numba
conda install -y matplotlib
conda install -y pandas
conda install -y scipy
conda isntall scikit-learn
conda install -c bioconda pyimzml
conda install -c conda-forge pathlib
conda install -c conda-forge uncertainties
conda install -c conda-forge pyyaml
conda install -c conda-forge umap-learn
conda install -c conda-forge imagecodecs
# conda install -c bioconda cellprofiler
conda install -c conda-forge spyder
conda install dask
conda install ipykernel
conda install pip
pip install centrosome
# pip install CellProfiler
python -m pip install -e .
