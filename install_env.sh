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
conda install -c -y bioconda pyimzml
conda install -c -y conda-forge pathlib
conda install -c -y conda-forge uncertainties
conda install -c -y conda-forge pyyaml
conda install -c -y conda-forge tensorflow
conda install -c -y conda-forge tensorflow-probability
conda install -c -y conda-forge umap-learn
conda install -c -y conda-forge imagecodecs
# conda install -c bioconda cellprofiler
conda install -c conda-forge spyder
conda install -y dask
conda install -y ipykernel
conda install -y pip
pip install -y centrosome
# pip install CellProfiler
python -m pip install -e .
