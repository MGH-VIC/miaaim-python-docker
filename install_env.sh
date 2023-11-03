#!/bin/bash
mkdir -p miniconda3/miaaim-dev
conda create -p miniconda3/miaaim-dev
source activate miniconda3/miaaim-dev
conda install -y numpy
conda insatll -y numba
conda install -y matplotlib
conda install -y pandas
conda install -y scipy
conda install scikit-learn
conda install scikit-image
conda install -c -y bioconda pyimzml
conda install -c -y conda-forge pathlib
conda install -c -y conda-forge uncertainties
conda install -c -y conda-forge pyyaml
conda install -c conda-forge tensorflow
conda install -c conda-forge squidpy
# conda install -c -y conda-forge tensorflow-probability
conda install -c -y conda-forge umap-learn
# conda install -c bioconda cellprofiler
# conda install -c conda-forge leidenalg
conda install -c conda-forge spyder
conda install -y dask
conda install -y ipykernel
conda install -y pip
python -m pip install centrosome
# conda install -c -y conda-forge imagecodecs
# pip install CellProfiler
pip install --upgrade nbconvert
python -m pip install -e .
