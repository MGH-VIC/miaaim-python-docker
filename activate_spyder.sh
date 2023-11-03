#!/bin/bash
source activate "$PWD/miniconda3/miaaim-dev"  &&
python -m ipykernel install --user --name miaaim-dev &&
spyder
