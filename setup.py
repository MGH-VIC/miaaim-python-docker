# Python MIAAIM implementation
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import setuptools
import setuptools
import os

# Get the readme
def readme():
    with open("README.md", encoding="UTF-8") as readme_file:
        return readme_file.read()

# # Get the readme
# def readme():
#     with open(os.path.abspath("./app/README.md"), encoding="UTF-8") as readme_file:
#         return readme_file.read()
#


# Pip configuration
configuration = {
    "name": "miaaim-python",
    "version": "0.0.2",
    "description": "Multi-omics Image Alignment and Analysis by Information Manifolds",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    "keywords": "high-dimensional imaging mulitplex manifold-learning, cobordism-learning",
    "url": "https://github.com/JoshuaHess12/miaaim-python",
    "maintainer": "Joshua Hess",
    "maintainer_email": "joshmhess12@gmail.com",
    "license": "MIT",
    "packages": setuptools.find_packages(),
    "install_requires":[
              'scikit-image>=0.17.2',
              'numpy>=1.19.5',
              'pandas>=1.1.5',
              'pyimzML>=1.4.1',
              'nibabel>=3.2.1',
              'scipy>=1.5.4',
              'h5py>=3.1.0',
              "pathlib",
              "umap-learn>=0.5.1",
              "uncertainties",
              "seaborn",
              "matplotlib",
              "scikit-learn",
              "PyYAML"
          ]
}

# Apply setup
setuptools.setup(**configuration)
