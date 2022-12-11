# YAML parsing and implementation of Ilastik
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path
import logging

# Import custom modules
#from HDIprep import hdi_prep
from miaaim.seg.ilastik import _ilastik

# pars = "/Users/joshuahess/Desktop/miaaim-development-TMA/01_IMCextractCores/yaml/01_ROI012.yaml"

# Define parsing function
def RunIlastikYAML(pars,force_export):
    """Parsing YAML file for Ilastik pixel classification workflow.
    """

    # Ensure the path is a pathlib object
    path_to_yaml = Path(pars)

    # Open the yaml file
    with open(path_to_yaml, "r") as stream:
        # Try to load the yaml
        try:
            # Load the yaml file
            yml = yaml.full_load(stream)
            logging.info(yml)
        # Throw exception if it fails
        except yaml.YAMLError as exc:
            # Print error
            logging.info(exc)

    # Use the import options in the yml object to import all datasets
    il = _ilastik.Ilastik(**yml["ImportOptions"])

    # Iterate through each step
    for s in range(len(yml["ProcessingSteps"])):
        # Get the step -- either a string (if no extra input arguments, or a dictionary with key and value)
        step = yml["ProcessingSteps"][s]

        # Check if the step has input arguments
        if isinstance(step, dict):
            # Get the key value
            step = list(yml["ProcessingSteps"][s].keys())[0]
            # If this is a dictionary and is export nifti, add output dir
            if (step == "PrepareTraining") and (not force_export):
                # pass the step
                continue
            # Apply the processing step
            getattr(il, step)(**yml["ProcessingSteps"][s][step])

        # Otherwise raise an exception
        else:
            raise (Exception(f'{step} is not a valid processing step.'))
