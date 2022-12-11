# YAML parsing and implementation of CellProfiler based segmentation
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path
import logging

# Import custom modules
#from HDIprep import hdi_prep
from miaaim.seg.hdiseg import hdiseg


# Define parsing function
def RunHDIsegmentationYAML(pars,force_export):
    """Parsing YAML file for HDISegmentation workflow.
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
    il = hdiseg.HDISegmentation(**yml["ImportOptions"])

    # Iterate through each step
    for s in range(len(yml["ProcessingSteps"])):
        # Get the step -- either a string (if no extra input arguments, or a dictionary with key and value)
        step = yml["ProcessingSteps"][s]

        # Check if the step has input arguments
        if isinstance(step, dict):
            # Get the key value
            step = list(yml["ProcessingSteps"][s].keys())[0]
            # Apply the processing step
            getattr(il, step)(**yml["ProcessingSteps"][s][step])

        # Otherwise raise an exception
        else:
            raise (Exception(f'{step} is not a valid processing step.'))
