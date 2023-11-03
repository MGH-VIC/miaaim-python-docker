# YAML parsing and implementation of HDIprep module
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import external modules
import yaml
from pathlib import Path

# Import custom modules
from miaaim.reg import _elastix

# Define parsing function
def RunElastixYAML(fixed, moving, path_to_yaml, out_dir):
    """Parsing YAML file to feed into elastix. Subsequent
    processing of files based on input parameters

    path_to_yaml: Path to .yaml file to parse that includes steps for processing
    """

    # Ensure the path is a pathlib object
    path_to_yaml = Path(path_to_yaml)

    # Open the yaml file
    with open(path_to_yaml, "r") as stream:
        # Try to load the yaml
        try:
            # Load the yaml file
            yml = yaml.full_load(stream)
            print(yml)
        # Throw exception if it fails
        except yaml.YAMLError as exc:
            # Print error
            print(exc)

    # Use the import options in the yml object to import all datasets
    _elastix.Elastix(fixed=fixed, moving=moving, out_dir=out_dir, **yml)
