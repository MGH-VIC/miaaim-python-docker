# Command line implementation for HDIprep module using YAML files
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import custom modules
from miaaim.cli.probs.ilastik import _parse
from miaaim.probs.ilastik import _yaml_ilastik

# Parse the command line arguments
args = _parse.ParseIlastikYAML()

# Run the MultiExtractSingleCells function
_yaml_ilastik.RunIlastikYAML(**args)
