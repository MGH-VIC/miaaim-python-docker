# Command line implementation for HDIsegmentation module using YAML files
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import custom modules
from miaaim.cli.quant import _parse
from miaaim.quant import _yaml_quantification

# Parse the command line arguments
args = _parse.ParseHDIquantYAML()

# Run the RunHDIsegmentationYAML function
_yaml_quantification.RunQuantificationYAML(**args)
