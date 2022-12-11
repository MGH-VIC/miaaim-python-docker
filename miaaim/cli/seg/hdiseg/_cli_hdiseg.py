# Command line implementation for HDIsegmentation module using YAML files
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import custom modules
from miaaim.cli.seg.hdiseg import _parse
from miaaim.seg.hdiseg import _yaml_hdiseg

# Parse the command line arguments
args = _parse.ParseHDIsegYAML()

# Run the RunHDIsegmentationYAML function
_yaml_ilastik.RunHDIsegmentationYAML(**args)
