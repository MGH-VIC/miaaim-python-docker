# Command line implementation for HDIprep module using YAML files
# Developer: Joshua M. Hess, BSc
# Developed at the Vaccine & Immunotherapy Center, Mass. General Hospital

# Import custom modules
from miaaim.cli.proc import _parse
from miaaim.proc import _yaml_proc

# Parse the command line arguments
args = _parse.ParseCommandYAML()

# Run the MultiExtractSingleCells function
_yaml_proc.RunHDIprepYAML(**args)
